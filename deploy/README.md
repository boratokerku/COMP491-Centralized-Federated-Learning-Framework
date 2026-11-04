# Deployment

Everything runs in containers built from `backend/Dockerfile` (`server` and `client` targets) and, for the website, `frontend/Dockerfile`. Decisions behind this layout: `planning/interfaces.md`, section 6.

```
            Internet (443)                         compose network (not published)
 clients ──► fl.<ip>.sslip.io ─┐                ┌─► gateway:8000  ─┐
 browsers ─► <ip>.sslip.io ────┴─► caddy (TLS) ─┼─► web:3000       ├─► postgres:5432
                                                └─► grafana:3000   │   models volume
                                                    coordinator ───┘
```

| Service | Image | Role |
|---|---|---|
| `postgres` | `postgres:16` | State (runs, rounds, updates, events, metrics). Creates the read-only `cfl_web` role on first start. |
| `migrate` | `cfl-server` | `alembic upgrade head`, then exits; everything else waits for it. |
| `coordinator` | `cfl-server` | `python -m cfl.coordinator`: resumes running runs, drives the rounds. Safe to stop at any time. |
| `gateway` | `cfl-server` | Client API (FastAPI). |
| `web` | `cfl-web` | Website, optional profile `web` (Hilal). |
| `grafana` | `grafana/grafana` | Dashboards on the read-only role, served under `/grafana/`. |
| `caddy` | `caddy:2.8` | Only service with published ports. Let's Encrypt for public hostnames, its own CA for `*.localhost`. |

## Server: local test

```bash
cp deploy/.env.example deploy/.env                 # set the four passwords
docker compose -f deploy/docker-compose.yml up -d --build
curl -k --resolve fl.localhost:443:127.0.0.1 https://fl.localhost/health
```

If ports 80/443 (or 5432) are taken, set `CFL_HTTP_PORT`, `CFL_HTTPS_PORT` (`CFL_POSTGRES_PORT`) in `deploy/.env`.

## Server: cloud VM

1. Create a small VM (2 vCPU, 4 GB RAM, Ubuntu) with a public IP. Firewall: allow 22, 80, 443 (TCP) and 443 (UDP) only.
2. Install Docker Engine and the compose plugin; clone the repository.
3. In `deploy/.env`: the passwords, and for public IP `203.0.113.7`:
   `CFL_GATEWAY_DOMAIN=fl.203-0-113-7.sslip.io`, `CFL_WEB_DOMAIN=203-0-113-7.sslip.io`.
4. Copy only the held-out test sets to the VM (`data/<modality>/partitions/<dataset>/test.csv`); client partitions never go to the server.
5. `docker compose -f deploy/docker-compose.yml --profile web up -d --build`. Caddy obtains the certificates on first start.
6. Back up the `pgdata` and `models` volumes (e.g. a nightly `pg_dump` and `tar` of the models volume).

## Starting a run

The coordinator service keeps running; a new run is created with a one-off container and picked up automatically:

```bash
docker compose -f deploy/docker-compose.yml run --rm coordinator \
    python -m cfl.coordinator --models /models --start configs/demo_tabular.yaml --no-serve
docker compose -f deploy/docker-compose.yml logs -f coordinator
```

`configs/demo_*.yaml` are short distributed runs for demonstrations; `configs/sim_*.yaml` are for `run_simulation.py`.

## Client sites

The administrator approves each site and hands it its token out of band (shown once; only its hash is stored):

```bash
docker compose -f deploy/docker-compose.yml run --rm gateway cfl-admin add-client bank-a
docker compose -f deploy/docker-compose.yml run --rm gateway cfl-admin list
# rotate-token <id> after a leak, disable <id> / enable <id> to block or unblock a site
```

Each participating organization runs the `cfl-client` image on its own machine and network (`deploy/client/`):

```bash
cd deploy/client
cp client.env.example client.env          # server URL, client id, task, partition folder
printf '%s' '<token>' > token.txt && chmod 600 token.txt
docker compose --env-file client.env run --rm client check   # setup + gateway reachable over HTTPS
docker compose --env-file client.env up -d
```

### Without Docker

The same client runs directly with Python (e.g. on a team member's laptop or a cluster node): install the backend (`pip install -e backend`), copy `deploy/client/client.yaml.example` to `client.yaml`, fill it in and run

```bash
cfl-client check --config client.yaml
cfl-client run --config client.yaml
```

Every setting can also be given as a command-line option or a `CFL_*` environment variable; precedence is option > environment > config file > default (`cfl-client --help` lists them). Relative paths in the config file are relative to the file. The token is only ever read from `token_file`, never passed as a value.

### What the commands do

`check` verifies the token file, the partition files, the task, the state directory and the TLS connection to the gateway, and exits non-zero on any problem. The partition is mounted read-only; the token is a Docker secret, never an environment variable or part of the image.

`run` (the default command) is the shared client runtime (`backend/src/cfl/client/runtime.py`): it registers, passes the conformance check for its dataset (waiting until a run uses it), then polls for tasks, trains, uploads and, when a run it took part in has finished, downloads and verifies the final model into `/state/final/<run_id>/`. Every update is written to `/state` before it is uploaded and removed once the gateway has decided, so a site that crashes or loses its connection resends the same `update_id` after a restart and is counted once. Keep the `state` volume across restarts. Optional settings: `CFL_POLL_INTERVAL` (seconds, default 3), `CFL_SEED` (local training seed, default 0); `run --exit-after-final` stops after the final model of the latest run the site trained in (for demos).

## Publishing images (managed-code mode)

A version tag builds and pushes both images to GitHub Container Registry (`.github/workflows/images.yml`):

```bash
git tag v0.1.0 && git push origin v0.1.0
# ghcr.io/boratokerku/cfl-server:0.1.0, ghcr.io/boratokerku/cfl-client:0.1.0 (+ :latest)
```

The job summary lists the image digests; sites should pin `cfl-client:<version>@sha256:<digest>`. Sites without registry access get the archive (also attached to the workflow run):

```bash
deploy/scripts/save_client_image.sh ghcr.io/boratokerku/cfl-client:0.1.0
# at the site: sha256sum -c cfl-client-0.1.0.tar.gz.sha256 && gunzip -c cfl-client-0.1.0.tar.gz | docker load
```

University clusters that only allow Apptainer can build from the same archive: `apptainer build cfl-client.sif docker-archive://cfl-client-0.1.0.tar` (not yet tried on the cluster).

## Where each team member plugs in

- **Client code (Merve: text, Enes: tabular).** Managed-code mode means the training code is part of the image, not mounted at run time. Implement `TaskSpec` (`backend/src/cfl/core/task.py`) in the repository, e.g. `backend/src/cfl/tasks/text/spec.py`; the next image build contains it. Sites select it with `CFL_TASK_SPEC=cfl.tasks.text.spec:SmsTaskSpec` (the class is constructed without arguments), and `check` confirms it imports. While developing, test it against the manifest without any server:

  ```bash
  cd backend
  CFL_TASK=text CFL_TASK_SPEC=cfl.tasks.text.spec:SmsTaskSpec \
      python -m cfl.client conformance --manifest ../planning/examples/sms_spam_manifest.json
  ```

  It runs every reference test (your `preprocess` on the synthetic raw inputs) and forward test (your model under the reference weights), checks the parameter names and shapes, shows the largest difference and where it is, and prints the decision the gateway would take; exit code 0 means conformant. Without `--manifest` it fetches the manifest of the running run from the gateway (same settings as `check`) and still submits nothing. The runtime builds and submits the real report itself (`cfl.conformance.report.build_report`), so nothing else is needed from you. Without `CFL_TASK_SPEC` the image uses `cfl.client.reference.ReferenceTaskSpec`, the reference behaviour and a starting point for your own implementation. New Python dependencies go into `backend/pyproject.toml`.
- **Independent-implementation client (Enes).** Not a `cfl-client` image: it talks to the gateway's REST API directly and only has to follow `planning/api-spec.md` and the conformance report format.
- **Website (Hilal).** Add `frontend/Dockerfile` that serves the Next.js app on port 3000 and start the stack with `--profile web`. It receives `DATABASE_URL` (read-only role `cfl_web`), `REPORTS_DIR=/models` (read-only) and `PUBLIC_URL`. Grafana dashboards go into `deploy/grafana/provisioning/dashboards/` (provisioned on start).

## Security notes

- Only Caddy is reachable from outside; PostgreSQL listens on `127.0.0.1` for administration over SSH.
- All containers except Caddy and PostgreSQL run as the unprivileged user `cfl` (uid 10001).
- `cfl_web` can only `SELECT`, and cannot read `clients.token_hash` (migration 0002).
- Secrets live in `deploy/.env` and `deploy/client/token.txt`, both gitignored.
