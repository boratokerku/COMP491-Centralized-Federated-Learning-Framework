# Team Interfaces and Decisions

**Status:** decided (2026-11-04, Bora Toker, with the team's authority). Change only by updating this file.

Who owns what, and the contracts between the parts, so that each person can work without waiting for the others.

## 1. Ownership

| Area | Owner |
|---|---|
| Aggregator, coordinator, orchestration, resume, ONNX final model | Bora |
| Gateway (FastAPI client endpoints, tokens, conformance re-check), HTTPS transport | Bora |
| Shared client runtime (polling, retries, pending-update persistence, conformance report, final-model verification) | Bora |
| Docker images, compose, Caddy, cloud VM deployment | Bora |
| Telecom client: text `TaskSpec`, tuning, baselines, text mismatch variants | Merve |
| Bank client: tabular `TaskSpec`, tuning, baselines, independent-implementation client and mismatch variants | Enes |
| API specification (`planning/api-spec.md`) and review of the gateway against it | Enes |
| Website (public + internal area), web read-only API, Grafana dashboards, end-of-run report | Hilal |
| Experiments, final report, poster, demo video | Everyone |

## 2. `TaskSpec`: what each client team implements

Defined in `backend/src/cfl/core/task.py`. The shared client runtime (`backend/src/cfl/client/runtime.py`) calls it; everything else (registration, conformance report, polling, download, upload, retries, crash-safe resending, final model) is the runtime's job. Reference implementation for both tasks: `backend/src/cfl/client/reference.py` (`ReferenceTaskSpec`), used when a site sets no `CFL_TASK_SPEC`; own implementations are selected with `CFL_TASK_SPEC=module:Class` and constructed without arguments. `python -m cfl.client conformance --manifest <file>` checks an implementation against a manifest offline (deploy/README.md); the conformance report itself is built and submitted by the runtime.

| Method | Contract |
|---|---|
| `build_model()` | The run's reference architecture (`bank_mlp` or `sms_hash_embed`), unchanged parameter names. |
| `preprocess(raw: dict) -> list[float \| int]` | One raw record (manifest `synthetic_raw_input`) to one model input row. Used for the manifest reference tests, so it must be the same code that prepares training data. |
| `load_split(path) -> (inputs, labels)` | One partition CSV (`train.csv`, `val.csv`, `test.csv`) to tensors; labels are 0/1 with 1 = positive (subscribed / spam). |
| `train(model, data, config, global_weights, seed) -> metrics` | Train in place. Use exactly the round config (section 3). Must return `train_loss`. Must be deterministic for a given `seed`. |
| `evaluate(model, data) -> metrics` | Return the keys of `cfl.core.evaluation.classification_metrics`: `loss`, `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`. |

## 3. Round config (`TaskAssignment.config`)

Produced by `Coordinator.round_config`; clients must not add their own hyperparameters.

| Key | Meaning |
|---|---|
| `strategy` | `fedavg` or `fedprox` |
| `local_epochs`, `batch_size`, `learning_rate`, `weight_decay` | Local training settings (Adam optimizer) |
| `proximal_mu` | FedProx coefficient; `0.0` for FedAvg. Add `(mu/2)·‖w − w_global‖²` to the loss when > 0 (`cfl.core.strategy.proximal_term`). |
| `class_weights` | `[w_negative, w_positive]` for the cross-entropy loss, fixed for the whole federation |

## 4. Metric names

All metrics end up in the `metrics` table (one row per value). Names and units are fixed so the website, Grafana and the report can rely on them.

**Client → `UpdateMetadata.metrics`** (stored with the `client_` prefix and the client's id):

| Name | Unit | Meaning |
|---|---|---|
| `train_loss` | value | Mean training loss of the round |
| `train_s` | seconds (stored as `client_train_ms`) | Local training time |
| `val_<metric>_before` | value | `evaluate()` on the local `val` split with the received global model |
| `val_<metric>_after` | value | `evaluate()` on the local `val` split after local training |

**Coordinator** (per round, no client id): `global_<metric>` (held-out test set), `updates_received`, `updates_used`, `updates_rejected`, `total_samples` (count), `bytes_read`, `model_bytes` (bytes), `load_ms`, `aggregation_ms`, `evaluation_ms`, `write_ms`, `waiting_ms`, `round_duration_ms` (ms).

**Gateway** (per request, with client id and `message_type` ∈ `model_download`, `update_upload`, `final_model_download`): `gateway_ms` (ms, time inside the gateway, without the network), `message_bytes` (bytes).

**Client → `POST /transfer`** (with client id and `message_type`, the three above plus `register`, `conformance` and `task`): `transfer_ms` (ms, the transfer as the client observed it, network included), and `message_bytes` for those three control messages. `register` and `conformance` arrive with the first task of a run, `task` once per round; empty polls are not reported. In `UpdateMetadata.metrics`: `rtt_ms` (ms, from `GET /health` round trips), `retry_count` (count, retries since the previous update).

## 5. Conformance report

The payload format is the "Report payload contract" in `planning/conformance-manifest.md`, plus `forward_outputs: [{test_id, logits}]` for the manifest's `forward_reference_tests`. The independent-implementation client (Enes) must produce exactly this JSON; the shared runtime produces it for managed clients. The gateway re-checks every value; `client_passed` is never trusted.

## 6. Deployment and the website

- One cloud VM, Caddy in front, Let's Encrypt certificates, only ports 80/443 (and 22) open.
- Two hostnames on the same IP (`sslip.io`): `fl.<ip>.sslip.io` → gateway (client API only), `<ip>.sslip.io` → website, with Grafana under `/grafana/`. Clients use port 443, not 8443, because institutional networks often block other ports.
- The website is a Next.js service in the same compose file. Its read-only API (`/api/v1/...` in `api-spec.md` section 4) is implemented inside the website (Next.js route handlers), not in the gateway, so the internet-facing client API stays minimal. The website connects to PostgreSQL with a **read-only role** (`cfl_web`, `SELECT` only) and reads report files from the model volume mounted read-only.
- Internal area login: a handful of accounts (team and advisors), passwords stored as Argon2/bcrypt hashes, httpOnly session cookies. Client tokens are never reused for website logins.
- The public area never shows client identities, update details or event logs of real runs; see section 8.

## 7. Client tokens

- `cfl-admin add-client <id>` approves a client and prints its token once (`--token-file` writes it to a mode-600 file instead). The site stores it as its Docker secret; it is never committed, baked into an image or sent anywhere else.
- Format `cfl_<43 url-safe characters>` (256 random bits). The `clients.token_hash` column holds its SHA-256. This deliberately differs from the Argon2/bcrypt example in `api-spec.md` section 2.2: slow salted hashes protect guessable passwords, while these tokens cannot be guessed, and an unsalted hash lets the gateway find the client with one indexed lookup. GitHub stores its API tokens the same way.
- Every client endpoint requires `Authorization: Bearer <token>`: missing or unknown token → 401 (`missing_token`, `invalid_token`), disabled client → 403 (`client_disabled`). Each authenticated request updates `clients.last_seen_at`.
- `rotate-token` replaces a leaked token immediately; `disable` / `enable` block and unblock a client. Every administrative action is recorded in `events` (`CLIENT_APPROVED`, `CLIENT_TOKEN_ROTATED`, `CLIENT_DISABLED`, `CLIENT_ENABLED`).

## 8. Public "how it works" demonstration

The public site shows the training process without exposing a live system:

- An animated replay of a completed demo run, driven by its exported report (`summary.json`): rounds advancing, clients reporting, a dropout and the coordinator restart, the global F1/ROC-AUC curve. Static data, so it always works and cannot be abused.
- Optionally a read-only live status of one designated public demo run (round number, state, global metric curve only), served by the website from the read-only role.
- Visitors cannot start runs or reach the gateway from the site.

## 9. Gateway protocol

`planning/api-spec.md` (version 1.1.0, maintainer: Enes) is the authoritative description of the client API and is kept in line with `backend/src/cfl/gateway/app.py`. Its "Changes in 1.1.0" line lists the decisions taken while implementing the gateway: manifest identity `"<dataset>@<version>"`, `GET /conformance`, the extended conformance report, registration before every other endpoint, participant-only model downloads, final-model `format` values, idempotent retries before stale checks, SHA-256 token hashes, protocol `0.1`, port 443, and the web API living in the website.
