# frontend/

Next.js web application (WP6, owner: Hilal): public project site plus a login-protected internal
area (live round view, update inbox, event log, metrics panels, results, human-readable reports,
client compatibility status). Decisions: `planning/interfaces.md`, sections 6 and 8.

Scaffolded with Next.js (TypeScript, App Router, Tailwind). Public area: `/`, `/docs`, `/results`,
`/team` (page structure: `planning/scope.md`, section 4.1). Internal area under `/app`, behind a
login: live round view, runs (config, versions, final model, metric panels), update inbox, event
log, clients and conformance, reports and plots.

## Internal area

- **Login.** Accounts are `name:bcrypt-hash`. Set `CFL_WEB_USERS` and `CFL_WEB_SESSION_SECRET` in
  `deploy/.env` (see `deploy/.env.example`); alternatively point `WEB_USERS_FILE` at a secret
  file. Create a hash with
  `docker run --rm -it -v "$PWD/frontend":/app -w /app node:22-alpine node scripts/hash-password.mjs`.
  Sessions are signed httpOnly cookies (8 h); failed logins are limited to 5 per 10 minutes per
  address and user. Without users and a 32+ character secret the form is disabled.
- **Read only.** The database role `cfl_web` has `SELECT` only and cannot see
  `clients.token_hash`; report files come from the models volume mounted read-only. The
  browser never talks to PostgreSQL: the pages and the `/api/v1/...` route handlers
  (`planning/api-spec.md`, section 4) read it on the server, and every handler checks the session.
- **Live view.** `/app/rounds` renders on the server and then refreshes through
  `/api/v1/runs/{id}/rounds` every 3 seconds, with a deadline countdown.
- **Reports.** Run `cfl-report <run_id>` (backend) to write `<run_id>/report/` into the models
  volume; `/app/reports` shows `report.md` and the plots.
- `GET /api/auth/verify` answers 200 or 401 for the session cookie; Caddy uses it to protect Grafana.

Weekly updates: add an entry at the top of `updates` in `src/content/site.ts`.

Node is only needed for local development (`npm install && npm run dev`). The Docker build
does not need it on the host.

## Contract with the deployment (`deploy/docker-compose.yml`, service `web`)

- Provide a `Dockerfile` here that serves the app on port **3000** (e.g. Next.js `output: "standalone"`).
- The app reads data on the server side only, through its own route handlers (`/api/v1/...`,
  `planning/api-spec.md` section 4). It connects to PostgreSQL with the read-only role given in
  `DATABASE_URL` (`cfl_web`: `SELECT` only, no access to client token hashes). The browser never
  talks to PostgreSQL or to the gateway.
- Run reports and artifacts are under `REPORTS_DIR` (the models volume, mounted read-only).
- `PUBLIC_URL` is the site's own address. Grafana is served by Caddy under `/grafana/`.
- Start it with the rest of the stack: `docker compose -f deploy/docker-compose.yml --profile web up -d --build`.
