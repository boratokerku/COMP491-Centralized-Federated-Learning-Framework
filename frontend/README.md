# frontend/

Next.js web application (WP6, owner: Hilal): public project site plus a login-protected internal
area (live round view, update inbox, event log, metrics panels, results, human-readable reports,
client compatibility status). Decisions: `planning/interfaces.md`, sections 6 and 8.

Scaffolded with Next.js (TypeScript, App Router, Tailwind). The public area is in place:
`/` (project home and weekly progress), `/docs`, `/results`, `/team`. The internal area (`/app`)
is still to do. Page structure: `planning/scope.md`, section 4.1.

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
