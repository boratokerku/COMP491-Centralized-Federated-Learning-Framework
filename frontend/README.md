# frontend/

Next.js web application (WP6): public project site plus an authenticated internal area
(live round view, update inbox, logs, client compatibility status).

Not scaffolded yet. To start:

```bash
npx create-next-app@latest . --ts --eslint --app --src-dir
```

The internal area should read data through read-only gateway/API endpoints, not by connecting
to PostgreSQL directly.
