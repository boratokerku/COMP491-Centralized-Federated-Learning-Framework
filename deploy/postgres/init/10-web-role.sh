#!/bin/sh
# Runs once, when the PostgreSQL volume is first created. Creates `cfl_web`, the read-only
# role used by the website and Grafana: it can read every table the schema owner creates,
# and migration 0002 hides the clients' token hashes from it.
set -eu
psql -v ON_ERROR_STOP=1 -v pw="$CFL_WEB_DB_PASSWORD" \
     --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<'SQL'
CREATE ROLE cfl_web LOGIN PASSWORD :'pw';
SELECT format('GRANT CONNECT ON DATABASE %I TO cfl_web', current_database()) \gexec
GRANT USAGE ON SCHEMA public TO cfl_web;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO cfl_web;
SQL
