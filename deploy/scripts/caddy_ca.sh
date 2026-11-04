#!/bin/sh
# Export the root certificate of Caddy's local CA (used for *.localhost hostnames, or with
# `tls internal`), so test clients can trust the gateway: CFL_CA_BUNDLE=<this file>.
# Not needed with public hostnames: Let's Encrypt certificates are trusted everywhere.
set -eu
out="${1:-deploy/client/ca/root.crt}"
mkdir -p "$(dirname "$out")"
docker compose -f deploy/docker-compose.yml exec -T caddy \
    cat /data/caddy/pki/authorities/local/root.crt > "$out"
echo "wrote $out"
