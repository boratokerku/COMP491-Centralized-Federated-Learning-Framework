#!/bin/sh
# Generate an mTLS Client CA and per-client certificates for Caddy's `require_and_verify` mode.
#
# Usage:
#   deploy/scripts/mtls_certs.sh init                       # creates deploy/mtls/ca.{crt,key}
#   deploy/scripts/mtls_certs.sh client bank-a [out_dir]    # signs client cert with CN=bank-a
#
# Copy <out_dir>/bank-a.crt and <out_dir>/bank-a.key to the site (chmod 600 on the key) and set
# CFL_CLIENT_CERT / CFL_CLIENT_KEY on cfl-client. Never distribute ca.key.
set -eu

MTLS_DIR="${CFL_MTLS_DIR:-deploy/mtls}"
CA_CRT="$MTLS_DIR/ca.crt"
CA_KEY="$MTLS_DIR/ca.key"

cmd="${1:-}"
case "$cmd" in
  init)
    mkdir -p "$MTLS_DIR"
    if [ -f "$CA_CRT" ] || [ -f "$CA_KEY" ]; then
      echo "CA already exists in $MTLS_DIR; remove it first to regenerate" >&2
      exit 1
    fi
    openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
      -nodes -days 365 \
      -keyout "$CA_KEY" -out "$CA_CRT" \
      -subj "/O=CFL-Framework/CN=CFL-Client-CA" \
      -addext "basicConstraints=critical,CA:TRUE" \
      -addext "keyUsage=critical,keyCertSign,cRLSign"
    chmod 600 "$CA_KEY"
    echo "created mTLS Client CA: $CA_CRT"
    ;;
  client)
    client_id="${2:-}"
    if [ -z "$client_id" ]; then
      echo "usage: $0 client <client_id> [out_dir]" >&2
      exit 2
    fi
    if [ ! -f "$CA_CRT" ] || [ ! -f "$CA_KEY" ]; then
      echo "CA not found in $MTLS_DIR; run '$0 init' first" >&2
      exit 1
    fi
    out_dir="${3:-$MTLS_DIR/clients/$client_id}"
    mkdir -p "$out_dir"
    key="$out_dir/${client_id}.key"
    csr="$out_dir/${client_id}.csr"
    crt="$out_dir/${client_id}.crt"
    ext="$out_dir/${client_id}.ext"
    openssl req -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
      -nodes -keyout "$key" -out "$csr" \
      -subj "/O=CFL-Client/CN=${client_id}"
    printf 'basicConstraints=CA:FALSE\nkeyUsage=digitalSignature\nextendedKeyUsage=clientAuth\n' > "$ext"
    openssl x509 -req -in "$csr" -CA "$CA_CRT" -CAkey "$CA_KEY" -CAcreateserial \
      -out "$crt" -days 365 -extfile "$ext"
    rm -f "$csr" "$ext"
    chmod 600 "$key"
    echo "signed mTLS client cert: $crt (key: $key)"
    ;;
  *)
    echo "usage: $0 init | client <client_id> [out_dir]" >&2
    exit 2
    ;;
esac
