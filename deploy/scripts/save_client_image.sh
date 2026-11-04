#!/bin/sh
# Package the client image as a file for sites that cannot pull from GitHub Container
# Registry, plus its SHA-256 so the site can verify what it received:
#
#   deploy/scripts/save_client_image.sh ghcr.io/boratokerku/cfl-client:0.1.0
#   # at the site:
#   sha256sum -c cfl-client-0.1.0.tar.gz.sha256 && gunzip -c cfl-client-0.1.0.tar.gz | docker load
#
# On a cluster that only allows Apptainer/Singularity, build an image from the same archive:
#   apptainer build cfl-client.sif docker-archive://cfl-client-0.1.0.tar
set -eu
image="${1:?usage: save_client_image.sh <image:tag>}"
tag="${image##*:}"
out="cfl-client-${tag}.tar.gz"
docker image inspect "$image" >/dev/null 2>&1 || docker pull "$image"
docker save "$image" | gzip -9 > "$out"
if command -v sha256sum >/dev/null; then sha256sum "$out" > "$out.sha256"; else shasum -a 256 "$out" > "$out.sha256"; fi
echo "wrote $out ($(du -h "$out" | cut -f1)) and $out.sha256"
echo "image id: $(docker image inspect --format '{{.Id}}' "$image")"
