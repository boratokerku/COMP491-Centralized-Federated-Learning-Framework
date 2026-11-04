# syntax=docker/dockerfile:1
# Images for the federated learning framework. Build context: backend/
#
#   docker build --target server -t cfl-server:<version> backend   # gateway, coordinator, migrations
#   docker build --target client -t cfl-client:<version> backend   # managed-code client
#
# Both use CPU-only PyTorch (the CUDA wheels would add several GB) and run as a non-root user.
# No secrets, tokens or data are baked in: they come from environment variables, Docker secrets
# and mounted volumes at run time.

ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN groupadd --system --gid 10001 cfl && useradd --system --uid 10001 --gid cfl --home /app cfl

# Dependencies first, so source changes do not reinstall PyTorch.
COPY pyproject.toml ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && python - <<'EOF'
import subprocess, sys, tomllib
deps = tomllib.load(open("pyproject.toml", "rb"))["project"]["dependencies"]
deps = [d for d in deps if not d.startswith("torch")]
subprocess.check_call([sys.executable, "-m", "pip", "install", *deps])
EOF

# The package is installed in editable mode so it keeps the repository layout
# (src/, configs/, migrations/) that the code resolves paths against.
COPY src ./src
COPY configs ./configs
RUN pip install --no-deps -e .

ARG CFL_VERSION=0.0.0-dev
ARG VCS_REF=unknown
LABEL org.opencontainers.image.source="https://github.com/boratokerku/COMP491-Centralized-Federated-Learning-Framework" \
      org.opencontainers.image.version="${CFL_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}"
ENV CFL_VERSION=${CFL_VERSION}


FROM base AS server
COPY alembic.ini ./
COPY migrations ./migrations
# The server reads the held-out test sets from /srv/data (data_dir in the run config is resolved
# against CFL_REPO_ROOT) and keeps model files in /models.
ENV CFL_REPO_ROOT=/srv
RUN mkdir -p /models /srv/data && chown cfl:cfl /models
USER cfl
EXPOSE 8000
# Default: the gateway. The coordinator and the migration step override the command.
CMD ["uvicorn", "cfl.gateway.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]


FROM base AS client
# /data: this site's partition, mounted read-only. /state: pending updates and downloaded models
# that must survive a container restart. The token comes from a secret file, never an env var.
ENV CFL_CLIENT_VERSION=cfl-client/${CFL_VERSION} \
    CFL_PARTITION_DIR=/data \
    CFL_STATE_DIR=/state \
    CFL_TOKEN_FILE=/run/secrets/cfl_token
RUN mkdir -p /state && chown cfl:cfl /state
USER cfl
ENTRYPOINT ["python", "-m", "cfl.client"]
CMD ["run"]
