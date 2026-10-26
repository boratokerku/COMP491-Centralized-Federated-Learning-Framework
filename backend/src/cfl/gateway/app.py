"""Gateway app. Endpoints (register, poll task, download model, upload update) follow the
API spec from WP1; TLS is terminated by Caddy in front of this app (see deploy/)."""

from __future__ import annotations

from fastapi import FastAPI

from cfl import __version__
from cfl.protocol import PROTOCOL_VERSION

app = FastAPI(title="CFL Gateway", version=__version__)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__, "protocol_version": PROTOCOL_VERSION}
