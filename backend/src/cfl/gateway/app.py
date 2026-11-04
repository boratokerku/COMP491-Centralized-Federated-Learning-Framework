"""Gateway app: the only entry point for clients (planning/api-spec.md, section 3). TLS is
terminated by Caddy in front of it (deploy/). Every client endpoint requires a bearer token
(cfl.gateway.auth); `/health` does not.

`app` connects to the database named by CFL_DATABASE_URL on first use, so importing this module
needs no database; tests and tools build their own app with create_app(engine).
"""

from __future__ import annotations

from functools import cache

from fastapi import FastAPI, Request
from sqlalchemy import Engine, text

from cfl import __version__
from cfl.db.state_store import engine_from_env
from cfl.gateway.auth import GatewayError, error_response
from cfl.gateway.clients import ClientRegistry
from cfl.protocol import PROTOCOL_VERSION


def create_app(engine: Engine | None = None) -> FastAPI:
    app = FastAPI(title="CFL Gateway", version=__version__)

    @cache
    def db() -> Engine:
        return engine if engine is not None else engine_from_env()

    @cache
    def registry() -> ClientRegistry:
        return ClientRegistry(db())

    app.state.db = db
    app.state.registry = registry

    @app.exception_handler(GatewayError)
    async def _gateway_error(request: Request, exc: GatewayError):
        return error_response(exc)

    @app.get("/health")
    def health() -> dict:
        try:
            with db().connect() as conn:
                conn.execute(text("SELECT 1"))
            database = "ok"
        except Exception:  # unconfigured or unreachable; health must still answer
            database = "unavailable"
        return {
            "status": "ok",
            "version": __version__,
            "protocol_version": PROTOCOL_VERSION,
            "database": database,
        }

    return app


app = create_app()
