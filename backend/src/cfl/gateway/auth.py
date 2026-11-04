"""Bearer-token authentication for every client endpoint (planning/api-spec.md, section 2.2).

Use the `CurrentClient` dependency on a route; the request is rejected before the route runs
when the token is missing or unknown (401) or the client is disabled (403). Errors use the
spec's error schema: {"error": {"code", "message", "details", "timestamp"}}.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, Request
from fastapi.responses import JSONResponse

from cfl.gateway.clients import AuthError, ClientRecord, ClientRegistry


class GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}


def error_response(error: GatewayError) -> JSONResponse:
    headers = {"WWW-Authenticate": 'Bearer realm="cfl"'} if error.status == 401 else None
    return JSONResponse(
        status_code=error.status,
        content={
            "error": {
                "code": error.code,
                "message": error.message,
                "details": error.details,
                "timestamp": datetime.now(UTC).isoformat(),
            }
        },
        headers=headers,
    )


def get_registry(request: Request) -> ClientRegistry:
    return request.app.state.registry()


def current_client(
    request: Request, registry: Annotated[ClientRegistry, Depends(get_registry)]
) -> ClientRecord:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise GatewayError(401, "missing_token", "send Authorization: Bearer <client token>")
    try:
        client = registry.authenticate(token.strip())
    except AuthError as exc:
        raise GatewayError(exc.status, exc.code, exc.message) from exc
    request.state.client_id = client.client_id
    return client


CurrentClient = Annotated[ClientRecord, Depends(current_client)]
