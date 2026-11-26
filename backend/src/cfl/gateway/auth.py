"""Bearer-token authentication for every client endpoint (planning/api-spec.md, section 2.2).

Use the `CurrentClient` dependency on a route; the request is rejected before the route runs
when the token is missing or unknown (401) or the client is disabled (403). Errors use the
spec's error schema: {"error": {"code", "message", "details", "timestamp"}}.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Annotated

from fastapi import Depends, Request
from fastapi.responses import JSONResponse

from cfl.gateway.clients import AuthError, ClientRecord, ClientRegistry


def _extract_cert_cn(subject: str) -> str | None:
    """Extract CN from Caddy's RFC 2253 `X-Client-Cert-Subject` header (e.g. 'CN=bank-a,O=CFL')."""
    for part in subject.split(","):
        key, sep, val = part.strip().partition("=")
        if sep and key.strip().upper() == "CN":
            return val.strip()
    return None


def _require_mtls() -> bool:
    return os.environ.get("CFL_REQUIRE_MTLS", "").strip().lower() in ("1", "true", "yes")


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
    if _require_mtls():
        subject = request.headers.get("x-client-cert-subject", "").strip()
        cn = _extract_cert_cn(subject)
        if not cn:
            raise GatewayError(
                403, "missing_client_cert", "mTLS client certificate is required by this gateway"
            )
        if cn != client.client_id:
            raise GatewayError(
                403,
                "client_cert_mismatch",
                "mTLS certificate CN does not match the bearer token client_id",
                {"cert_cn": cn, "token_client_id": client.client_id},
            )
    request.state.client_id = client.client_id
    return client


CurrentClient = Annotated[ClientRecord, Depends(current_client)]
