"""Approved clients and their bearer tokens (`clients` table).

An administrator approves a client with `cfl-admin add-client <id>`, which prints a token once;
the token is handed to the site out of band and never stored. The database keeps only its
SHA-256 hash.

Why SHA-256 rather than Argon2/bcrypt: slow, salted hashes protect low-entropy passwords against
guessing. These tokens are 256-bit random values that cannot be guessed, so a fast unsalted
hash is just as safe, and it lets the gateway find the client with one indexed lookup
(`token_hash` is UNIQUE) instead of trying a salted hash against every row. GitHub's API tokens
are stored the same way.
"""

from __future__ import annotations

import hashlib
import re
import secrets
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import Engine, insert, select, update
from sqlalchemy.exc import IntegrityError

from cfl.db import schema as t

TOKEN_PREFIX = "cfl_"
CLIENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,62}$")


class ClientStatus(StrEnum):
    APPROVED = "APPROVED"  # may connect; has not registered yet
    REGISTERED = "REGISTERED"  # has called POST /register
    DISABLED = "DISABLED"  # rejected with 403 on every request


class AuthError(Exception):
    """`status` is the HTTP status the gateway answers with; `code` the stable error code."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ClientRecord:
    client_id: str
    status: ClientStatus
    client_version: str | None
    protocol_version: str | None
    created_at: datetime
    registered_at: datetime | None
    last_seen_at: datetime | None


def new_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)  # 256 random bits


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _record(row) -> ClientRecord:
    return ClientRecord(
        client_id=row.client_id,
        status=ClientStatus(row.status),
        client_version=row.client_version,
        protocol_version=row.protocol_version,
        created_at=row.created_at,
        registered_at=row.registered_at,
        last_seen_at=row.last_seen_at,
    )


def _event(event_type: str, client_id: str, now: datetime, **details) -> dict:
    return {
        "event_type": event_type,
        "client_id": client_id,
        "occurred_at": now,
        "details": details,
    }


class ClientRegistry:
    def __init__(self, engine: Engine, clock=lambda: datetime.now(UTC)):
        self.engine = engine
        self.clock = clock

    # -- administration (cfl-admin) ----------------------------------------------------------
    def add_client(self, client_id: str) -> str:
        """Approve a new client and return its token. The token is not stored anywhere."""
        if not CLIENT_ID_PATTERN.match(client_id):
            raise ValueError(
                "client id must be 2-63 characters: lowercase letters, digits, '-' or '_'"
            )
        token = new_token()
        now = self.clock()
        try:
            with self.engine.begin() as conn:
                conn.execute(
                    insert(t.clients).values(
                        client_id=client_id,
                        token_hash=hash_token(token),
                        status=ClientStatus.APPROVED.value,
                        created_at=now,
                    )
                )
                conn.execute(insert(t.events).values(**_event("CLIENT_APPROVED", client_id, now)))
        except IntegrityError as exc:
            raise ValueError(f"client {client_id!r} already exists") from exc
        return token

    def rotate_token(self, client_id: str) -> str:
        """Replace the token (e.g. after a leak); the old one stops working immediately."""
        token = new_token()
        self._update(client_id, "CLIENT_TOKEN_ROTATED", token_hash=hash_token(token))
        return token

    def disable(self, client_id: str) -> None:
        self._update(client_id, "CLIENT_DISABLED", status=ClientStatus.DISABLED.value)

    def enable(self, client_id: str) -> None:
        client = self.get(client_id)
        status = ClientStatus.REGISTERED if client.registered_at else ClientStatus.APPROVED
        self._update(client_id, "CLIENT_ENABLED", status=status.value)

    def get(self, client_id: str) -> ClientRecord:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(t.clients).where(t.clients.c.client_id == client_id)
            ).first()
        if row is None:
            raise KeyError(client_id)
        return _record(row)

    def list(self) -> list[ClientRecord]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(t.clients).order_by(t.clients.c.client_id)).all()
        return [_record(r) for r in rows]

    def _update(self, client_id: str, event_type: str, **values) -> None:
        now = self.clock()
        with self.engine.begin() as conn:
            result = conn.execute(
                update(t.clients).where(t.clients.c.client_id == client_id).values(**values)
            )
            if result.rowcount != 1:
                raise KeyError(client_id)
            conn.execute(insert(t.events).values(**_event(event_type, client_id, now)))

    # -- every request ------------------------------------------------------------------------
    def authenticate(self, token: str) -> ClientRecord:
        """The client a bearer token belongs to. Raises AuthError (401 unknown token, 403
        disabled client). Updates last_seen_at on success."""
        if not token.startswith(TOKEN_PREFIX):
            raise AuthError(401, "invalid_token", "bearer token is malformed")
        now = self.clock()
        with self.engine.begin() as conn:
            row = conn.execute(
                select(t.clients).where(t.clients.c.token_hash == hash_token(token))
            ).first()
            if row is None:
                raise AuthError(401, "invalid_token", "bearer token is not valid")
            if row.status == ClientStatus.DISABLED.value:
                raise AuthError(403, "client_disabled", f"client {row.client_id} is disabled")
            conn.execute(
                update(t.clients)
                .where(t.clients.c.client_id == row.client_id)
                .values(last_seen_at=now)
            )
        return replace(_record(row), last_seen_at=now)
