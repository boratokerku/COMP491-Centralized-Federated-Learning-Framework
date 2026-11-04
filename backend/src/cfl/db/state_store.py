"""PostgreSQL implementation of the coordinator's StateStore (distributed mode).

Every interface method that the StateStore documents as atomic runs in one transaction.
Round transitions lock the round row and update it only if `state_version` is unchanged, so a
second coordinator process (or a stale read) can never overwrite a newer state.

The schema is created by the Alembic migrations in backend/migrations:
    CFL_DATABASE_URL=postgresql+psycopg://... alembic upgrade head      (from backend/)
"""

from __future__ import annotations

import os
import uuid
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from datetime import datetime
from enum import Enum
from typing import Any

from sqlalchemy import Engine, and_, create_engine, insert, select, update
from sqlalchemy.engine import Connection, Row
from sqlalchemy.exc import IntegrityError

from cfl.coordinator.state import RoundState, transition
from cfl.coordinator.state_store import (
    EventRecord,
    MetricRecord,
    ParticipantRecord,
    ParticipantStatus,
    RoundRecord,
    RunRecord,
    RunStatus,
    StaleStateError,
    StateStore,
    StoredUpdate,
)
from cfl.db import schema as t


def engine_from_env(var: str = "CFL_DATABASE_URL") -> Engine:
    url = os.environ.get(var)
    if not url:
        raise RuntimeError(f"set {var}, e.g. postgresql+psycopg://cfl:<password>@127.0.0.1/cfl")
    return create_engine(url)


def _plain(values: dict[str, Any]) -> dict[str, Any]:
    """Enums are stored as their text value."""
    return {k: (v.value if isinstance(v, Enum) else v) for k, v in values.items()}


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
    except (ValueError, TypeError, AttributeError):
        return False
    return True


def _record(cls, row: Row, **overrides):
    names = {f.name for f in dataclass_fields(cls)}
    data = {k: v for k, v in row._mapping.items() if k in names}
    return cls(**(data | overrides))


def _run(row: Row) -> RunRecord:
    return _record(RunRecord, row, status=RunStatus(row.status))


def _round(row: Row) -> RoundRecord:
    return _record(RoundRecord, row, state=RoundState(row.state))


def _participant(row: Row) -> ParticipantRecord:
    return _record(ParticipantRecord, row, status=ParticipantStatus(row.status))


def _event_values(event: EventRecord) -> dict[str, Any]:
    payload = event.payload or {}
    return {
        "event_type": event.event_type,
        "occurred_at": event.created_at,
        "run_id": event.run_id,
        "round_id": event.round_id,
        "client_id": event.client_id,
        "message_type": payload.get("message_type"),
        "reason_code": payload.get("reason_code"),
        "message_bytes": payload.get("bytes"),
        "duration_ms": payload.get("duration_ms"),
        "details": payload,
    }


def _metric_values(m: MetricRecord) -> dict[str, Any]:
    return {
        "run_id": m.run_id,
        "round_id": m.round_id,
        "client_id": m.client_id,
        "metric_name": m.metric_name,
        "value": m.value,
        "unit": m.unit,
        "message_type": m.message_type,
        "recorded_at": m.recorded_at,
    }


class PostgresStateStore(StateStore):
    def __init__(self, engine: Engine):
        self.engine = engine

    # -- simulation / development shortcut --------------------------------------------------
    def mark_eligible(self, manifest_version: str, client_ids: list[str]) -> None:
        """Approve simulated clients and record a PASSED conformance report for them.

        In distributed mode this comes from the gateway: an administrator approves the client
        and the client passes the conformance check. Never use this for real clients.
        """
        with self.engine.begin() as conn:
            for client_id in client_ids:
                exists = conn.execute(
                    select(t.clients.c.client_id).where(t.clients.c.client_id == client_id)
                ).first()
                if exists is None:
                    conn.execute(
                        insert(t.clients).values(
                            client_id=client_id,
                            token_hash=f"simulation-no-token:{client_id}",
                            status="REGISTERED",
                        )
                    )
                conn.execute(
                    insert(t.conformance_reports).values(
                        report_id=str(uuid.uuid4()),
                        client_id=client_id,
                        manifest_version=manifest_version,
                        client_version="simulation",
                        protocol_version="simulation",
                        status="PASSED",
                        report_payload={"source": "simulation"},
                    )
                )

    # -- runs --------------------------------------------------------------------------------
    def create_run(self, run: RunRecord, event: EventRecord) -> None:
        values = _plain({f.name: getattr(run, f.name) for f in dataclass_fields(RunRecord)})
        try:
            with self.engine.begin() as conn:
                conn.execute(insert(t.runs).values(**values))
                conn.execute(insert(t.events).values(**_event_values(event)))
        except IntegrityError as exc:
            raise ValueError(f"run {run.run_id} already exists") from exc

    def get_run(self, run_id: str) -> RunRecord:
        if not _is_uuid(run_id):
            raise KeyError(run_id)
        with self.engine.connect() as conn:
            row = conn.execute(select(t.runs).where(t.runs.c.run_id == run_id)).first()
        if row is None:
            raise KeyError(run_id)
        return _run(row)

    def running_runs(self) -> list[RunRecord]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(t.runs).where(t.runs.c.status == RunStatus.RUNNING.value)
            ).all()
        return [_run(r) for r in rows]

    def update_run(self, run_id: str, event: EventRecord, **fields: Any) -> RunRecord:
        with self.engine.begin() as conn:
            conn.execute(update(t.runs).where(t.runs.c.run_id == run_id).values(**_plain(fields)))
            conn.execute(insert(t.events).values(**_event_values(event)))
        return self.get_run(run_id)

    def eligible_clients(self, manifest_version: str) -> list[str]:
        """Active clients whose latest conformance report for this manifest version passed."""
        latest = (
            select(
                t.conformance_reports.c.client_id,
                t.conformance_reports.c.status,
            )
            .where(t.conformance_reports.c.manifest_version == manifest_version)
            .distinct(t.conformance_reports.c.client_id)
            .order_by(t.conformance_reports.c.client_id, t.conformance_reports.c.checked_at.desc())
            .subquery()
        )
        query = (
            select(t.clients.c.client_id)
            .join(latest, latest.c.client_id == t.clients.c.client_id)
            .where(latest.c.status == "PASSED", t.clients.c.status != "DISABLED")
            .order_by(t.clients.c.client_id)
        )
        with self.engine.connect() as conn:
            return list(conn.execute(query).scalars())

    # -- rounds ------------------------------------------------------------------------------
    @staticmethod
    def _round_key(run_id: str, round_id: int):
        return and_(t.rounds.c.run_id == run_id, t.rounds.c.round_id == round_id)

    def create_round(
        self, round_: RoundRecord, participants: list[ParticipantRecord], event: EventRecord
    ) -> None:
        values = _plain({f.name: getattr(round_, f.name) for f in dataclass_fields(RoundRecord)})
        try:
            with self.engine.begin() as conn:
                conn.execute(insert(t.rounds).values(**values))
                if participants:
                    conn.execute(
                        insert(t.round_participants),
                        [
                            _plain({f.name: getattr(p, f.name) for f in dataclass_fields(p)})
                            for p in participants
                        ],
                    )
                conn.execute(insert(t.events).values(**_event_values(event)))
        except IntegrityError as exc:
            raise ValueError(f"cannot create round {round_.run_id}/{round_.round_id}") from exc

    def get_round(self, run_id: str, round_id: int) -> RoundRecord:
        if not _is_uuid(run_id):
            raise KeyError((run_id, round_id))
        with self.engine.connect() as conn:
            row = conn.execute(select(t.rounds).where(self._round_key(run_id, round_id))).first()
        if row is None:
            raise KeyError((run_id, round_id))
        return _round(row)

    def latest_round(self, run_id: str) -> RoundRecord | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(t.rounds)
                .where(t.rounds.c.run_id == run_id)
                .order_by(t.rounds.c.round_id.desc())
                .limit(1)
            ).first()
        return _round(row) if row is not None else None

    def participants(self, run_id: str, round_id: int) -> list[ParticipantRecord]:
        p = t.round_participants
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(p)
                .where(p.c.run_id == run_id, p.c.round_id == round_id)
                .order_by(p.c.client_id)
            ).all()
        if not rows:
            self.get_round(run_id, round_id)  # KeyError for an unknown round, like in memory
        return [_participant(r) for r in rows]

    def transition_round(
        self,
        run_id: str,
        round_id: int,
        expected_version: int,
        target: RoundState,
        event: EventRecord,
        run_fields: dict[str, Any] | None = None,
        metrics: list[MetricRecord] | None = None,
        **fields: Any,
    ) -> RoundRecord:
        with self.engine.begin() as conn:
            row = conn.execute(
                select(t.rounds).where(self._round_key(run_id, round_id)).with_for_update()
            ).first()
            if row is None:
                raise KeyError((run_id, round_id))
            current = _round(row)
            if current.state_version != expected_version:
                raise StaleStateError(
                    f"round {run_id}/{round_id} is at version {current.state_version}, "
                    f"expected {expected_version}"
                )
            new_state = transition(current.state, target)
            changes = {**fields, "state": new_state, "state_version": expected_version + 1}
            result = conn.execute(
                update(t.rounds)
                .where(
                    self._round_key(run_id, round_id),
                    t.rounds.c.state_version == expected_version,
                )
                .values(**_plain(changes))
            )
            if result.rowcount != 1:
                raise StaleStateError(f"round {run_id}/{round_id} changed concurrently")
            if run_fields:
                conn.execute(
                    update(t.runs).where(t.runs.c.run_id == run_id).values(**_plain(run_fields))
                )
            self._insert_metrics(conn, metrics or [])
            conn.execute(insert(t.events).values(**_event_values(event)))
        return replace(current, **changes)

    # -- updates and participants ------------------------------------------------------------
    def get_update(self, update_id: str) -> StoredUpdate | None:
        if not _is_uuid(update_id):
            return None
        with self.engine.connect() as conn:
            row = conn.execute(
                select(t.updates).where(t.updates.c.update_id == update_id)
            ).first()
        return _record(StoredUpdate, row) if row is not None else None

    def find_update(self, run_id: str, round_id: int, client_id: str) -> StoredUpdate | None:
        u = t.updates
        with self.engine.connect() as conn:
            row = conn.execute(
                select(u).where(
                    u.c.run_id == run_id, u.c.round_id == round_id, u.c.client_id == client_id
                )
            ).first()
        return _record(StoredUpdate, row) if row is not None else None

    def accept_update(self, update_: StoredUpdate, event: EventRecord) -> None:
        if not _is_uuid(update_.update_id):
            raise ValueError(f"update_id {update_.update_id!r} is not a UUID")
        values = {f.name: getattr(update_, f.name) for f in dataclass_fields(StoredUpdate)}
        p = t.round_participants
        try:
            with self.engine.begin() as conn:
                # The FK to round_participants rejects clients that were not selected.
                conn.execute(insert(t.updates).values(**values))
                conn.execute(
                    update(p)
                    .where(
                        p.c.run_id == update_.run_id,
                        p.c.round_id == update_.round_id,
                        p.c.client_id == update_.client_id,
                    )
                    .values(
                        status=ParticipantStatus.UPDATE_ACCEPTED.value,
                        finished_at=update_.received_at,
                    )
                )
                conn.execute(insert(t.events).values(**_event_values(event)))
        except IntegrityError as exc:
            raise ValueError("duplicate update or client not selected for this round") from exc

    def updates(self, run_id: str, round_id: int) -> list[StoredUpdate]:
        u = t.updates
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(u)
                .where(u.c.run_id == run_id, u.c.round_id == round_id)
                .order_by(u.c.client_id)
            ).all()
        return [_record(StoredUpdate, r) for r in rows]

    def set_participant_status(
        self,
        run_id: str,
        round_id: int,
        client_id: str,
        status: ParticipantStatus,
        at: datetime,
        event: EventRecord,
        reason_code: str | None = None,
    ) -> None:
        p = t.round_participants
        with self.engine.begin() as conn:
            result = conn.execute(
                update(p)
                .where(p.c.run_id == run_id, p.c.round_id == round_id, p.c.client_id == client_id)
                .values(status=status.value, finished_at=at, reason_code=reason_code)
            )
            if result.rowcount != 1:
                raise KeyError((run_id, round_id, client_id))
            conn.execute(insert(t.events).values(**_event_values(event)))

    # -- events and metrics ------------------------------------------------------------------
    def add_event(self, event: EventRecord) -> None:
        with self.engine.begin() as conn:
            conn.execute(insert(t.events).values(**_event_values(event)))

    def events(self, run_id: str | None = None) -> list[EventRecord]:
        query = select(t.events).order_by(t.events.c.event_id)
        if run_id is not None:
            query = query.where(t.events.c.run_id == run_id)
        with self.engine.connect() as conn:
            rows = conn.execute(query).all()
        return [
            EventRecord(
                r.event_type, r.occurred_at, r.run_id, r.round_id, r.client_id, dict(r.details)
            )
            for r in rows
        ]

    @staticmethod
    def _insert_metrics(conn: Connection, metrics: list[MetricRecord]) -> None:
        if metrics:
            conn.execute(insert(t.metrics), [_metric_values(m) for m in metrics])

    def add_metrics(self, metrics: list[MetricRecord]) -> None:
        with self.engine.begin() as conn:
            self._insert_metrics(conn, metrics)

    def metrics(self, run_id: str) -> list[MetricRecord]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(t.metrics)
                .where(t.metrics.c.run_id == run_id)
                .order_by(t.metrics.c.metric_id)
            ).all()
        return [
            MetricRecord(
                r.run_id,
                r.metric_name,
                r.value,
                r.unit,
                r.recorded_at,
                r.round_id,
                r.client_id,
                r.message_type,
            )
            for r in rows
        ]
