"""Persistent coordinator state: runs, rounds, round participants, eligible clients and events.

The records mirror the PostgreSQL tables in planning/db-schema.md. The simulation uses
InMemoryStateStore; the distributed mode will use a PostgreSQL implementation of the same
interface, so the coordinator code does not change between modes.

Every round transition goes through `transition_round`, which checks the expected
`state_version` (optimistic locking) so a stale writer, such as a second coordinator process,
cannot overwrite a newer state.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import StrEnum
from typing import Any

from cfl.coordinator.state import RoundState, transition


class RunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


class ParticipantStatus(StrEnum):
    SELECTED = "SELECTED"
    UPDATE_ACCEPTED = "UPDATE_ACCEPTED"
    DROPOUT = "DROPOUT"
    REJECTED = "REJECTED"


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    name: str
    status: RunStatus
    strategy: str
    task_name: str
    seed: int
    config: dict[str, Any]  # immutable effective run configuration
    config_sha256: str
    protocol_version: str
    manifest_version: str
    manifest_sha256: str
    preprocessing_version: str
    model_architecture: str
    model_architecture_version: str
    model_fingerprint: str
    model_version: int  # latest committed global model
    model_path: str
    model_sha256: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    failure_reason: str | None = None
    final_model_path: str | None = None  # model.onnx of the final artifact
    final_model_sha256: str | None = None
    final_model_format: str | None = None
    final_model_bytes: int | None = None


@dataclass(frozen=True)
class RoundRecord:
    run_id: str
    round_id: int
    state: RoundState
    input_model_version: int
    input_model_path: str
    input_model_sha256: str
    min_participants: int
    deadline_at: datetime
    opened_at: datetime
    state_version: int = 0
    output_model_version: int | None = None
    output_model_path: str | None = None
    output_model_sha256: str | None = None
    closed_at: datetime | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class ParticipantRecord:
    run_id: str
    round_id: int
    client_id: str
    status: ParticipantStatus
    selected_at: datetime
    finished_at: datetime | None = None
    reason_code: str | None = None


@dataclass(frozen=True)
class StoredUpdate:
    """An accepted upload (`updates` table). The weights live in the ModelStore."""

    update_id: str
    run_id: str
    round_id: int
    client_id: str
    model_version: int
    num_samples: int
    weights_path: str
    checksum_sha256: str
    payload_bytes: int
    received_at: datetime
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricRecord:
    run_id: str
    metric_name: str
    value: float
    unit: str
    recorded_at: datetime
    round_id: int | None = None
    client_id: str | None = None
    message_type: str | None = None  # communication metrics: register, model_download, ...


@dataclass(frozen=True)
class EventRecord:
    event_type: str
    created_at: datetime
    run_id: str | None = None
    round_id: int | None = None
    client_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class StaleStateError(RuntimeError):
    """The round changed since it was read; re-read it and decide again."""


class StateStore(ABC):
    @abstractmethod
    def create_run(self, run: RunRecord, event: EventRecord) -> None: ...

    @abstractmethod
    def get_run(self, run_id: str) -> RunRecord: ...

    @abstractmethod
    def running_runs(self) -> list[RunRecord]: ...

    @abstractmethod
    def eligible_clients(self, manifest_version: str) -> list[str]:
        """Approved clients with a PASSED conformance report for this manifest version."""

    @abstractmethod
    def create_round(
        self, round_: RoundRecord, participants: list[ParticipantRecord], event: EventRecord
    ) -> None:
        """Insert the round and its selected participants in one transaction."""

    @abstractmethod
    def get_round(self, run_id: str, round_id: int) -> RoundRecord: ...

    @abstractmethod
    def latest_round(self, run_id: str) -> RoundRecord | None: ...

    @abstractmethod
    def participants(self, run_id: str, round_id: int) -> list[ParticipantRecord]: ...

    @abstractmethod
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
        """In one transaction: move the round to `target` (checked against the state machine),
        set its `fields`, bump state_version, apply `run_fields` to the run (e.g. the new
        global model pointer), insert `metrics` and `event`. Raise StaleStateError if the
        stored state_version is not `expected_version`."""

    @abstractmethod
    def update_run(self, run_id: str, event: EventRecord, **fields: Any) -> RunRecord: ...

    @abstractmethod
    def get_update(self, update_id: str) -> StoredUpdate | None: ...

    @abstractmethod
    def find_update(self, run_id: str, round_id: int, client_id: str) -> StoredUpdate | None: ...

    @abstractmethod
    def accept_update(self, update: StoredUpdate, event: EventRecord) -> None:
        """Insert the update and mark its participant UPDATE_ACCEPTED in one transaction.
        Raise ValueError on a duplicate update_id or (run, round, client)."""

    @abstractmethod
    def updates(self, run_id: str, round_id: int) -> list[StoredUpdate]: ...

    @abstractmethod
    def set_participant_status(
        self,
        run_id: str,
        round_id: int,
        client_id: str,
        status: ParticipantStatus,
        at: datetime,
        event: EventRecord,
        reason_code: str | None = None,
    ) -> None: ...

    @abstractmethod
    def add_event(self, event: EventRecord) -> None: ...

    @abstractmethod
    def events(self, run_id: str | None = None) -> list[EventRecord]: ...

    @abstractmethod
    def add_metrics(self, metrics: list[MetricRecord]) -> None: ...

    @abstractmethod
    def metrics(self, run_id: str) -> list[MetricRecord]: ...


class InMemoryStateStore(StateStore):
    """State kept in dictionaries. Records are immutable, so callers never alias stored state."""

    def __init__(self):
        self._runs: dict[str, RunRecord] = {}
        self._rounds: dict[tuple[str, int], RoundRecord] = {}
        self._participants: dict[tuple[str, int], dict[str, ParticipantRecord]] = {}
        self._eligible: dict[str, set[str]] = {}  # manifest_version -> client_ids
        self._updates: dict[str, StoredUpdate] = {}
        self._events: list[EventRecord] = []
        self._metrics: list[MetricRecord] = []

    # Simulation setup: in distributed mode this comes from registration and conformance.
    def mark_eligible(self, manifest_version: str, client_ids: list[str]) -> None:
        self._eligible.setdefault(manifest_version, set()).update(client_ids)

    def create_run(self, run: RunRecord, event: EventRecord) -> None:
        if run.run_id in self._runs:
            raise ValueError(f"run {run.run_id} already exists")
        self._runs[run.run_id] = run
        self._events.append(event)

    def get_run(self, run_id: str) -> RunRecord:
        return self._runs[run_id]

    def running_runs(self) -> list[RunRecord]:
        return [r for r in self._runs.values() if r.status is RunStatus.RUNNING]

    def eligible_clients(self, manifest_version: str) -> list[str]:
        return sorted(self._eligible.get(manifest_version, set()))

    def create_round(
        self, round_: RoundRecord, participants: list[ParticipantRecord], event: EventRecord
    ) -> None:
        key = (round_.run_id, round_.round_id)
        if key in self._rounds:
            raise ValueError(f"round {key} already exists")
        self._rounds[key] = round_
        self._participants[key] = {p.client_id: p for p in participants}
        self._events.append(event)

    def get_round(self, run_id: str, round_id: int) -> RoundRecord:
        return self._rounds[(run_id, round_id)]

    def latest_round(self, run_id: str) -> RoundRecord | None:
        ids = [rid for (run, rid) in self._rounds if run == run_id]
        return self._rounds[(run_id, max(ids))] if ids else None

    def participants(self, run_id: str, round_id: int) -> list[ParticipantRecord]:
        return sorted(self._participants[(run_id, round_id)].values(), key=lambda p: p.client_id)

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
        current = self._rounds[(run_id, round_id)]
        if current.state_version != expected_version:
            raise StaleStateError(
                f"round {run_id}/{round_id} is at version {current.state_version}, "
                f"expected {expected_version}"
            )
        new_state = transition(current.state, target)
        updated = replace(
            current, state=new_state, state_version=current.state_version + 1, **fields
        )
        if run_fields:
            self._runs[run_id] = replace(self._runs[run_id], **run_fields)
        self._rounds[(run_id, round_id)] = updated
        self._metrics.extend(metrics or [])
        self._events.append(event)
        return updated

    def update_run(self, run_id: str, event: EventRecord, **fields: Any) -> RunRecord:
        self._runs[run_id] = replace(self._runs[run_id], **fields)
        self._events.append(event)
        return self._runs[run_id]

    def get_update(self, update_id: str) -> StoredUpdate | None:
        return self._updates.get(update_id)

    def find_update(self, run_id: str, round_id: int, client_id: str) -> StoredUpdate | None:
        for u in self._updates.values():
            if (u.run_id, u.round_id, u.client_id) == (run_id, round_id, client_id):
                return u
        return None

    def accept_update(self, update: StoredUpdate, event: EventRecord) -> None:
        if update.update_id in self._updates:
            raise ValueError(f"update {update.update_id} already exists")
        if self.find_update(update.run_id, update.round_id, update.client_id) is not None:
            raise ValueError("an update for this client and round already exists")
        key = (update.run_id, update.round_id)
        participant = self._participants[key][update.client_id]  # FK: must be selected
        self._participants[key][update.client_id] = replace(
            participant, status=ParticipantStatus.UPDATE_ACCEPTED, finished_at=update.received_at
        )
        self._updates[update.update_id] = update
        self._events.append(event)

    def updates(self, run_id: str, round_id: int) -> list[StoredUpdate]:
        found = [u for u in self._updates.values() if (u.run_id, u.round_id) == (run_id, round_id)]
        return sorted(found, key=lambda u: u.client_id)

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
        key = (run_id, round_id)
        self._participants[key][client_id] = replace(
            self._participants[key][client_id],
            status=status,
            finished_at=at,
            reason_code=reason_code,
        )
        self._events.append(event)

    def add_event(self, event: EventRecord) -> None:
        # Same foreign keys as the events table, so code that passes here also passes on
        # PostgreSQL (events.run_id -> runs, events.(run_id, round_id) -> rounds).
        if event.run_id is not None and event.run_id not in self._runs:
            raise ValueError(f"event references unknown run {event.run_id}")
        if event.round_id is not None and (event.run_id, event.round_id) not in self._rounds:
            raise ValueError(f"event references unknown round {event.run_id}/{event.round_id}")
        self._events.append(event)

    def events(self, run_id: str | None = None) -> list[EventRecord]:
        return [copy.deepcopy(e) for e in self._events if run_id is None or e.run_id == run_id]

    def add_metrics(self, metrics: list[MetricRecord]) -> None:
        self._metrics.extend(metrics)

    def metrics(self, run_id: str) -> list[MetricRecord]:
        return [m for m in self._metrics if m.run_id == run_id]
