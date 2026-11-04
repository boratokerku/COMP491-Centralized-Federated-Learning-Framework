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
        **fields: Any,
    ) -> RoundRecord:
        """Move the round to `target` (checked against the state machine), set `fields`, bump
        state_version and insert `event`, all at once. Raise StaleStateError if the stored
        state_version is not `expected_version`."""

    @abstractmethod
    def add_event(self, event: EventRecord) -> None: ...

    @abstractmethod
    def events(self, run_id: str | None = None) -> list[EventRecord]: ...


class InMemoryStateStore(StateStore):
    """State kept in dictionaries. Records are immutable, so callers never alias stored state."""

    def __init__(self):
        self._runs: dict[str, RunRecord] = {}
        self._rounds: dict[tuple[str, int], RoundRecord] = {}
        self._participants: dict[tuple[str, int], dict[str, ParticipantRecord]] = {}
        self._eligible: dict[str, set[str]] = {}  # manifest_version -> client_ids
        self._events: list[EventRecord] = []

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
        self._rounds[(run_id, round_id)] = updated
        self._events.append(event)
        return updated

    def add_event(self, event: EventRecord) -> None:
        self._events.append(event)

    def events(self, run_id: str | None = None) -> list[EventRecord]:
        return [copy.deepcopy(e) for e in self._events if run_id is None or e.run_id == run_id]
