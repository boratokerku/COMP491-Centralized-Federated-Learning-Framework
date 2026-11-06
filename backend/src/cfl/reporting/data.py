"""Everything the report needs about one run, read through the StateStore interface."""

from __future__ import annotations

from dataclasses import dataclass

from cfl.coordinator.state_store import (
    EventRecord,
    MetricRecord,
    ParticipantRecord,
    RoundRecord,
    RunRecord,
    StateStore,
    StoredUpdate,
)


@dataclass(frozen=True)
class RunData:
    run: RunRecord
    rounds: list[RoundRecord]
    participants: dict[int, list[ParticipantRecord]]
    updates: dict[int, list[StoredUpdate]]
    events: list[EventRecord]
    metrics: list[MetricRecord]


def load_run_data(state: StateStore, run_id: str) -> RunData:
    """Raises KeyError if the run does not exist."""
    run = state.get_run(run_id)
    latest = state.latest_round(run_id)
    last = latest.round_id if latest else 0
    rounds = [state.get_round(run_id, i) for i in range(1, last + 1)]
    return RunData(
        run=run,
        rounds=rounds,
        participants={r.round_id: state.participants(run_id, r.round_id) for r in rounds},
        updates={r.round_id: state.updates(run_id, r.round_id) for r in rounds},
        events=state.events(run_id),
        metrics=state.metrics(run_id),
    )
