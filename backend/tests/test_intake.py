"""UpdateIntake rejections whose run_id/round_id do not exist (POST /update -> 404 unknown_round).

The events table has foreign keys to runs and rounds, and InMemoryStateStore enforces the same
keys, so a rejection event that names a made-up run or round fails here as it did on PostgreSQL.
"""

import uuid
from datetime import UTC, datetime

import pytest

from cfl.coordinator.intake import IntakeReason, UpdateIntake
from cfl.coordinator.state_store import (
    EventRecord,
    InMemoryStateStore,
    RunRecord,
    RunStatus,
)
from cfl.core.serialization import sha256
from cfl.core.store import InMemoryModelStore
from cfl.protocol.messages import UpdateMetadata

T0 = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)
RUN_ID = "22222222-2222-4222-8222-222222222222"
SHA = "a" * 64
WEIGHTS = b"not really safetensors"


def _store_with_run() -> InMemoryStateStore:
    store = InMemoryStateStore()
    store.create_run(
        RunRecord(
            run_id=RUN_ID,
            name="intake-test",
            status=RunStatus.RUNNING,
            strategy="fedavg",
            task_name="tabular",
            seed=1,
            config={},
            config_sha256=SHA,
            protocol_version="1.0",
            manifest_version="1.0.0",
            manifest_sha256=SHA,
            preprocessing_version="pre-1",
            model_architecture="mlp",
            model_architecture_version="1",
            model_fingerprint=SHA,
            model_version=0,
            model_path=f"{RUN_ID}/models/0.safetensors",
            model_sha256=SHA,
            created_at=T0,
        ),
        EventRecord("RUN_STARTED", T0, run_id=RUN_ID),
    )
    return store


def _meta(run_id: str, **kw) -> UpdateMetadata:
    fields = {
        "run_id": run_id,
        "round_id": 1,
        "client_id": "bank-a",
        "update_id": str(uuid.uuid4()),
        "model_version": 0,
        "num_samples": 10,
        "checksum_sha256": sha256(WEIGHTS),
    }
    fields.update(kw)
    return UpdateMetadata(**fields)


def _submit(store: InMemoryStateStore, meta: UpdateMetadata, weights: bytes = WEIGHTS):
    return UpdateIntake(store, InMemoryModelStore(), lambda: T0).submit(meta, weights)


@pytest.mark.parametrize(
    ("run_id", "with_run"),
    [
        (str(uuid.uuid4()), False),  # run does not exist
        (RUN_ID, True),  # run exists, round does not
        ("not-a-uuid", False),  # would not even fit the UUID column
    ],
)
def test_unknown_round_is_rejected_and_logged_without_foreign_keys(run_id, with_run):
    store = _store_with_run() if with_run else InMemoryStateStore()

    ack = _submit(store, _meta(run_id))

    assert not ack.accepted
    assert ack.reason == IntakeReason.UNKNOWN_ROUND
    event = store.events()[-1]
    assert event.event_type == "UPDATE_REJECTED"
    assert (event.run_id, event.round_id) == (None, None)
    assert event.client_id == "bank-a"
    assert event.payload["reason_code"] == "unknown_round"
    assert event.payload["claimed_run_id"] == run_id
    assert event.payload["claimed_round_id"] == 1


@pytest.mark.parametrize(
    ("meta_kw", "weights", "reason"),
    [
        ({"update_id": "not-a-uuid"}, WEIGHTS, IntakeReason.INVALID_UPDATE_ID),
        ({}, b"other bytes", IntakeReason.CHECKSUM_MISMATCH),
    ],
)
def test_checks_before_the_round_lookup_survive_an_unknown_run(meta_kw, weights, reason):
    store = InMemoryStateStore()

    ack = _submit(store, _meta(str(uuid.uuid4()), **meta_kw), weights)

    assert ack.reason == reason
    event = store.events()[-1]
    assert (event.run_id, event.round_id) == (None, None)
    assert event.payload["reason_code"] == reason.value


def test_in_memory_events_enforce_the_same_foreign_keys_as_postgres():
    store = _store_with_run()
    with pytest.raises(ValueError):
        store.add_event(EventRecord("X", T0, run_id=str(uuid.uuid4())))
    with pytest.raises(ValueError):
        store.add_event(EventRecord("X", T0, run_id=RUN_ID, round_id=7))
    store.add_event(EventRecord("X", T0, run_id=RUN_ID))  # run-level event: fine
