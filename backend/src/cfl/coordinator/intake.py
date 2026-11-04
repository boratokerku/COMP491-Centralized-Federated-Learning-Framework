"""Update intake: the rules for accepting a client's upload (planning/api-spec.md, POST /update).

The simulation calls this directly through the in-memory transport; the gateway applies the
same rules on PostgreSQL. Checks, in order:

1. Same update_id already accepted -> accepted again (idempotent retry, counted once). This
   comes first so a retry whose original upload was accepted but whose response was lost still
   gets "accepted" after the round has moved on.
2. Checksum of the uploaded bytes                       -> checksum_mismatch
3. Client selected for this round                       -> not_selected
4. Round COLLECTING and deadline not passed             -> stale_round / deadline_passed
5. model_version equals the round's input version       -> model_version_mismatch
6. No other update from this client in this round       -> duplicate_update
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from enum import StrEnum

from cfl.coordinator.state import RoundState
from cfl.coordinator.state_store import EventRecord, StateStore, StoredUpdate
from cfl.core.serialization import sha256
from cfl.core.store import ModelStore
from cfl.protocol.messages import UpdateAck, UpdateMetadata


class IntakeReason(StrEnum):
    DUPLICATE_IDEMPOTENT = "duplicate_idempotent"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    UNKNOWN_ROUND = "unknown_round"
    NOT_SELECTED = "not_selected"
    STALE_ROUND = "stale_round"
    DEADLINE_PASSED = "deadline_passed"
    MODEL_VERSION_MISMATCH = "model_version_mismatch"
    DUPLICATE_UPDATE = "duplicate_update"


def update_key(run_id: str, round_id: int, client_id: str) -> str:
    return f"{run_id}/round_{round_id}/{client_id}.safetensors"


class UpdateIntake:
    def __init__(self, state: StateStore, models: ModelStore, clock: Callable[[], datetime]):
        self.state = state
        self.models = models
        self.clock = clock

    def submit(self, meta: UpdateMetadata, weights: bytes) -> UpdateAck:
        now = self.clock()

        previous = self.state.get_update(meta.update_id)
        if previous is not None and (previous.run_id, previous.round_id, previous.client_id) == (
            meta.run_id,
            meta.round_id,
            meta.client_id,
        ):
            self._event("UPDATE_RETRY_ACKNOWLEDGED", meta, now, len(weights))
            return UpdateAck(accepted=True, reason=IntakeReason.DUPLICATE_IDEMPOTENT)

        reason = self._reject_reason(meta, weights, now)
        if reason is not None:
            self._event("UPDATE_REJECTED", meta, now, len(weights), reason)
            return UpdateAck(accepted=False, reason=reason)

        # The file is written before the row that references it (db-schema.md).
        key = update_key(meta.run_id, meta.round_id, meta.client_id)
        self.models.put(key, weights)
        try:
            self.state.accept_update(
                StoredUpdate(
                    update_id=meta.update_id,
                    run_id=meta.run_id,
                    round_id=meta.round_id,
                    client_id=meta.client_id,
                    model_version=meta.model_version,
                    num_samples=meta.num_samples,
                    weights_path=key,
                    checksum_sha256=meta.checksum_sha256,
                    payload_bytes=len(weights),
                    received_at=now,
                    metrics=dict(meta.metrics),
                ),
                self._make_event("UPDATE_ACCEPTED", meta, now, len(weights)),
            )
        except ValueError:  # lost a race with a concurrent upload for the same client/round
            self._event("UPDATE_REJECTED", meta, now, len(weights), IntakeReason.DUPLICATE_UPDATE)
            return UpdateAck(accepted=False, reason=IntakeReason.DUPLICATE_UPDATE)
        return UpdateAck(accepted=True)

    def _reject_reason(
        self, meta: UpdateMetadata, weights: bytes, now: datetime
    ) -> IntakeReason | None:
        if sha256(weights) != meta.checksum_sha256:
            return IntakeReason.CHECKSUM_MISMATCH
        try:
            round_ = self.state.get_round(meta.run_id, meta.round_id)
        except KeyError:
            return IntakeReason.UNKNOWN_ROUND
        selected = {p.client_id for p in self.state.participants(meta.run_id, meta.round_id)}
        if meta.client_id not in selected:
            return IntakeReason.NOT_SELECTED
        if round_.state is not RoundState.COLLECTING:
            return IntakeReason.STALE_ROUND
        if now > round_.deadline_at:
            return IntakeReason.DEADLINE_PASSED
        if meta.model_version != round_.input_model_version:
            return IntakeReason.MODEL_VERSION_MISMATCH
        if self.state.find_update(meta.run_id, meta.round_id, meta.client_id) is not None:
            return IntakeReason.DUPLICATE_UPDATE
        return None

    @staticmethod
    def _make_event(
        event_type: str,
        meta: UpdateMetadata,
        now: datetime,
        size: int,
        reason: IntakeReason | None = None,
    ) -> EventRecord:
        payload = {"update_id": meta.update_id, "message_type": "update_upload", "bytes": size}
        if reason is not None:
            payload["reason_code"] = reason.value
        return EventRecord(
            event_type,
            now,
            run_id=meta.run_id,
            round_id=meta.round_id,
            client_id=meta.client_id,
            payload=payload,
        )

    def _event(self, *args, **kwargs) -> None:
        self.state.add_event(self._make_event(*args, **kwargs))
