"""In-memory transport: simulated clients reach the coordinator's state through the same
interface (cfl.core.transport.Transport) that the HTTPS transport implements for real clients."""

from __future__ import annotations

from cfl.coordinator.aggregator import global_model_key
from cfl.coordinator.coordinator import Coordinator
from cfl.coordinator.intake import UpdateIntake
from cfl.coordinator.state_store import InMemoryStateStore
from cfl.core.store import ModelStore
from cfl.core.transport import Transport
from cfl.protocol import PROTOCOL_VERSION
from cfl.protocol.messages import (
    RegisterRequest,
    RegisterResponse,
    TaskAssignment,
    UpdateAck,
    UpdateMetadata,
)


class InMemoryTransport(Transport):
    def __init__(
        self,
        coordinator: Coordinator,
        intake: UpdateIntake,
        state: InMemoryStateStore,
        models: ModelStore,
    ):
        self.coordinator = coordinator
        self.intake = intake
        self.state = state
        self.models = models

    def register(self, request: RegisterRequest) -> RegisterResponse:
        """Simulation shortcut: every simulated client runs the managed reference code, so it
        is made eligible for the running runs without a conformance report."""
        if request.protocol_version != PROTOCOL_VERSION:
            return RegisterResponse(accepted=False, reason="protocol_version_mismatch")
        for run in self.state.running_runs():
            self.state.mark_eligible(run.manifest_version, [request.client_id])
        return RegisterResponse(accepted=True)

    def poll_task(self, client_id: str) -> TaskAssignment | None:
        return self.coordinator.active_task(client_id)

    def download_model(self, run_id: str, model_version: int) -> bytes:
        return self.models.get(global_model_key(run_id, model_version))

    def upload_update(self, metadata: UpdateMetadata, weights: bytes) -> UpdateAck:
        return self.intake.submit(metadata, weights)
