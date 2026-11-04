"""Client-side transport interface.

The client always initiates communication (polling), so the same interface works behind NAT.
Implementations: cfl.simulation.transport.InMemoryTransport (simulation) and
cfl.client.http.HttpTransport (distributed mode, HTTPS with retries and checksum checks).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from cfl.protocol.messages import (
    RegisterRequest,
    RegisterResponse,
    TaskAssignment,
    UpdateAck,
    UpdateMetadata,
)


class Transport(ABC):
    @abstractmethod
    def register(self, request: RegisterRequest) -> RegisterResponse: ...

    @abstractmethod
    def poll_task(self, client_id: str) -> TaskAssignment | None:
        """Return the next task for this client, or None if there is nothing to do yet."""

    @abstractmethod
    def download_model(self, run_id: str, model_version: int) -> bytes: ...

    @abstractmethod
    def upload_update(self, metadata: UpdateMetadata, weights: bytes) -> UpdateAck: ...

    @abstractmethod
    def download_final_artifact(self, run_id: str) -> tuple[dict[str, bytes], str] | None:
        """The final model artifact files and the server-published SHA-256 of model.onnx, or
        None while the run is not completed (GET /model/{run_id}/final)."""
