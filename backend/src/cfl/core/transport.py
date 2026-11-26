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
    RoundKeysResponse,
    RoundKeySubmit,
    TaskAssignment,
    UnmaskSharesRequest,
    UnmaskSharesResponse,
    UnmaskStatusResponse,
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

    # -- Optional SecAgg+ key exchange & unmasking (implemented by HTTP and in-memory transports)
    def submit_round_keys(self, submit: RoundKeySubmit) -> RoundKeysResponse:
        raise NotImplementedError("transport does not support SecAgg+ key exchange")

    def get_round_keys(self, run_id: str, round_id: int, client_id: str) -> RoundKeysResponse:
        raise NotImplementedError("transport does not support SecAgg+ key exchange")

    def get_unmask_status(self, run_id: str, round_id: int, client_id: str) -> UnmaskStatusResponse:
        raise NotImplementedError("transport does not support SecAgg+ unmasking")

    def submit_unmask_shares(self, req: UnmaskSharesRequest) -> UnmaskSharesResponse:
        raise NotImplementedError("transport does not support SecAgg+ unmasking")
