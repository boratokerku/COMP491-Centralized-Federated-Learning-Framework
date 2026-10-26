"""Client-side transport interface.

The client always initiates communication (polling), so the same interface works behind NAT.
Implementations: InMemoryTransport (simulation), HttpTransport (distributed, over TLS),
and FaultyTransport (wraps another transport to inject delays, drops and duplicates in tests).
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
