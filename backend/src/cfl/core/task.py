"""Per-modality task definition (tabular, image, ...)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class TaskSpec(ABC):
    """Everything modality-specific: model, data loading, local training, evaluation."""

    name: str

    @abstractmethod
    def build_model(self) -> torch.nn.Module: ...

    @abstractmethod
    def load_partition(self, client_id: str) -> Any:
        """Load this client's local data. Raw data never leaves the client."""

    @abstractmethod
    def train(
        self,
        model: torch.nn.Module,
        data: Any,
        config: dict,
        global_weights: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, float]:
        """Train in place for config['local_epochs']; apply the FedProx term if config has
        'proximal_mu'. Returns training metrics."""

    @abstractmethod
    def evaluate(self, model: torch.nn.Module, data: Any) -> dict[str, float]: ...
