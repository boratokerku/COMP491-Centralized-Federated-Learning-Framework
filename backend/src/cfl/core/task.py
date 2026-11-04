"""Per-modality client task definition (tabular: bank client, text: telecom client).

Contract and metric names: planning/interfaces.md, section 2. The shared client runtime calls
these methods; polling, uploads, retries and conformance reports are not part of a TaskSpec.
Reference behaviour: cfl.simulation.trainer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch

Split = tuple[torch.Tensor, torch.Tensor]  # (model inputs, labels with 1 = positive class)


class TaskSpec(ABC):
    """Everything modality-specific: model, pre-processing, data loading, training, evaluation."""

    name: str

    @abstractmethod
    def build_model(self) -> torch.nn.Module:
        """The run's reference architecture with unchanged parameter names."""

    @abstractmethod
    def preprocess(self, raw: dict) -> list[float] | list[int]:
        """One raw record (a manifest `synthetic_raw_input`) to one model input row. Must be the
        same code that prepares the training data; the conformance reference tests run it."""

    @abstractmethod
    def load_split(self, path: Path) -> Split:
        """One partition CSV (train.csv, val.csv or test.csv). Raw data never leaves the client."""

    @abstractmethod
    def train(
        self,
        model: torch.nn.Module,
        data: Split,
        config: dict,
        global_weights: dict[str, torch.Tensor] | None,
        seed: int,
    ) -> dict[str, float]:
        """Train in place with exactly the round config (planning/interfaces.md, section 3);
        add the FedProx term when config['proximal_mu'] > 0. Deterministic for a given seed.
        Returns at least 'train_loss'."""

    @abstractmethod
    def evaluate(self, model: torch.nn.Module, data: Split) -> dict[str, float]:
        """The keys of cfl.core.evaluation.classification_metrics."""
