"""TaskSpec implementation for the SMS Spam Collection client."""

from __future__ import annotations

from pathlib import Path

import torch

from cfl.core.task import Split, TaskSpec
from cfl.simulation.trainer import evaluate, train_local
from cfl.tasks.text.data import load_split
from cfl.tasks.text.model import SmsHashEmbed
from cfl.tasks.text.tokenizer import encode


class SmsTaskSpec(TaskSpec):
    """Client-side SMS model, preprocessing, data loading, training, and evaluation."""

    name = "text"

    def build_model(self) -> torch.nn.Module:
        return SmsHashEmbed()

    def preprocess(self, raw: dict) -> list[int]:
        return encode(raw["text"])

    def load_split(self, path: Path) -> Split:
        return load_split(path)

    def train(
        self,
        model: torch.nn.Module,
        data: Split,
        config: dict,
        global_weights: dict[str, torch.Tensor] | None,
        seed: int,
    ) -> dict[str, float]:
        return train_local(
            model,
            *data,
            config,
            global_weights=global_weights,
            seed=seed,
        )

    def evaluate(self, model: torch.nn.Module, data: Split) -> dict[str, float]:
        return evaluate(model, *data)
