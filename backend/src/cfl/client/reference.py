"""Reference TaskSpec for both tasks, used when a site sets no CFL_TASK_SPEC.

It is the behaviour the client teams' own implementations must match (planning/interfaces.md,
section 2) and a starting point for them: the reference model, the shared pre-processing
(fixed scaling and one-hot encoding for tabular data, the hashing tokenizer for text), and the
reference local training of the simulation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from cfl.core.task import Split, TaskSpec
from cfl.simulation.trainer import evaluate, train_local
from cfl.tasks.registry import CONFIGS_DIR, get_task


class ReferenceTaskSpec(TaskSpec):
    def __init__(self, task_name: str):
        self._task = get_task(task_name)
        self.name = task_name
        self.dataset = self._task.dataset
        if task_name == "tabular":
            from cfl.tasks.tabular.preprocessing import load_dataset_config, transform

            config = load_dataset_config(CONFIGS_DIR / "datasets" / "bank_marketing.yaml")
            self._preprocess = lambda raw: transform(pd.DataFrame([raw]), config)[0].tolist()
        else:
            from cfl.tasks.text.tokenizer import encode

            self._preprocess = lambda raw: encode(raw["text"])

    def build_model(self) -> torch.nn.Module:
        return self._task.build_model()

    def preprocess(self, raw: dict) -> list[float] | list[int]:
        return self._preprocess(raw)

    def load_split(self, path: Path) -> Split:
        return self._task.load_split(path)

    def train(self, model, data: Split, config: dict, global_weights, seed: int):
        return train_local(model, *data, config, global_weights=global_weights, seed=seed)

    def evaluate(self, model, data: Split) -> dict[str, float]:
        return evaluate(model, *data)
