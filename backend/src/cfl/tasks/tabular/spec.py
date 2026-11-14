"""TaskSpec implementation for the bank client (UCI Bank Marketing, tabular).

Everything the bank site does on its own: build bank_mlp, turn raw records into the 65 model
inputs, read its partition files, train with the round config exactly as received, and evaluate.
The pre-processing is fixed by backend/configs/datasets/bank_marketing.yaml (public mean/std,
pdays sentinel 999 with its indicator column, fixed one-hot vocabularies, `duration` dropped);
nothing is fitted on the site's own data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from cfl.core.evaluation import classification_metrics
from cfl.core.strategy import proximal_term
from cfl.core.task import Split, TaskSpec
from cfl.tasks.registry import CONFIGS_DIR
from cfl.tasks.tabular.data import load_split
from cfl.tasks.tabular.model import BankMLP
from cfl.tasks.tabular.preprocessing import DatasetConfig, load_dataset_config, transform

DATASET_CONFIG = CONFIGS_DIR / "datasets" / "bank_marketing.yaml"
EVAL_BATCH_SIZE = 4096


class BankTaskSpec(TaskSpec):
    """Client-side bank model, preprocessing, data loading, training, and evaluation."""

    name = "tabular"

    def __init__(self) -> None:
        self.config: DatasetConfig = load_dataset_config(DATASET_CONFIG)

    def build_model(self) -> torch.nn.Module:
        return BankMLP()

    def preprocess(self, raw: dict) -> list[float]:
        return transform(pd.DataFrame([raw]), self.config)[0].tolist()

    def load_split(self, path: Path) -> Split:
        return load_split(path, self.config)

    def train(
        self,
        model: torch.nn.Module,
        data: Split,
        config: dict,
        global_weights: dict[str, torch.Tensor] | None,
        seed: int,
    ) -> dict[str, float]:
        inputs, labels = data
        mu = float(config.get("proximal_mu") or 0.0)
        if mu > 0 and global_weights is None:
            raise ValueError("FedProx needs the global weights")
        class_weights = config.get("class_weights")
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay") or 0.0,
        )
        generator = torch.Generator().manual_seed(seed)

        model.train()
        total, batches = 0.0, 0
        with torch.random.fork_rng(devices=[]):  # dropout masks are reproducible per seed too
            torch.manual_seed(seed)
            for _ in range(config["local_epochs"]):
                order = torch.randperm(len(labels), generator=generator)
                for batch in torch.split(order, config["batch_size"]):
                    optimizer.zero_grad()
                    loss = loss_fn(model(inputs[batch]), labels[batch])
                    if mu > 0:
                        loss = loss + proximal_term(model, global_weights, mu)
                    loss.backward()
                    optimizer.step()
                    total += loss.item()
                    batches += 1
        return {"train_loss": total / max(batches, 1)}

    def evaluate(self, model: torch.nn.Module, data: Split) -> dict[str, float]:
        inputs, labels = data
        model.eval()
        with torch.no_grad():
            logits = torch.cat([model(b) for b in torch.split(inputs, EVAL_BATCH_SIZE)])
        return classification_metrics(logits, labels)
