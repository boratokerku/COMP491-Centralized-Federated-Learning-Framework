"""Classification metrics and the server-side held-out evaluator.

The classes are imbalanced (about 11% positive in Bank Marketing, 13% in SMS Spam), so accuracy
alone is misleading: a model that always predicts the negative class already scores ~89%.
F1, ROC-AUC and PR-AUC are reported for the positive class (label 1).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from cfl.core.contract import StateDict


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    y = labels.cpu().numpy()
    loss = torch.nn.functional.cross_entropy(logits, labels).item()
    prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    pred = logits.argmax(dim=1).cpu().numpy()
    metrics = {
        "loss": loss,
        "accuracy": float((pred == y).mean()),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
    }
    if len(np.unique(y)) == 2:  # undefined when only one class is present
        metrics["roc_auc"] = float(roc_auc_score(y, prob))
        metrics["pr_auc"] = float(average_precision_score(y, prob))
    return metrics


class HeldOutEvaluator:
    """Evaluates global weights on the server's held-out test set (never part of client data)."""

    def __init__(
        self,
        build_model: Callable[[], torch.nn.Module],
        inputs: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 1024,
    ):
        if len(inputs) != len(labels) or len(labels) == 0:
            raise ValueError("inputs and labels must be non-empty and the same length")
        self.build_model = build_model
        self.inputs = inputs
        self.labels = labels
        self.batch_size = batch_size

    def __call__(self, weights: StateDict) -> dict[str, float]:
        model = self.build_model()
        model.load_state_dict(weights)
        model.eval()
        with torch.no_grad():
            logits = torch.cat(
                [model(batch) for batch in torch.split(self.inputs, self.batch_size)]
            )
        return classification_metrics(logits, self.labels)
