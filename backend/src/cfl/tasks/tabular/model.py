"""bank_mlp: the tabular model every client trains (managed or independent implementation)."""

from __future__ import annotations

import torch
from torch import nn

from cfl.core.contract import ModelContract

ARCHITECTURE_ID = "bank_mlp"
ARCHITECTURE_VERSION = "1.0.0"

INPUT_DIM = 65  # len(output_feature_names(bank_marketing config))
NUM_CLASSES = 2  # class 1 = positive label ("yes": subscribed)


class BankMLP(nn.Module):
    """x[65] -> fc1(64) -> ReLU -> Dropout -> fc2(32) -> ReLU -> fc3(2 logits)."""

    def __init__(self, input_dim: int = INPUT_DIM, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, NUM_CLASSES)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(torch.relu(self.fc1(x)))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def contract(input_dim: int = INPUT_DIM) -> ModelContract:
    return ModelContract.from_model(BankMLP(input_dim), ARCHITECTURE_ID, ARCHITECTURE_VERSION)
