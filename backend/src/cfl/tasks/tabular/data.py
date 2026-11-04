"""Load a partition split (raw CSV rows) as model-ready tensors."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from cfl.tasks.tabular.preprocessing import DatasetConfig, encode_labels, transform


def load_split(path: str | Path, config: DatasetConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """(inputs float32 [N, 65], labels int64 [N]) for one CSV written by the partitioner."""
    df = pd.read_csv(path, sep=config.separator)
    return torch.from_numpy(transform(df, config)), torch.from_numpy(encode_labels(df, config))
