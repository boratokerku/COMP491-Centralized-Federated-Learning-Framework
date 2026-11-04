"""SMS Spam Collection: reading the raw file, de-duplication and labels."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cfl.tasks.text.tokenizer import encode_batch

DATASET_NAME = "sms_spam"
RAW_FILE = "data/text/raw/SMSSpamCollection"  # relative to the repository root

LABEL_COLUMN = "label"
TEXT_COLUMN = "text"
POSITIVE = "spam"
NEGATIVE = "ham"


def read_raw(path: str | Path) -> pd.DataFrame:
    """The raw file is one message per line: `<ham|spam>\\t<text>`, no header, no quoting."""
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=[LABEL_COLUMN, TEXT_COLUMN],
        quoting=csv.QUOTE_NONE,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the first copy of each exact message text.

    The collection repeats several hundred messages (mostly spam templates). Without this, the
    same message can land in a client's train split and in a test split, inflating results.
    A text that appears with both labels would be ambiguous, so it is dropped entirely.
    """
    conflicting = df.groupby(TEXT_COLUMN)[LABEL_COLUMN].transform("nunique") > 1
    return df[~conflicting].drop_duplicates(TEXT_COLUMN).reset_index(drop=True)


def encode_labels(df: pd.DataFrame) -> np.ndarray:
    """Map the label column to 1 (spam) / 0 (ham)."""
    values = df[LABEL_COLUMN].to_numpy()
    unexpected = sorted(set(values) - {POSITIVE, NEGATIVE})
    if unexpected:
        raise ValueError(f"label values not in {{{POSITIVE!r}, {NEGATIVE!r}}}: {unexpected}")
    return (values == POSITIVE).astype(np.int64)


def load_split(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """(token ids int64 [N, MAX_LEN], labels int64 [N]) for one CSV written by the partitioner."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    ids = encode_batch(df[TEXT_COLUMN].tolist())
    return torch.from_numpy(ids), torch.from_numpy(encode_labels(df))
