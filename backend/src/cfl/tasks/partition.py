"""Seeded IID partitioning of a labelled dataset into simulated client sites (any modality).

Each client partition stands for one organization's private data and is split into
train / val / test: train for local training, val for per-round federated evaluation and
local early stopping, test for the final per-client metrics. A separate held-out test set
stays on the server side for global evaluation and baselines; it is never part of any client's
data. Partitions hold raw rows: pre-processing runs on each client. This is a development
tool; the coordinator never receives any of these files.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

SUMMARY_FILE = "summary.json"
CLIENT_SPLITS = ("train", "val", "test")

LabelFn = Callable[[pd.DataFrame], np.ndarray]  # raw rows -> 0/1 labels


@dataclass(frozen=True)
class ClientPartition:
    train: pd.DataFrame
    val: pd.DataFrame  # local validation split: per-round federated evaluation, early stopping
    test: pd.DataFrame  # local test split: final per-client metrics


@dataclass(frozen=True)
class Partitioning:
    clients: list[ClientPartition]
    test: pd.DataFrame  # held-out set for experiments; never part of any client's data


def iid_partition(
    df: pd.DataFrame,
    labels: LabelFn,
    num_clients: int,
    seed: int,
    test_size: float = 0.2,
    local_val_size: float = 0.1,
    local_test_size: float = 0.2,
) -> Partitioning:
    """Shuffle, hold out a test set, then split the rest into IID client partitions, each
    split again into train / val / test. `local_val_size` and `local_test_size` are shares of
    each client's data; train gets the rest.

    Every split is stratified on the label, so all clients keep the dataset's class ratio.
    The same inputs and seed always give the same partitions.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")
    for name, value in (
        ("test_size", test_size),
        ("local_val_size", local_val_size),
        ("local_test_size", local_test_size),
    ):
        if not 0 < value < 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if local_val_size + local_test_size >= 1:
        raise ValueError("local_val_size + local_test_size must be below 1")

    pool, test = train_test_split(df, test_size=test_size, stratify=labels(df), random_state=seed)

    pool = pool.reset_index(drop=True)
    if num_clients == 1:
        shards = [pool]
    else:
        folds = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)
        shards = [pool.iloc[idx] for _, idx in folds.split(pool, labels(pool))]

    clients = []
    for shard in shards:
        rest, local_test = train_test_split(
            shard, test_size=local_test_size, stratify=labels(shard), random_state=seed
        )
        train, val = train_test_split(
            rest,
            test_size=local_val_size / (1 - local_test_size),
            stratify=labels(rest),
            random_state=seed,
        )
        clients.append(
            ClientPartition(
                train.reset_index(drop=True),
                val.reset_index(drop=True),
                local_test.reset_index(drop=True),
            )
        )
    return Partitioning(clients, test.reset_index(drop=True))


def _stats(part: pd.DataFrame, labels: LabelFn) -> dict:
    positive = int(labels(part).sum())
    return {
        "rows": len(part),
        "positive": positive,
        "positive_ratio": round(positive / len(part), 4),
    }


def summarize(partitioning: Partitioning, labels: LabelFn, dataset: str, **settings) -> dict:
    return {
        "dataset": dataset,
        "partition": "iid_stratified",
        **settings,
        "clients": {
            f"client_{i}": {split: _stats(getattr(c, split), labels) for split in CLIENT_SPLITS}
            for i, c in enumerate(partitioning.clients)
        },
        "test": _stats(partitioning.test, labels),
    }


def format_summary(summary: dict, out_dir: Path | None = None) -> str:
    """Readable table of a summary(): rows, positives and positive ratio per split, so a skewed
    split is visible at a glance. The full summary is in summary.json."""
    width = 72
    lines = [
        "=" * width,
        f"{summary['dataset']} partition: {summary.get('partition', '')}, "
        f"seed={summary.get('seed', '-')}, clients={len(summary['clients'])}",
    ]
    if "duplicates_removed" in summary:
        lines.append(
            f"{summary['raw_rows']:,d} raw rows, "
            f"{summary['duplicates_removed']:,d} duplicates removed"
        )
    lines += [
        "=" * width,
        f"{'split':<28}{'rows':>12}{'positive':>14}{'positive %':>18}",
        "-" * width,
    ]
    total = 0

    def row(label: str, stats: dict) -> None:
        nonlocal total
        total += stats["rows"]
        lines.append(
            f"{label:<28}{stats['rows']:>12,d}{stats['positive']:>14,d}"
            f"{stats['positive_ratio'] * 100:>17.2f}%"
        )

    for client_id, splits in sorted(summary["clients"].items()):
        for split in CLIENT_SPLITS:
            row(f"{client_id} {split}", splits[split])
    lines.append("-" * width)
    row("held-out test (server)", summary["test"])
    lines += ["=" * width, f"total rows: {total:,d}"]
    if out_dir is not None:
        lines.append(f"written to: {out_dir} (details in {SUMMARY_FILE})")
    return "\n".join(lines)


def write_partitions(
    partitioning: Partitioning, out_dir: Path, summary: dict, sep: str = ","
) -> None:
    """Write client_<i>/{train,val,test}.csv, test.csv and summary.json.

    An existing output directory is replaced only if it was written by this function
    (it contains summary.json), so stale client folders never mix with new ones.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not (out_dir / SUMMARY_FILE).exists():
            raise FileExistsError(f"{out_dir} exists and was not created by the partitioner")
        shutil.rmtree(out_dir)
    for i, client in enumerate(partitioning.clients):
        client_dir = out_dir / f"client_{i}"
        client_dir.mkdir(parents=True)
        for split in CLIENT_SPLITS:
            getattr(client, split).to_csv(client_dir / f"{split}.csv", sep=sep, index=False)
    partitioning.test.to_csv(out_dir / "test.csv", sep=sep, index=False)
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
