"""Seeded IID partitioning of a tabular dataset into simulated client sites.

Each client partition stands for one organization's private data. Partitions hold raw rows:
pre-processing runs on each client (see preprocessing.py). This is a development tool; the
coordinator never receives any of these files.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from cfl.tasks.tabular.preprocessing import DatasetConfig, encode_labels

SUMMARY_FILE = "summary.json"


@dataclass(frozen=True)
class ClientPartition:
    train: pd.DataFrame
    test: pd.DataFrame  # local test split, used for federated evaluation


@dataclass(frozen=True)
class Partitioning:
    clients: list[ClientPartition]
    test: pd.DataFrame  # held-out set for experiments; never part of any client's data


def iid_partition(
    df: pd.DataFrame,
    config: DatasetConfig,
    num_clients: int,
    seed: int,
    test_size: float = 0.2,
    local_test_size: float = 0.2,
) -> Partitioning:
    """Shuffle, hold out a test set, then split the rest into IID client partitions.

    Every split is stratified on the label, so all clients keep the dataset's class ratio.
    The same inputs and seed always give the same partitions.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1")
    for name, value in (("test_size", test_size), ("local_test_size", local_test_size)):
        if not 0 < value < 1:
            raise ValueError(f"{name} must be between 0 and 1")

    y = encode_labels(df, config)
    pool, test = train_test_split(df, test_size=test_size, stratify=y, random_state=seed)

    pool = pool.reset_index(drop=True)
    y_pool = encode_labels(pool, config)
    if num_clients == 1:
        shards = [pool]
    else:
        folds = StratifiedKFold(n_splits=num_clients, shuffle=True, random_state=seed)
        shards = [pool.iloc[idx] for _, idx in folds.split(pool, y_pool)]

    clients = []
    for shard in shards:
        train, local_test = train_test_split(
            shard,
            test_size=local_test_size,
            stratify=encode_labels(shard, config),
            random_state=seed,
        )
        clients.append(
            ClientPartition(train.reset_index(drop=True), local_test.reset_index(drop=True))
        )
    return Partitioning(clients, test.reset_index(drop=True))


def _stats(part: pd.DataFrame, config: DatasetConfig) -> dict:
    positive = int(encode_labels(part, config).sum())
    return {
        "rows": len(part),
        "positive": positive,
        "positive_ratio": round(positive / len(part), 4),
    }


def summarize(partitioning: Partitioning, config: DatasetConfig, **settings) -> dict:
    return {
        "dataset": config.name,
        "preprocessing_version": config.preprocessing_version,
        "partition": "iid_stratified",
        **settings,
        "clients": {
            f"client_{i}": {
                "train": _stats(c.train, config),
                "test": _stats(c.test, config),
            }
            for i, c in enumerate(partitioning.clients)
        },
        "test": _stats(partitioning.test, config),
    }


def write_partitions(
    partitioning: Partitioning, config: DatasetConfig, out_dir: Path, summary: dict
) -> None:
    """Write client_<i>/train.csv, client_<i>/test.csv, test.csv and summary.json.

    An existing output directory is replaced only if it was written by this function
    (it contains summary.json), so stale client folders never mix with new ones.
    """
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not (out_dir / SUMMARY_FILE).exists():
            raise FileExistsError(f"{out_dir} exists and was not created by the partitioner")
        shutil.rmtree(out_dir)
    sep = config.separator
    for i, client in enumerate(partitioning.clients):
        client_dir = out_dir / f"client_{i}"
        client_dir.mkdir(parents=True)
        client.train.to_csv(client_dir / "train.csv", sep=sep, index=False)
        client.test.to_csv(client_dir / "test.csv", sep=sep, index=False)
    partitioning.test.to_csv(out_dir / "test.csv", sep=sep, index=False)
    (out_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2))
