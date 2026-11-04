"""Tabular partitioning: the shared IID partitioner (cfl.tasks.partition) driven by a dataset
config, so labels and the CSV separator come from the config."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cfl.tasks import partition
from cfl.tasks.partition import ClientPartition, Partitioning
from cfl.tasks.tabular.preprocessing import DatasetConfig, encode_labels

__all__ = ["ClientPartition", "Partitioning", "iid_partition", "summarize", "write_partitions"]


def _labels(config: DatasetConfig) -> partition.LabelFn:
    return lambda df: encode_labels(df, config)


def iid_partition(
    df: pd.DataFrame,
    config: DatasetConfig,
    num_clients: int,
    seed: int,
    test_size: float = 0.2,
    local_val_size: float = 0.1,
    local_test_size: float = 0.2,
) -> Partitioning:
    return partition.iid_partition(
        df,
        _labels(config),
        num_clients=num_clients,
        seed=seed,
        test_size=test_size,
        local_val_size=local_val_size,
        local_test_size=local_test_size,
    )


def summarize(partitioning: Partitioning, config: DatasetConfig, **settings) -> dict:
    return partition.summarize(
        partitioning,
        _labels(config),
        config.name,
        preprocessing_version=config.preprocessing_version,
        **settings,
    )


def write_partitions(
    partitioning: Partitioning, config: DatasetConfig, out_dir: Path, summary: dict
) -> None:
    partition.write_partitions(partitioning, out_dir, summary, sep=config.separator)
