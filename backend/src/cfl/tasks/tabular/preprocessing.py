"""Config-driven pre-processing for tabular datasets.

Runs on the client side. Everything dataset-specific (columns, order, categories, scaling
statistics) comes from a dataset config (see backend/configs/datasets/). Nothing is fitted on
client data: categories and scaling statistics are fixed in the config, so every client
encodes the same input identically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class NumericFeature(BaseModel):
    name: str
    type: Literal["numeric"]
    unit: str
    mean: float
    std: float = Field(gt=0)
    sentinel: float | None = None  # value meaning "not available", e.g. pdays=999
    sentinel_fill: float | None = None

    @model_validator(mode="after")
    def _sentinel_needs_fill(self) -> NumericFeature:
        if (self.sentinel is None) != (self.sentinel_fill is None):
            raise ValueError(f"{self.name}: sentinel and sentinel_fill must be set together")
        return self


class CategoricalFeature(BaseModel):
    name: str
    type: Literal["categorical"]
    categories: list[str] = Field(min_length=1)

    @field_validator("categories")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("categories must be unique")
        return v


Feature = Annotated[NumericFeature | CategoricalFeature, Field(discriminator="type")]


class LabelSpec(BaseModel):
    column: str
    positive: str
    negative: str


class DatasetConfig(BaseModel):
    name: str
    preprocessing_version: str
    file: str
    separator: str = ","
    label: LabelSpec
    features: list[Feature] = Field(min_length=1)
    drop: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent_columns(self) -> DatasetConfig:
        names = [f.name for f in self.features]
        if len(set(names)) != len(names):
            raise ValueError("feature names must be unique")
        if self.label.column in names:
            raise ValueError("the label column cannot also be a feature")
        overlap = set(self.drop) & set(names)
        if overlap:
            raise ValueError(f"dropped columns cannot be features: {sorted(overlap)}")
        return self


def load_dataset_config(path: str | Path) -> DatasetConfig:
    with open(path) as f:
        return DatasetConfig.model_validate(yaml.safe_load(f))


def read_raw(config: DatasetConfig, base_dir: str | Path) -> pd.DataFrame:
    """Read the raw file named in the config; `base_dir` is the directory it is relative to."""
    return pd.read_csv(Path(base_dir) / config.file, sep=config.separator)


def output_feature_names(config: DatasetConfig) -> list[str]:
    """Names of the model input columns, in order."""
    names: list[str] = []
    for f in config.features:
        if isinstance(f, NumericFeature):
            names.append(f.name)
            if f.sentinel is not None:
                names.append(f"{f.name}_is_sentinel")
        else:
            names.extend(f"{f.name}={c}" for c in f.categories)
    return names


def transform(df: pd.DataFrame, config: DatasetConfig) -> np.ndarray:
    """Turn raw rows into the model input matrix (float32), without fitting anything."""
    missing = [f.name for f in config.features if f.name not in df.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")

    blocks: list[np.ndarray] = []
    for f in config.features:
        col = df[f.name]
        if isinstance(f, NumericFeature):
            values = pd.to_numeric(col, errors="raise").to_numpy(dtype=np.float64)
            if np.isnan(values).any():
                raise ValueError(f"{f.name}: missing numeric values")
            if f.sentinel is not None:
                is_sentinel = values == f.sentinel
                values = np.where(is_sentinel, f.sentinel_fill, values)
                blocks.append(((values - f.mean) / f.std)[:, None])
                blocks.append(is_sentinel.astype(np.float64)[:, None])
            else:
                blocks.append(((values - f.mean) / f.std)[:, None])
        else:
            values = col.astype(str).to_numpy()
            unexpected = sorted(set(values) - set(f.categories))
            if unexpected:
                raise ValueError(f"{f.name}: values not in config categories: {unexpected}")
            one_hot = (values[:, None] == np.asarray(f.categories)[None, :]).astype(np.float64)
            blocks.append(one_hot)

    return np.hstack(blocks).astype(np.float32)


def encode_labels(df: pd.DataFrame, config: DatasetConfig) -> np.ndarray:
    """Map the label column to 1 (positive) / 0 (negative)."""
    spec = config.label
    values = df[spec.column].astype(str).to_numpy()
    unexpected = sorted(set(values) - {spec.positive, spec.negative})
    if unexpected:
        raise ValueError(f"label values not in config: {unexpected}")
    return (values == spec.positive).astype(np.int64)
