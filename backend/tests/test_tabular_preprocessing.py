from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from cfl.tasks.tabular.preprocessing import (
    DatasetConfig,
    encode_labels,
    load_dataset_config,
    output_feature_names,
    transform,
)

BANK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "datasets" / "bank_marketing.yaml"


def _config(**overrides) -> DatasetConfig:
    raw = {
        "name": "toy",
        "preprocessing_version": "1.0",
        "file": "toy.csv",
        "label": {"column": "y", "positive": "yes", "negative": "no"},
        "features": [
            {"name": "age", "type": "numeric", "unit": "years", "mean": 40.0, "std": 10.0},
            {
                "name": "pdays",
                "type": "numeric",
                "unit": "days",
                "mean": 5.0,
                "std": 2.0,
                "sentinel": 999,
                "sentinel_fill": 5.0,
            },
            {"name": "job", "type": "categorical", "categories": ["a", "b", "unknown"]},
        ],
        "drop": ["duration"],
    }
    raw.update(overrides)
    return DatasetConfig.model_validate(raw)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [30, 50],
            "pdays": [999, 9],
            "job": ["b", "unknown"],
            "duration": [100, 200],
            "y": ["no", "yes"],
        }
    )


def test_output_names_follow_config_order():
    assert output_feature_names(_config()) == [
        "age",
        "pdays",
        "pdays_is_sentinel",
        "job=a",
        "job=b",
        "job=unknown",
    ]


def test_transform_values():
    x = transform(_frame(), _config())
    assert x.dtype == np.float32
    expected = np.array(
        [
            [-1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # age 30, pdays sentinel -> fill (mean) + flag
            [1.0, 2.0, 0.0, 0.0, 0.0, 1.0],  # age 50, pdays 9
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(x, expected)


def test_transform_is_deterministic_and_ignores_extra_columns():
    df = _frame()
    reordered = df[["y", "job", "duration", "pdays", "age"]]
    np.testing.assert_array_equal(transform(df, _config()), transform(reordered, _config()))


def test_unseen_category_is_rejected():
    df = _frame()
    df.loc[0, "job"] = "astronaut"
    with pytest.raises(ValueError, match="not in config categories"):
        transform(df, _config())


def test_missing_feature_column_is_rejected():
    with pytest.raises(ValueError, match="missing feature columns"):
        transform(_frame().drop(columns=["age"]), _config())


def test_labels():
    np.testing.assert_array_equal(encode_labels(_frame(), _config()), [0, 1])
    df = _frame()
    df.loc[0, "y"] = "maybe"
    with pytest.raises(ValueError, match="label values"):
        encode_labels(df, _config())


def test_invalid_configs_are_rejected():
    with pytest.raises(ValidationError):
        _config(features=[{"name": "y", "type": "categorical", "categories": ["a"]}])
    with pytest.raises(ValidationError):
        _config(drop=["age"])
    with pytest.raises(ValidationError):
        _config(features=[{"name": "age", "type": "numeric", "unit": "y", "mean": 0, "std": 0}])


def test_bank_marketing_config_loads():
    config = load_dataset_config(BANK_CONFIG)
    names = output_feature_names(config)
    assert config.label.column == "y"
    assert "duration" in config.drop
    assert len(names) == len(set(names)) == 65
