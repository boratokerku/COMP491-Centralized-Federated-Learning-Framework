"""Conformance manifest for the tabular task, built from the dataset config and bank_mlp."""

from __future__ import annotations

import pandas as pd
import torch

from cfl.core.manifest import VALIDATION, as_float32, forward_tests, model_section
from cfl.tasks.tabular import model as tabular_model
from cfl.tasks.tabular.preprocessing import DatasetConfig, output_feature_names, transform

MANIFEST_VERSION = "1.0.0"

DATASET_VERSION = "bank-additional-full-uci-222"
SOURCE_ID = "UCI ML Repository dataset 222; CC BY 4.0"

PREPROCESSING = {
    "implementation_id": "cfl.tasks.tabular.preprocessing.transform",
    "encoding": "fixed-order one-hot; fixed category vocabularies from this manifest",
    "scaling": "fixed mean/std; pdays sentinel replaced by mean and accompanied by an indicator",
    "fit_policy": "public_reference_statistics",
}

# Synthetic records (never real customers). 001 sits on the means, so any scaling error shows
# up only in 002; together they cover sentinel handling and every categorical block.
REFERENCE_INPUTS: list[tuple[str, dict]] = [
    (
        "bank-marketing-reference-001",
        {
            "age": 40.0241,
            "campaign": 2.56759,
            "pdays": 999,
            "previous": 0.172963,
            "emp.var.rate": 0.0818855,
            "cons.price.idx": 93.5757,
            "cons.conf.idx": -40.5026,
            "euribor3m": 3.62129,
            "nr.employed": 5167.04,
            "job": "unknown",
            "marital": "unknown",
            "education": "unknown",
            "default": "unknown",
            "housing": "unknown",
            "loan": "unknown",
            "contact": "cellular",
            "month": "jan",
            "day_of_week": "mon",
            "poutcome": "nonexistent",
        },
    ),
    (
        "bank-marketing-reference-002",
        {
            "age": 57,
            "campaign": 3,
            "pdays": 6,
            "previous": 1,
            "emp.var.rate": -1.8,
            "cons.price.idx": 92.893,
            "cons.conf.idx": -46.2,
            "euribor3m": 1.299,
            "nr.employed": 5099.1,
            "job": "retired",
            "marital": "married",
            "education": "university.degree",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "telephone",
            "month": "may",
            "day_of_week": "fri",
            "poutcome": "success",
        },
    ),
]


def build_manifest(config: DatasetConfig) -> dict:
    feature_names = output_feature_names(config)
    contract = tabular_model.contract(len(feature_names))

    reference_tests = []
    for test_id, raw in REFERENCE_INPUTS:
        expected = transform(pd.DataFrame([raw]), config)[0]
        reference_tests.append(
            {
                "test_id": test_id,
                "synthetic_raw_input": raw,
                "expected_model_input": as_float32(expected),
            }
        )

    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset": {
            "name": config.name,
            "dataset_version": DATASET_VERSION,
            "source_id": SOURCE_ID,
            "raw_label_column": config.label.column,
        },
        "model": model_section(
            contract,
            task="binary_classification",
            input_dim=len(feature_names),
            output_dim=tabular_model.NUM_CLASSES,
        ),
        "preprocessing": {"version": config.preprocessing_version, **PREPROCESSING},
        "input_schema": {
            "features": [f.model_dump(exclude_none=True) for f in config.features],
            "model_input": {
                "feature_names": feature_names,
                "shape": [1, len(feature_names)],
                "dtype": "float32",
            },
        },
        "label": {
            "column": config.label.column,
            "positive_value": config.label.positive,
            "negative_value": config.label.negative,
            "encoded_values": [0, 1],
            "meaning": (
                "bank customer subscribed to a term deposit; 1=yes (positive), 0=no (negative)"
            ),
        },
        "reference_tests": reference_tests,
        "forward_reference_tests": forward_tests(
            tabular_model.BankMLP(len(feature_names)),
            contract,
            [
                (t["test_id"].replace("reference", "forward"), t["expected_model_input"])
                for t in reference_tests
            ],
            torch.float32,
        ),
        "validation": VALIDATION,
    }
