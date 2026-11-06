import numpy as np
import pandas as pd
import pytest

from cfl.coordinator.coordinator import _metric_rows
from cfl.tasks.partition import label_skew_partition


def _frame(n: int = 2000) -> pd.DataFrame:
    return pd.DataFrame({"row_id": np.arange(n), "y": np.arange(n) % 5 == 0})


def _labels(df: pd.DataFrame) -> np.ndarray:
    return df["y"].to_numpy().astype(np.int64)


def _ids(part: pd.DataFrame) -> set[int]:
    return set(part["row_id"])


def test_label_skew_gives_different_class_ratios_and_keeps_every_row():
    pool, test = _frame(), _frame(10)
    parts = label_skew_partition(pool, test, _labels, num_clients=4, seed=1, alpha=0.2)
    rows = [pd.concat([c.train, c.val, c.test]) for c in parts.clients]

    ratios = [_labels(r).mean() for r in rows]
    assert max(ratios) - min(ratios) > 0.2
    assert sum(len(r) for r in rows) == len(pool)
    assert set.union(*(_ids(r) for r in rows)) == set(pool["row_id"])
    pd.testing.assert_frame_equal(parts.test, test)


def test_label_skew_is_reproducible():
    a = label_skew_partition(_frame(), _frame(10), _labels, 4, seed=7, alpha=0.3)
    b = label_skew_partition(_frame(), _frame(10), _labels, 4, seed=7, alpha=0.3)
    for ca, cb in zip(a.clients, b.clients, strict=True):
        pd.testing.assert_frame_equal(ca.train, cb.train)


def test_label_skew_needs_a_positive_alpha():
    with pytest.raises(ValueError, match="alpha"):
        label_skew_partition(_frame(), _frame(10), _labels, 4, seed=1, alpha=0)


def test_millisecond_metrics_are_stored_as_ms():
    rows = _metric_rows("run", 1, {"rtt_ms": 2.5, "train_s": 1.0, "train_loss": 0.3}, None)
    assert {r.metric_name: r.unit for r in rows} == {
        "rtt_ms": "ms",
        "train_ms": "ms",
        "train_loss": "value",
    }
