import json

import numpy as np
import pandas as pd
import pytest

from cfl.tasks.tabular.partition import iid_partition, summarize, write_partitions
from cfl.tasks.tabular.preprocessing import DatasetConfig

CONFIG = DatasetConfig.model_validate(
    {
        "name": "toy",
        "preprocessing_version": "1.0",
        "file": "toy.csv",
        "separator": ";",
        "label": {"column": "y", "positive": "yes", "negative": "no"},
        "features": [{"name": "x", "type": "numeric", "unit": "u", "mean": 0.0, "std": 1.0}],
    }
)


def _frame(n: int = 1000, positive_ratio: float = 0.2) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n_pos = int(n * positive_ratio)
    return pd.DataFrame(
        {
            "row_id": np.arange(n),
            "x": rng.normal(size=n),
            "y": ["yes"] * n_pos + ["no"] * (n - n_pos),  # sorted, like the real file by date
        }
    )


def _ids(part: pd.DataFrame) -> set[int]:
    return set(part["row_id"])


def test_same_seed_gives_identical_partitions():
    a = iid_partition(_frame(), CONFIG, num_clients=4, seed=7)
    b = iid_partition(_frame(), CONFIG, num_clients=4, seed=7)
    pd.testing.assert_frame_equal(a.test, b.test)
    for ca, cb in zip(a.clients, b.clients, strict=True):
        pd.testing.assert_frame_equal(ca.train, cb.train)
        pd.testing.assert_frame_equal(ca.test, cb.test)


def test_different_seed_gives_different_partitions():
    a = iid_partition(_frame(), CONFIG, num_clients=4, seed=1)
    b = iid_partition(_frame(), CONFIG, num_clients=4, seed=2)
    assert _ids(a.test) != _ids(b.test)


def test_splits_cover_all_rows_without_overlap():
    df = _frame()
    p = iid_partition(df, CONFIG, num_clients=4, seed=0)
    parts = [p.test] + [c.train for c in p.clients] + [c.test for c in p.clients]
    all_ids = [i for part in parts for i in part["row_id"]]
    assert len(all_ids) == len(set(all_ids)) == len(df)
    assert len(p.test) == 200
    assert all(len(c.test) > 0 for c in p.clients)


def test_class_ratio_is_kept_in_every_split():
    p = iid_partition(_frame(), CONFIG, num_clients=4, seed=0)
    for part in [p.test] + [c.train for c in p.clients] + [c.test for c in p.clients]:
        assert (part["y"] == "yes").mean() == pytest.approx(0.2, abs=0.03)


def test_single_client():
    p = iid_partition(_frame(), CONFIG, num_clients=1, seed=0)
    assert len(p.clients) == 1
    assert len(p.clients[0].train) + len(p.clients[0].test) == 800


@pytest.mark.parametrize(
    "kwargs", [{"num_clients": 0}, {"test_size": 0.0}, {"local_test_size": 1.0}]
)
def test_invalid_arguments(kwargs):
    args = {"num_clients": 2, "seed": 0} | kwargs
    with pytest.raises(ValueError):
        iid_partition(_frame(), CONFIG, **args)


def test_write_and_rewrite(tmp_path):
    out = tmp_path / "parts"
    p = iid_partition(_frame(), CONFIG, num_clients=3, seed=0)
    write_partitions(p, CONFIG, out, summarize(p, CONFIG, seed=0, num_clients=3))

    summary = json.loads((out / "summary.json").read_text())
    assert summary["seed"] == 0
    assert sorted(summary["clients"]) == ["client_0", "client_1", "client_2"]
    train = pd.read_csv(out / "client_0" / "train.csv", sep=";")
    assert list(train.columns) == ["row_id", "x", "y"]

    # Rewriting with fewer clients removes the stale client folder.
    p2 = iid_partition(_frame(), CONFIG, num_clients=2, seed=0)
    write_partitions(p2, CONFIG, out, summarize(p2, CONFIG, seed=0, num_clients=2))
    assert not (out / "client_2").exists()


def test_refuses_to_overwrite_foreign_directory(tmp_path):
    (tmp_path / "notes.txt").write_text("keep me")
    p = iid_partition(_frame(), CONFIG, num_clients=2, seed=0)
    with pytest.raises(FileExistsError):
        write_partitions(p, CONFIG, tmp_path, summarize(p, CONFIG))
    assert (tmp_path / "notes.txt").exists()


def test_same_seed_produces_identical_files_on_disk(tmp_path):
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    df = _frame(n=500, positive_ratio=0.2)

    pa = iid_partition(df, CONFIG, num_clients=3, seed=42)
    write_partitions(pa, CONFIG, dir_a, summarize(pa, CONFIG, seed=42, num_clients=3))

    pb = iid_partition(df, CONFIG, num_clients=3, seed=42)
    write_partitions(pb, CONFIG, dir_b, summarize(pb, CONFIG, seed=42, num_clients=3))

    assert (dir_a / "test.csv").read_bytes() == (dir_b / "test.csv").read_bytes()
    for i in range(3):
        assert (dir_a / f"client_{i}/train.csv").read_bytes() == (dir_b / f"client_{i}/train.csv").read_bytes()
        assert (dir_a / f"client_{i}/test.csv").read_bytes() == (dir_b / f"client_{i}/test.csv").read_bytes()
    assert (dir_a / "summary.json").read_bytes() == (dir_b / "summary.json").read_bytes()

