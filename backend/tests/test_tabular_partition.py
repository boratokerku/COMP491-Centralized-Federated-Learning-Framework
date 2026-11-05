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
        pd.testing.assert_frame_equal(ca.val, cb.val)
        pd.testing.assert_frame_equal(ca.test, cb.test)


def test_different_seed_gives_different_partitions():
    a = iid_partition(_frame(), CONFIG, num_clients=4, seed=1)
    b = iid_partition(_frame(), CONFIG, num_clients=4, seed=2)
    assert _ids(a.test) != _ids(b.test)


def test_splits_cover_all_rows_without_overlap():
    df = _frame()
    p = iid_partition(df, CONFIG, num_clients=4, seed=0)
    parts = [p.test] + [getattr(c, s) for c in p.clients for s in ("train", "val", "test")]
    all_ids = [i for part in parts for i in part["row_id"]]
    assert len(all_ids) == len(set(all_ids)) == len(df)
    assert len(p.test) == 200
    for c in p.clients:  # 200 rows per client: 70% train, 10% val, 20% test
        assert (len(c.train), len(c.val), len(c.test)) == (140, 20, 40)


def test_class_ratio_is_kept_in_every_split():
    p = iid_partition(_frame(), CONFIG, num_clients=4, seed=0)
    for part in [p.test] + [getattr(c, s) for c in p.clients for s in ("train", "val", "test")]:
        assert (part["y"] == "yes").mean() == pytest.approx(0.2, abs=0.03)


def test_single_client():
    p = iid_partition(_frame(), CONFIG, num_clients=1, seed=0)
    assert len(p.clients) == 1
    c = p.clients[0]
    assert len(c.train) + len(c.val) + len(c.test) == 800


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_clients": 0},
        {"test_size": 0.0},
        {"local_test_size": 1.0},
        {"local_val_size": 0.0},
        {"local_val_size": 0.5, "local_test_size": 0.5},
    ],
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
    assert sorted(summary["clients"]["client_0"]) == ["test", "train", "val"]
    assert summary["preprocessing_version"] == "1.0"
    for split in ("train", "val", "test"):
        part = pd.read_csv(out / "client_0" / f"{split}.csv", sep=";")
        assert list(part.columns) == ["row_id", "x", "y"]

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
    """Reproducibility at the file level: a site that re-runs the partition script with the
    published seed gets byte-identical CSVs, not just equal row sets."""
    written = []
    for name in ("run_a", "run_b"):
        p = iid_partition(_frame(n=500), CONFIG, num_clients=3, seed=42)
        out = tmp_path / name
        write_partitions(p, CONFIG, out, summarize(p, CONFIG, seed=42, num_clients=3))
        written.append(out)
    a, b = written
    files = sorted(f.relative_to(a) for f in a.rglob("*") if f.is_file())
    assert len(files) == 3 * 3 + 2  # train/val/test per client, held-out test.csv, summary
    for f in files:
        assert (a / f).read_bytes() == (b / f).read_bytes(), f


def test_format_summary_lists_every_split():
    from cfl.tasks.partition import format_summary

    p = iid_partition(_frame(), CONFIG, num_clients=2, seed=0)
    text = format_summary(summarize(p, CONFIG, seed=0))
    for split in ("client_0 train", "client_0 val", "client_1 test", "held-out test (server)"):
        assert split in text
    assert "total rows: 1,000" in text
