"""Create client partitions for a tabular dataset (proposal T1.4 / D1.5).

Reads the dataset config, holds out a stratified test set, and splits the rest
into `num_clients` IID partitions (stratified on the label, so every client
gets the same class ratio). It writes raw rows, because preprocessing runs on each client.

Output (gitignored):
    data/tabular/partitions/<dataset>/client_0.csv ... client_{N-1}.csv
    data/tabular/partitions/<dataset>/test.csv
    data/tabular/partitions/<dataset>/summary.json

Usage:
    python scripts/partition_tabular.py \
        --config backend/configs/datasets/bank_marketing.yaml --num-clients 4 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))

from cfl.tasks.tabular.config import load_dataset_config  # noqa: E402
from cfl.tasks.tabular.preprocessing import load_dataframe  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="backend/configs/datasets/bank_marketing.yaml")
    p.add_argument("--num-clients", type=int, default=4)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = load_dataset_config(ROOT / args.config)
    df = load_dataframe(cfg, ROOT)
    y = (df[cfg.label.column] == cfg.label.positive).astype(int)

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    out = Path(args.out) if args.out else ROOT / "data" / "tabular" / "partitions" / cfg.name
    out.mkdir(parents=True, exist_ok=True)

    sep = cfg.read_options.get("sep", ",")
    train_df = train_df.reset_index(drop=True)
    y_train = (train_df[cfg.label.column] == cfg.label.positive).astype(int)
    skf = StratifiedKFold(n_splits=args.num_clients, shuffle=True, random_state=args.seed)

    def stats(part):
        pos = int((part[cfg.label.column] == cfg.label.positive).sum())
        return {"rows": len(part), "positive": pos, "positive_ratio": round(pos / len(part), 4)}

    summary = {
        "dataset": cfg.name,
        "preprocessing_version": cfg.preprocessing_version,
        "partition": "iid_stratified",
        "seed": args.seed,
        "num_clients": args.num_clients,
        "test_size": args.test_size,
        "clients": {},
    }
    for i, (_, idx) in enumerate(skf.split(np.zeros(len(y_train)), y_train)):
        part = train_df.iloc[idx]
        part.to_csv(out / f"client_{i}.csv", sep=sep, index=False)
        summary["clients"][f"client_{i}"] = stats(part)
    test_df.to_csv(out / "test.csv", sep=sep, index=False)
    summary["test"] = stats(test_df)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
