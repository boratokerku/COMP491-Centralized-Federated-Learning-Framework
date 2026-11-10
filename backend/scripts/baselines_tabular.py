"""Centralized and local-only baselines for the Bank Marketing task, trained with BankTaskSpec.

Centralized    every client's train.csv pooled into one training set, one model
Local only     one model per client, trained on that client's train.csv alone

Both use the hyperparameters of sim_tabular.yaml and train for rounds x local_epochs epochs, so a
baseline sees as many passes over each row as a federated run (the same rule as
run_comparison.py). Each model is scored on the server's held-out test.csv; the local-only
models are also scored on their own client's test.csv. The two local-only views answer different
questions: held-out scores can be compared across methods, client-test scores show performance
on the client's own distribution and must not be pooled with the held-out ones.

The files have the same layout as the text baselines (results/text_baselines/text/), so both
modalities fit one table:
    results.json   settings and every run (cases: centralized, local_only, local_only_client_test)
    summary.csv    mean and standard deviation over the seeds
    report.md      the same table, readable

Needs the partitions written by partition_tabular.py.

Usage (from the repository root):
    python backend/scripts/baselines_tabular.py
    python backend/scripts/baselines_tabular.py --seeds 42 43 44 --out results/tabular_baselines
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch
from run_comparison import METRICS, _write

from cfl.coordinator.config import load_run_config
from cfl.core.task import Split
from cfl.tasks.tabular.spec import BankTaskSpec

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "backend" / "configs" / "sim_tabular.yaml"
PARTITIONS = "bank_marketing"


def train_model(spec: BankTaskSpec, data: Split, client: dict, epochs: int, seed: int):
    torch.manual_seed(seed)
    model = spec.build_model()
    config = {**client, "local_epochs": epochs, "proximal_mu": 0.0}
    spec.train(model, data, config, None, seed)
    return model


def mean_metrics(results: list[dict]) -> dict[str, float]:
    return {k: statistics.mean(r[k] for r in results) for k in METRICS if k in results[0]}


def run_seed(spec: BankTaskSpec, root: Path, clients: int, client: dict, epochs: int, seed: int):
    held_out = spec.load_split(root / "test.csv")
    splits = [spec.load_split(root / f"client_{i}" / "train.csv") for i in range(clients)]
    pooled = (torch.cat([s[0] for s in splits]), torch.cat([s[1] for s in splits]))
    centralized = spec.evaluate(train_model(spec, pooled, client, epochs, seed), held_out)

    on_held_out, on_own_test = [], []
    for i, split in enumerate(splits):
        model = train_model(spec, split, client, epochs, seed)
        on_held_out.append(spec.evaluate(model, held_out))
        on_own_test.append(spec.evaluate(model, spec.load_split(root / f"client_{i}" / "test.csv")))
    return {
        "centralized": {"final": centralized},
        "local_only": {
            "final": mean_metrics(on_held_out),
            "worst_client_f1": min(r["f1"] for r in on_held_out),
            "best_client_f1": max(r["f1"] for r in on_held_out),
        },
        "local_only_client_test": {
            "final": mean_metrics(on_own_test),
            "worst_client_f1": min(r["f1"] for r in on_own_test),
            "best_client_f1": max(r["f1"] for r in on_own_test),
            "per_client": {f"client_{i}": r for i, r in enumerate(on_own_test)},
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=None, help="override rounds.num_rounds")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "tabular_baselines")
    args = parser.parse_args()

    config = load_run_config(CONFIG)
    rounds = args.rounds or config.rounds.num_rounds
    client = config.client.model_dump()
    epochs = client["local_epochs"] * rounds
    root = REPO_ROOT / config.task.data_dir / "partitions" / PARTITIONS
    if not (root / "test.csv").is_file():
        parser.error(f"partitions not found at {root}; run partition_tabular.py first")

    spec = BankTaskSpec()
    runs: dict[str, list[dict]] = defaultdict(list)
    for seed in args.seeds:
        print(f"tabular seed={seed} baselines ({epochs} epochs)", flush=True)
        for case, result in run_seed(
            spec, root, config.task.num_clients, client, epochs, seed
        ).items():
            runs[case].append({"seed": seed, **result})

    settings = {"seeds": args.seeds, "rounds": args.rounds, "baselines_only": True}
    _write(args.out / "tabular", "tabular", runs, settings)
    print((args.out / "tabular" / "report.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
