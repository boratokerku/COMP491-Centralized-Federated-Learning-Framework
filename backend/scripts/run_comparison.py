"""Compare FedAvg, FedProx, centralized and local-only training on the same held-out test set.

For each task and seed:
    fedavg / fedprox (mu)   IID partitions, in local simulation mode
    the same on label_skew  non-IID clients from a Dirichlet split of the same rows
    centralized             one model on all clients' training data together
    local_only              each client alone, reported as mean, min and max over the clients

Centralized and local-only train for rounds x local_epochs epochs, so every method gets the same
amount of training per row. All numbers come from the server's held-out test.csv.

Output, one folder per task under results/comparison/:
    results.json   every run, with the global metrics per round
    summary.csv    mean and standard deviation over the seeds
    report.md      the same table, readable
    curves.png     F1 per round

Needs the IID partitions (partition_tabular.py / partition_text.py).

Usage (from the repository root):
    python backend/scripts/run_comparison.py
    python backend/scripts/run_comparison.py --tasks text --seeds 42 --rounds 5
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from cfl.coordinator.config import RunConfig, load_run_config
from cfl.simulation.runner import run_simulation
from cfl.simulation.trainer import evaluate, train_local
from cfl.tasks.partition import (
    Partitioning,
    label_skew_partition,
    summarize,
    write_partitions,
)
from cfl.tasks.registry import TaskDefinition, get_task
from cfl.tasks.tabular.preprocessing import encode_labels as encode_tabular_labels
from cfl.tasks.tabular.preprocessing import load_dataset_config
from cfl.tasks.text.data import encode_labels as encode_text_labels

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = REPO_ROOT / "backend" / "configs"
METRICS = ("f1", "roc_auc", "pr_auc", "accuracy", "precision", "recall")


def _override(config: RunConfig, **sections) -> RunConfig:
    raw = config.model_dump()
    for section, values in sections.items():
        raw[section] = {**raw[section], **values} if section != "strategy" else values
    return RunConfig.model_validate(raw)


def _read_format(task: str) -> tuple[Callable[[pd.DataFrame], object], str]:
    if task == "tabular":
        config = load_dataset_config(CONFIGS / "datasets" / "bank_marketing.yaml")
        return (lambda df: encode_tabular_labels(df, config)), config.separator
    return encode_text_labels, ","


def _make_label_skew(config: RunConfig, task: TaskDefinition, alpha: float) -> str:
    """Write the non-IID partitions next to the IID ones and return their data_dir."""
    labels, sep = _read_format(config.task.name)
    read = lambda path: pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)  # noqa: E731
    source = REPO_ROOT / config.task.data_dir / "partitions" / task.dataset
    pool = pd.concat(
        read(source / f"client_{i}" / f"{split}.csv")
        for i in range(config.task.num_clients)
        for split in ("train", "val", "test")
    )
    parts: Partitioning = label_skew_partition(
        pool, read(source / "test.csv"), labels, config.task.num_clients, 42, alpha
    )
    data_dir = f"{config.task.data_dir}/label_skew"
    summary = summarize(
        parts, labels, task.dataset, partition="label_skew_dirichlet", alpha=alpha, seed=42
    )
    write_partitions(parts, REPO_ROOT / data_dir / "partitions" / task.dataset, summary, sep=sep)
    return data_dir


def _federated(config: RunConfig) -> dict:
    result = run_simulation(config, repo_root=REPO_ROOT)
    run_id = result.run.run_id
    per_round: dict[int, dict[str, float]] = defaultdict(dict)
    for m in result.state.metrics(run_id):
        if m.client_id is None and m.metric_name.startswith("global_"):
            per_round[m.round_id][m.metric_name.removeprefix("global_")] = m.value
    rounds = [per_round[r] for r in sorted(per_round)]
    return {"final": rounds[-1], "rounds": rounds, "status": result.run.status.value}


def _train_alone(
    config: RunConfig, task: TaskDefinition, inputs: torch.Tensor, labels: torch.Tensor, seed: int
) -> torch.nn.Module:
    client = config.client.model_dump()
    client["local_epochs"] = config.client.local_epochs * config.rounds.num_rounds
    torch.manual_seed(seed)
    model = task.build_model()
    train_local(model, inputs, labels, client, seed=seed)
    return model


def _baselines(config: RunConfig, task: TaskDefinition, seed: int) -> dict[str, dict]:
    root = REPO_ROOT / config.task.data_dir / "partitions" / task.dataset
    test = task.load_split(root / "test.csv")
    splits = [
        task.load_split(root / f"client_{i}" / "train.csv") for i in range(config.task.num_clients)
    ]
    pooled = (torch.cat([s[0] for s in splits]), torch.cat([s[1] for s in splits]))
    centralized = evaluate(_train_alone(config, task, *pooled, seed), *test)

    alone = []
    for i, split in enumerate(splits):
        model = _train_alone(config, task, *split, seed)
        own_test = task.load_split(root / f"client_{i}" / "test.csv")
        alone.append(
            {
                "held_out_test": evaluate(model, *test),
                "client_test": evaluate(model, *own_test),
            }
        )
    held_out = [result["held_out_test"] for result in alone]
    client_test = [result["client_test"] for result in alone]
    return {
        "centralized": {"final": centralized},
        "local_only": {
            "final": {
                k: statistics.mean(a[k] for a in held_out) for k in METRICS if k in held_out[0]
            },
            "worst_client_f1": min(a["f1"] for a in held_out),
            "best_client_f1": max(a["f1"] for a in held_out),
        },
        "local_only_client_test": {
            "final": {
                k: statistics.mean(a[k] for a in client_test)
                for k in METRICS
                if k in client_test[0]
            },
            "worst_client_f1": min(a["f1"] for a in client_test),
            "best_client_f1": max(a["f1"] for a in client_test),
            "per_client": {
                f"client_{i}": result["client_test"] for i, result in enumerate(alone)
            },
        },
    }


def compare(
    task_name: str,
    seeds: list[int],
    rounds: int | None,
    mus: list[float],
    alpha: float,
    baselines_only: bool = False,
):
    config = load_run_config(CONFIGS / f"sim_{task_name}.yaml")
    if rounds is not None:
        config = _override(config, rounds={"num_rounds": rounds})
    task = get_task(task_name)
    skewed = None if baselines_only else _make_label_skew(config, task, alpha)

    strategies = [] if baselines_only else [("fedavg", {"name": "fedavg", "mu": None})]
    if not baselines_only:
        strategies += [(f"fedprox_mu{mu}", {"name": "fedprox", "mu": mu}) for mu in mus]
    runs: dict[str, list[dict]] = defaultdict(list)
    for seed in seeds:
        for name, strategy in strategies:
            for suffix, data_dir, partition in (
                ("iid", config.task.data_dir, "iid"),
                ("label_skew", skewed, "label_skew"),
            ):
                case = f"{name}_{suffix}"
                print(f"{task_name} seed={seed} {case}", flush=True)
                run = _override(
                    config,
                    run={"name": case, "seed": seed},
                    strategy=strategy,
                    task={"data_dir": data_dir, "partition": partition},
                )
                runs[case].append({"seed": seed, **_federated(run)})
        print(f"{task_name} seed={seed} baselines", flush=True)
        for case, result in _baselines(config, task, seed).items():
            runs[case].append({"seed": seed, **result})
    return runs


def _table(runs: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for case, results in runs.items():
        row: dict = {"case": case, "seeds": len(results)}
        for metric in METRICS:
            values = [r["final"][metric] for r in results if metric in r["final"]]
            if values:
                row[f"{metric}_mean"] = statistics.mean(values)
                row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        worst = [r["worst_client_f1"] for r in results if "worst_client_f1" in r]
        if worst:
            row["worst_client_f1_mean"] = statistics.mean(worst)
        rows.append(row)
    return rows


def _write(out: Path, task: str, runs: dict[str, list[dict]], settings: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"settings": settings, "runs": runs}, indent=2))
    rows = _table(runs)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [f"# Comparison: {task}", "", f"Settings: `{json.dumps(settings)}`", ""]
    lines += ["| Case | F1 | ROC-AUC | PR-AUC | Accuracy |", "|---|---|---|---|---|"]
    cell = lambda r, m: f"{r[f'{m}_mean']:.3f} ± {r[f'{m}_std']:.3f}"  # noqa: E731
    for r in rows:
        values = " | ".join(cell(r, m) for m in ("f1", "roc_auc", "pr_auc", "accuracy"))
        lines.append(f"| `{r['case']}` | {values} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")

    curve_data = {
        case: [[x["f1"] for x in r["rounds"]] for r in results if "rounds" in r]
        for case, results in runs.items()
    }
    curve_data = {case: curves for case, curves in curve_data.items() if curves}
    if curve_data:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for case, curves in curve_data.items():
            mean = [statistics.mean(v) for v in zip(*curves, strict=True)]
            ax.plot(range(1, len(mean) + 1), mean, label=case)
        ax.set(
            xlabel="Round", ylabel="Global F1 (held-out test set)", title=f"{task}: F1 per round"
        )
        ax.legend(fontsize=7)
        fig.savefig(out / "curves.png", dpi=130, bbox_inches="tight")
        plt.close(fig)
    else:
        (out / "curves.png").unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tasks", nargs="+", default=["tabular", "text"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=None, help="override rounds.num_rounds")
    parser.add_argument("--mus", nargs="+", type=float, default=[0.01, 0.1], help="FedProx mu")
    parser.add_argument("--alpha", type=float, default=0.3, help="Dirichlet alpha of label skew")
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="run only centralized and local-only baselines (skip federated comparisons)",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "comparison")
    args = parser.parse_args()

    for task in args.tasks:
        settings = {"seeds": args.seeds, "rounds": args.rounds}
        if not args.baselines_only:
            settings.update({"mus": args.mus, "alpha": args.alpha})
        settings["baselines_only"] = args.baselines_only
        runs = compare(task, args.seeds, args.rounds, args.mus, args.alpha, args.baselines_only)
        _write(args.out / task, task, runs, settings)
        print((args.out / task / "report.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
