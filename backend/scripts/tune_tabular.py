"""Run the Bank Marketing hyperparameter grid (federated, FedAvg, local simulation).

Usage (from the repository root):
    python backend/scripts/tune_tabular.py
    python backend/scripts/tune_tabular.py --seeds 42 43 44 --workers 4

The data is imbalanced (11.3% "yes"), so the score is F1, ROC-AUC and PR-AUC on the server's
held-out test set, never accuracy. The grid is the product of the value lists below; every
setting runs once per seed (the seed changes model initialisation, data order and dropout, the
partition stays the one written by partition_tabular.py). Everything else comes from
sim_tabular.yaml.

Besides the held-out metrics each run records the mean client validation F1 / PR-AUC of the last
round (the clients' own val splits, measured after local training) and the number of rounds it
took to converge: the first round after which every later global F1 stays within 0.005 of the
final one.

Written under results/tabular_tuning/: runs.csv (one row per run), summary.csv (mean and
standard deviation per setting) and results.json.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from cfl.coordinator.config import RunConfig, load_run_config
from cfl.simulation.runner import run_simulation

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("backend/configs/sim_tabular.yaml")
DEFAULT_OUTPUT = Path("results/tabular_tuning")
STABLE_F1_TOLERANCE = 0.005

GRID = {
    "learning_rate": (0.001, 0.003, 0.01),
    "local_epochs": (1, 2),
    "batch_size": (32, 64),
    "class_weights": ((1.0, 3.0), (1.0, 5.0), (1.0, 7.9)),
}
METRICS = ("f1", "roc_auc", "pr_auc", "accuracy", "precision", "recall")
VAL_METRICS = ("f1", "pr_auc")


@dataclass(frozen=True)
class Setting:
    learning_rate: float
    local_epochs: int
    batch_size: int
    class_weights: tuple[float, float]

    @property
    def name(self) -> str:
        weight = f"{self.class_weights[1]:g}"
        return f"lr{self.learning_rate:g}_ep{self.local_epochs}_bs{self.batch_size}_w{weight}"


def settings() -> list[Setting]:
    return [Setting(*values) for values in itertools.product(*GRID.values())]


def make_config(base: RunConfig, setting: Setting, seed: int) -> RunConfig:
    raw = base.model_dump()
    raw["run"]["name"] = f"{base.run.name}-{setting.name}-seed{seed}"
    raw["run"]["seed"] = seed
    raw["client"]["learning_rate"] = setting.learning_rate
    raw["client"]["local_epochs"] = setting.local_epochs
    raw["client"]["batch_size"] = setting.batch_size
    raw["client"]["class_weights"] = list(setting.class_weights)
    return RunConfig.model_validate(raw)


def stable_round(history: list[tuple[int, float]]) -> int | None:
    """First round whose remaining F1 values all stay within the tolerance of the final F1."""
    if not history:
        return None
    final = history[-1][1]
    for index, (round_id, _) in enumerate(history):
        if all(abs(value - final) <= STABLE_F1_TOLERANCE for _, value in history[index:]):
            return round_id
    return None


def run_one(job: tuple[Setting, int, dict]) -> dict:
    import torch

    torch.set_num_threads(1)  # several runs in parallel; one thread each is faster overall
    setting, seed, raw_config = job
    config = RunConfig.model_validate(raw_config)
    result = run_simulation(config, repo_root=REPO_ROOT)
    run_id = result.run.run_id
    global_by_round: dict[int, dict[str, float]] = {}
    client_val: dict[int, dict[str, list[float]]] = {}
    for m in result.state.metrics(run_id):
        if m.round_id is None:
            continue
        if m.client_id is None and m.metric_name.startswith("global_"):
            global_by_round.setdefault(m.round_id, {})[m.metric_name[7:]] = m.value
        elif m.client_id is not None and m.metric_name.startswith("client_val_"):
            name = m.metric_name.removeprefix("client_val_").removesuffix("_after")
            if name in VAL_METRICS and m.metric_name.endswith("_after"):
                client_val.setdefault(m.round_id, {}).setdefault(name, []).append(m.value)

    rounds = sorted(global_by_round)
    history = [(r, global_by_round[r]["f1"]) for r in rounds]
    final = global_by_round[rounds[-1]] if rounds else {}
    row = {
        "setting": setting.name,
        "seed": seed,
        "learning_rate": setting.learning_rate,
        "local_epochs": setting.local_epochs,
        "batch_size": setting.batch_size,
        "class_weights": json.dumps(list(setting.class_weights)),
        "status": result.run.status.value,
        "rounds_completed": len(rounds),
        "rounds_to_converge": stable_round(history),
        "verified_clients": len(result.verified_by),
        "run_id": run_id,
    }
    for metric in METRICS:
        row[f"{metric}_final"] = final.get(metric)
    last_val = client_val.get(rounds[-1], {}) if rounds else {}
    for metric in VAL_METRICS:
        values = last_val.get(metric, [])
        row[f"val_{metric}_final"] = statistics.mean(values) if values else None
    return row


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for setting in settings():
        group = [r for r in rows if r["setting"] == setting.name]
        if not group:
            continue
        summary = {
            "setting": setting.name,
            "learning_rate": setting.learning_rate,
            "local_epochs": setting.local_epochs,
            "batch_size": setting.batch_size,
            "class_weights": json.dumps(list(setting.class_weights)),
            "runs": len(group),
            "completed_runs": sum(r["status"] == "COMPLETED" for r in group),
        }
        keys = [f"{m}_final" for m in METRICS] + [f"val_{m}_final" for m in VAL_METRICS]
        for key in keys:
            values = [r[key] for r in group if r[key] is not None]
            summary[f"{key}_mean"] = statistics.mean(values) if values else None
            summary[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        converge = [r["rounds_to_converge"] for r in group if r["rounds_to_converge"]]
        summary["rounds_to_converge_mean"] = statistics.mean(converge) if converge else None
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=None, help="override configured rounds")
    parser.add_argument("--workers", type=int, default=4, help="runs in parallel")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if len(args.seeds) < 3:
        parser.error("use at least 3 seeds per setting")

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    base = load_run_config(config_path)
    if base.task.name != "tabular":
        parser.error("this runner needs a tabular-task config")
    if args.rounds is not None:
        raw = base.model_dump()
        raw["rounds"]["num_rounds"] = args.rounds
        base = RunConfig.model_validate(raw)
    partitions = REPO_ROOT / base.task.data_dir / "partitions" / "bank_marketing"
    if not (partitions / "test.csv").is_file():
        parser.error(f"partitions not found at {partitions}; run partition_tabular.py first")

    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    jobs = [
        (s, seed, make_config(base, s, seed).model_dump())
        for s in settings()
        for seed in args.seeds
    ]
    print(f"{len(jobs)} runs ({len(settings())} settings x {len(args.seeds)} seeds)", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, row in enumerate(pool.map(run_one, jobs), start=1):
            rows.append(row)
            print(
                f"[{index}/{len(jobs)}] {row['setting']} seed={row['seed']} {row['status']} "
                f"f1={row['f1_final']:.3f} roc_auc={row['roc_auc_final']:.3f} "
                f"pr_auc={row['pr_auc_final']:.3f} converge={row['rounds_to_converge']}",
                flush=True,
            )

    summaries = summarize(rows)
    write_csv(output / "runs.csv", rows)
    write_csv(output / "summary.csv", summaries)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(config_path.relative_to(REPO_ROOT)),
        "num_rounds": base.rounds.num_rounds,
        "seeds": args.seeds,
        "partition_seed": 42,
        "stable_f1_tolerance": STABLE_F1_TOLERANCE,
        "grid": {k: [list(v) if isinstance(v, tuple) else v for v in vs] for k, vs in GRID.items()},
        "settings": [asdict(s) | {"name": s.name} for s in settings()],
        "runs": rows,
        "summary": summaries,
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote runs.csv, summary.csv and results.json to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
