"""Run a small, reproducible SMS hyperparameter screening grid.

Usage (from the repository root or backend directory):
    python backend/scripts/tune_text.py
    python scripts/tune_text.py --seeds 42 43 44

The default grid changes one setting at a time from sim_text.yaml. Dataset partitions stay fixed;
the run seed changes client initialization, data order, and any seeded client selection.
Per-run CSV, grouped summary CSV, and JSON are written under results/text_tuning/.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from cfl.coordinator.config import RunConfig, load_run_config
from cfl.simulation.runner import run_simulation

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path("backend/configs/sim_text.yaml")
DEFAULT_OUTPUT = Path("results/text_tuning")
STABLE_F1_TOLERANCE = 0.005


@dataclass(frozen=True)
class Setting:
    name: str
    learning_rate: float
    local_epochs: int
    batch_size: int
    class_weights: tuple[float, float]


SETTINGS = (
    Setting("baseline", 0.01, 1, 64, (1.0, 6.9)),
    Setting("lr_0_005", 0.005, 1, 64, (1.0, 6.9)),
    Setting("epochs_2", 0.01, 2, 64, (1.0, 6.9)),
    Setting("batch_32", 0.01, 1, 32, (1.0, 6.9)),
    Setting("positive_weight_5", 0.01, 1, 64, (1.0, 5.0)),
)
METRICS = ("global_f1", "global_roc_auc", "global_pr_auc", "global_accuracy")


def _make_config(base: RunConfig, setting: Setting, seed: int) -> RunConfig:
    raw = base.model_dump()
    raw["run"]["name"] = f"{base.run.name}-{setting.name}-seed{seed}"
    raw["run"]["seed"] = seed
    raw["client"]["learning_rate"] = setting.learning_rate
    raw["client"]["local_epochs"] = setting.local_epochs
    raw["client"]["batch_size"] = setting.batch_size
    raw["client"]["class_weights"] = list(setting.class_weights)
    return RunConfig.model_validate(raw)


def _stable_f1_round(history: list[tuple[int, float]]) -> int | None:
    """First round whose remaining F1 values all stay within 0.005 of the final F1."""
    if not history:
        return None
    final = history[-1][1]
    for index, (round_id, _) in enumerate(history):
        if all(
            abs(value - final) <= STABLE_F1_TOLERANCE for _, value in history[index:]
        ):
            return round_id
    return None


def _collect_run(setting: Setting, seed: int, config: RunConfig) -> dict:
    result = run_simulation(config, repo_root=REPO_ROOT)
    by_round: dict[int, dict[str, float]] = {}
    for metric in result.state.metrics(result.run.run_id):
        if metric.client_id is None and metric.round_id is not None:
            by_round.setdefault(metric.round_id, {})[metric.metric_name] = metric.value

    completed = [
        (round_id, values)
        for round_id, values in sorted(by_round.items())
        if "global_f1" in values
    ]
    f1_history = [(round_id, values["global_f1"]) for round_id, values in completed]
    final_round, final_metrics = completed[-1] if completed else (None, {})
    best_round, best_f1 = max(f1_history, key=lambda item: item[1]) if f1_history else (None, None)

    row = {
        "setting": setting.name,
        "seed": seed,
        "learning_rate": setting.learning_rate,
        "local_epochs": setting.local_epochs,
        "batch_size": setting.batch_size,
        "class_weights": json.dumps(list(setting.class_weights)),
        "status": result.run.status.value,
        "rounds_completed": len(completed),
        "final_round": final_round,
        "rounds_to_converge": _stable_f1_round(f1_history),
        "best_f1": best_f1,
        "best_f1_round": best_round,
        "verified_clients": len(result.verified_by),
        "verified_client_ids": json.dumps(result.verified_by),
        "run_id": result.run.run_id,
    }
    for metric_name in METRICS:
        row[metric_name.removeprefix("global_") + "_final"] = final_metrics.get(metric_name)
    return row


def _summary(rows: list[dict]) -> list[dict]:
    summaries = []
    for setting in SETTINGS:
        group = [row for row in rows if row["setting"] == setting.name]
        summary = {
            "setting": setting.name,
            "learning_rate": setting.learning_rate,
            "local_epochs": setting.local_epochs,
            "batch_size": setting.batch_size,
            "class_weights": json.dumps(list(setting.class_weights)),
            "runs": len(group),
        }
        for key in ("f1_final", "roc_auc_final", "pr_auc_final", "accuracy_final", "best_f1"):
            values = [row[key] for row in group if row[key] is not None]
            summary[f"{key}_mean"] = statistics.mean(values) if values else None
            summary[f"{key}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        convergence = [row["rounds_to_converge"] for row in group if row["rounds_to_converge"]]
        summary["rounds_to_converge_mean"] = (
            statistics.mean(convergence) if convergence else None
        )
        summaries.append(summary)
    return summaries


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--rounds", type=int, default=None, help="override configured rounds")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    base = load_run_config(config_path)
    if base.task.name != "text":
        parser.error("the tuning runner requires a text-task config")
    if args.rounds is not None:
        raw = base.model_dump()
        raw["rounds"]["num_rounds"] = args.rounds
        base = RunConfig.model_validate(raw)

    partition_dir = REPO_ROOT / base.task.data_dir / "partitions" / "sms_spam"
    if not (partition_dir / "test.csv").is_file():
        parser.error(f"SMS partitions not found at {partition_dir}; run partition_text.py first")

    output_dir = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    total = len(SETTINGS) * len(args.seeds)
    rows = []
    for index, (setting, seed) in enumerate(
        ((setting, seed) for setting in SETTINGS for seed in args.seeds), start=1
    ):
        config = _make_config(base, setting, seed)
        print(
            f"[{index}/{total}] {setting.name} seed={seed} "
            f"lr={setting.learning_rate} epochs={setting.local_epochs} "
            f"batch={setting.batch_size} class_weights={list(setting.class_weights)}",
            flush=True,
        )
        row = _collect_run(setting, seed, config)
        rows.append(row)
        print(
            f"  status={row['status']} rounds={row['rounds_completed']} "
            f"f1={row['f1_final']} roc_auc={row['roc_auc_final']} "
            f"pr_auc={row['pr_auc_final']} converge_round={row['rounds_to_converge']}",
            flush=True,
        )
        gc.collect()

    summaries = _summary(rows)
    _write_csv(output_dir / "runs.csv", rows)
    _write_csv(output_dir / "summary.csv", summaries)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "config": str(config_path.relative_to(REPO_ROOT)),
        "num_rounds": base.rounds.num_rounds,
        "seeds": args.seeds,
        "partition_seed": 42,
        "stable_f1_tolerance": STABLE_F1_TOLERANCE,
        "settings": [asdict(setting) for setting in SETTINGS],
        "runs": rows,
        "summary": summaries,
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {output_dir / 'runs.csv'}")
    print(f"Wrote {output_dir / 'summary.csv'}")
    print(f"Wrote {output_dir / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
