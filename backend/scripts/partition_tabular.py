"""Create seeded IID client partitions for a tabular dataset (proposal T1.4 / D1.5).

Splits the tabular dataset into:
1. A server-side held-out test set (for global evaluation and centralized baselines).
2. Randomly partitioned IID client shards, where each client shard is further divided into
   local training and local test splits.

Output layout (gitignored), default data/tabular/partitions/<dataset>/:
    client_<i>/train.csv, client_<i>/test.csv   one folder per simulated client site
    test.csv                                    held-out server test set
    summary.json                                partition settings and class balance statistics

Usage (from the repository root):
    python backend/scripts/partition_tabular.py --num-clients 4 --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cfl.tasks.tabular.partition import iid_partition, summarize, write_partitions
from cfl.tasks.tabular.preprocessing import load_dataset_config, read_raw

REPO_ROOT = Path(__file__).resolve().parents[2]


def _from_root(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def print_summary_table(summary: dict, out_dir: Path) -> None:
    """Print a clean human-readable console summary showing sample counts and class balance."""
    dataset_name = summary.get("dataset", "unknown")
    seed = summary.get("seed", "N/A")
    num_clients = summary.get("num_clients", "N/A")

    print("\n" + "=" * 78)
    print(f"Tabular IID Partition Summary: {dataset_name} (seed={seed}, clients={num_clients})")
    print("=" * 78)
    print(f"{'Partition':<30} {'Rows':>10} {'Positive':>12} {'Pos Ratio (%)':>16}")
    print("-" * 78)

    clients = summary.get("clients", {})
    total_rows = 0
    for client_id, splits in sorted(clients.items()):
        for split_type in ("train", "test"):
            stats = splits.get(split_type, {})
            rows = stats.get("rows", 0)
            positive = stats.get("positive", 0)
            ratio = stats.get("positive_ratio", 0.0) * 100
            total_rows += rows
            label = f"{client_id} ({split_type})"
            print(f"{label:<30} {rows:>10,d} {positive:>12,d} {ratio:>15.2f}%")

    test_stats = summary.get("test", {})
    t_rows = test_stats.get("rows", 0)
    t_pos = test_stats.get("positive", 0)
    t_ratio = test_stats.get("positive_ratio", 0.0) * 100
    total_rows += t_rows

    print("-" * 78)
    print(f"{'server (held-out test)':<30} {t_rows:>10,d} {t_pos:>12,d} {t_ratio:>15.2f}%")
    print("=" * 78)
    print(f"Total partitioned rows:      {total_rows:,d}")
    print(f"Partitions written to:       {out_dir}")
    print("=" * 78 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("backend/configs/datasets/bank_marketing.yaml"),
        help="Path to dataset configuration YAML file",
    )
    parser.add_argument("--num-clients", type=int, default=4, help="Number of client partitions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Share of dataset reserved for server test set"
    )
    parser.add_argument(
        "--local-test-size",
        type=float,
        default=0.2,
        help="Share of each client partition reserved for local test set",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Target output directory (default: data/tabular/partitions/<dataset>)",
    )
    args = parser.parse_args()

    config = load_dataset_config(_from_root(args.config))
    df = read_raw(config, REPO_ROOT)
    settings = {
        "seed": args.seed,
        "num_clients": args.num_clients,
        "test_size": args.test_size,
        "local_test_size": args.local_test_size,
    }
    partitioning = iid_partition(
        df,
        config,
        num_clients=args.num_clients,
        seed=args.seed,
        test_size=args.test_size,
        local_test_size=args.local_test_size,
    )
    summary = summarize(partitioning, config, **settings)
    out = _from_root(args.out) if args.out else REPO_ROOT / "data/tabular/partitions" / config.name
    write_partitions(partitioning, config, out, summary)

    # Print human-readable summary table and JSON output
    print_summary_table(summary, out)
    print("Summary JSON:")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
