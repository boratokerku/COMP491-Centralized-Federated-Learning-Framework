"""Create seeded IID client partitions for a tabular dataset (proposal T1.4 / D1.5).

Output (gitignored), default data/tabular/partitions/<dataset>/:
    client_<i>/train.csv, val.csv, test.csv     one folder per simulated client site
    test.csv                                    held-out set for experiments
    summary.json                                settings and per-split class balance

Usage (from the repository root):
    python backend/scripts/partition_tabular.py --num-clients 4 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cfl.tasks.partition import format_summary
from cfl.tasks.tabular.partition import iid_partition, summarize, write_partitions
from cfl.tasks.tabular.preprocessing import load_dataset_config, read_raw

REPO_ROOT = Path(__file__).resolve().parents[2]


def _from_root(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("backend/configs/datasets/bank_marketing.yaml"),
        help="dataset config (columns, categories, scaling statistics)",
    )
    parser.add_argument("--num-clients", type=int, default=4, help="number of client sites")
    parser.add_argument("--seed", type=int, default=42, help="seed; same seed, same files")
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="share held out as the server's test set"
    )
    parser.add_argument(
        "--local-val-size", type=float, default=0.1, help="share of each client's data for val"
    )
    parser.add_argument(
        "--local-test-size", type=float, default=0.2, help="share of each client's data for test"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="output folder (default: data/<modality>/partitions)"
    )
    args = parser.parse_args()

    config = load_dataset_config(_from_root(args.config))
    df = read_raw(config, REPO_ROOT)
    settings = {
        "seed": args.seed,
        "num_clients": args.num_clients,
        "test_size": args.test_size,
        "local_val_size": args.local_val_size,
        "local_test_size": args.local_test_size,
    }
    partitioning = iid_partition(
        df,
        config,
        num_clients=args.num_clients,
        seed=args.seed,
        test_size=args.test_size,
        local_val_size=args.local_val_size,
        local_test_size=args.local_test_size,
    )
    summary = summarize(partitioning, config, **settings)
    out = _from_root(args.out) if args.out else REPO_ROOT / "data/tabular/partitions" / config.name
    write_partitions(partitioning, config, out, summary)
    print(format_summary(summary, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
