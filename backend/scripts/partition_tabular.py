"""Create seeded IID client partitions for a tabular dataset (proposal T1.4 / D1.5).

Output (gitignored), default data/tabular/partitions/<dataset>/:
    client_<i>/train.csv, client_<i>/test.csv   one folder per simulated client site
    test.csv                                    held-out set for experiments
    summary.json                                settings and per-split class balance

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config", type=Path, default=Path("backend/configs/datasets/bank_marketing.yaml")
    )
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2, help="held-out share")
    parser.add_argument("--local-test-size", type=float, default=0.2, help="per-client share")
    parser.add_argument("--out", type=Path, default=None)
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
    print(json.dumps(summary, indent=2))
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
