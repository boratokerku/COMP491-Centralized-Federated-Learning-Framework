"""Create seeded IID client partitions for the SMS Spam Collection.

Exact duplicate messages are removed first (see cfl.tasks.text.data.deduplicate).

Output (gitignored), default data/text/partitions/sms_spam/:
    client_<i>/train.csv, val.csv, test.csv     one folder per simulated client site
    test.csv                                    held-out set for experiments
    summary.json                                settings, de-duplication and class balance

Usage (from the repository root):
    python backend/scripts/download_sms_spam.py
    python backend/scripts/partition_text.py --num-clients 4 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cfl.tasks.partition import format_summary, iid_partition, summarize, write_partitions
from cfl.tasks.text.data import DATASET_NAME, RAW_FILE, deduplicate, encode_labels, read_raw

REPO_ROOT = Path(__file__).resolve().parents[2]


def _from_root(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--raw", type=Path, default=Path(RAW_FILE), help="SMSSpamCollection file (downloaded)"
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

    raw = read_raw(_from_root(args.raw))
    df = deduplicate(raw)
    settings = {
        "seed": args.seed,
        "num_clients": args.num_clients,
        "test_size": args.test_size,
        "local_val_size": args.local_val_size,
        "local_test_size": args.local_test_size,
        "raw_rows": len(raw),
        "duplicates_removed": len(raw) - len(df),
    }
    partitioning = iid_partition(
        df,
        encode_labels,
        num_clients=args.num_clients,
        seed=args.seed,
        test_size=args.test_size,
        local_val_size=args.local_val_size,
        local_test_size=args.local_test_size,
    )
    summary = summarize(partitioning, encode_labels, DATASET_NAME, **settings)
    out = _from_root(args.out) if args.out else REPO_ROOT / "data/text/partitions" / DATASET_NAME
    write_partitions(partitioning, out, summary)
    print(format_summary(summary, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
