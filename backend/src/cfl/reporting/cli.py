"""`cfl-report <run_id>`: write the report of a finished run.

The database comes from CFL_DATABASE_URL and the model volume from CFL_MODELS_DIR (or
--models-dir). Until the coordinator calls `generate_report` itself when a run completes,
run this by hand.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cfl.core.store import FileModelStore
from cfl.db.state_store import PostgresStateStore, engine_from_env
from cfl.reporting.generate import generate_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cfl-report", description=__doc__.splitlines()[0])
    parser.add_argument("run_id")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=os.environ.get("CFL_MODELS_DIR"),
        help="model volume root (default: $CFL_MODELS_DIR)",
    )
    args = parser.parse_args(argv)
    if not args.models_dir:
        parser.error("set --models-dir or CFL_MODELS_DIR")

    try:
        keys = generate_report(
            PostgresStateStore(engine_from_env()), FileModelStore(args.models_dir), args.run_id
        )
    except KeyError:
        print(f"cfl-report: run {args.run_id} not found in the database", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"cfl-report: {exc}", file=sys.stderr)
        return 1
    for key in keys:
        print(Path(args.models_dir) / key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
