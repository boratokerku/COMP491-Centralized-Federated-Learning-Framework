"""Run the coordinator process for distributed mode (state in PostgreSQL, models on disk).

On start it resumes every RUNNING run from its persisted state (never from round 0), then keeps
stepping them; the gateway serves clients from the same database and model directory. Kill it
at any time and start it again to resume.

Usage (from the repository root, after `alembic upgrade head`):
    export CFL_DATABASE_URL=postgresql+psycopg://cfl:<password>@127.0.0.1:5432/cfl
    python backend/scripts/run_coordinator.py --models /data/models
    python backend/scripts/run_coordinator.py --models /data/models --start backend/configs/x.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cfl.coordinator.config import load_run_config
from cfl.coordinator.coordinator import Coordinator, Step
from cfl.core.store import FileModelStore
from cfl.db.state_store import PostgresStateStore, engine_from_env

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=Path, required=True, help="model storage directory")
    parser.add_argument("--start", type=Path, default=None, help="start a new run from this config")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="seconds between steps")
    args = parser.parse_args()

    coordinator = Coordinator(PostgresStateStore(engine_from_env()), FileModelStore(args.models))
    for info in coordinator.resume():
        status = "resumed" if info.ok else f"FAILED ({info.detail})"
        print(f"run {info.run_id}: round {info.round_id} {info.round_state} -> {status}")

    if args.start is not None:
        path = args.start if args.start.is_absolute() else REPO_ROOT / args.start
        run = coordinator.start_run(load_run_config(path))
        print(f"started run {run.run_id} ({run.name})")

    last: dict[str, Step] = {}

    def log(run_id: str, step: Step) -> None:
        if last.get(run_id) is not step:
            print(f"run {run_id}: {step.value}", flush=True)
            last[run_id] = step

    try:
        coordinator.serve(poll_interval_s=args.poll_interval, on_step=log, resume=False)
    except KeyboardInterrupt:
        print("stopped; start again to resume")
    return 0


if __name__ == "__main__":
    sys.exit(main())
