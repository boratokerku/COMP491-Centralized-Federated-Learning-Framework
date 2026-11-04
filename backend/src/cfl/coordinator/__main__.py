"""Coordinator process for distributed mode: `python -m cfl.coordinator --models /models`.

State lives in PostgreSQL (CFL_DATABASE_URL), model files under --models. On start it resumes
every RUNNING run from its persisted state (never from round 0), then keeps stepping them. It
can be stopped at any time (SIGTERM, Ctrl+C) and started again.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from cfl.coordinator.config import load_run_config
from cfl.coordinator.coordinator import REPO_ROOT, Coordinator, Step
from cfl.core.store import FileModelStore
from cfl.db.state_store import PostgresStateStore, engine_from_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cfl.coordinator", description=__doc__)
    parser.add_argument("--models", type=Path, required=True, help="model storage directory")
    parser.add_argument("--start", type=Path, default=None, help="start a new run from this config")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="seconds between steps")
    parser.add_argument(
        "--no-serve",
        action="store_true",
        help="only start the run and exit; the running coordinator service picks it up",
    )
    args = parser.parse_args(argv)

    # As PID 1 in a container, SIGTERM is ignored unless handled. Stopping is always safe:
    # every decision is already persisted.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    coordinator = Coordinator(PostgresStateStore(engine_from_env()), FileModelStore(args.models))
    for info in [] if args.no_serve else coordinator.resume():
        status = "resumed" if info.ok else f"FAILED ({info.detail})"
        print(f"run {info.run_id}: round {info.round_id} {info.round_state} -> {status}")

    if args.start is not None:
        path = args.start
        if not path.is_absolute() and not path.exists():  # relative to the repository root
            path = REPO_ROOT / path
        run = coordinator.start_run(load_run_config(path))
        print(f"started run {run.run_id} ({run.name})", flush=True)
    if args.no_serve:
        return 0

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
