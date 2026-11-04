"""Run a federated training in local simulation mode and print the per-round results.

Needs the partitions of the configured dataset (partition_tabular.py / partition_text.py).

Usage (from the repository root):
    python backend/scripts/run_simulation.py --config backend/configs/sim_tabular.yaml
    python backend/scripts/run_simulation.py --config backend/configs/sim_text.yaml --rounds 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cfl.coordinator.config import load_run_config
from cfl.coordinator.state import RoundState
from cfl.coordinator.state_store import InMemoryStateStore, ParticipantStatus, RoundRecord
from cfl.core.store import FileModelStore, InMemoryModelStore
from cfl.simulation.runner import run_simulation

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=None, help="override rounds.num_rounds")
    parser.add_argument("--store", type=Path, default=None, help="keep model files here")
    args = parser.parse_args()

    config = load_run_config(args.config if args.config.is_absolute() else REPO_ROOT / args.config)
    if args.rounds is not None:
        raw = config.model_dump()
        raw["rounds"]["num_rounds"] = args.rounds
        config = type(config).model_validate(raw)
    models = FileModelStore(args.store) if args.store else InMemoryModelStore()

    def report(round_: RoundRecord, state: InMemoryStateStore) -> None:
        values = {
            m.metric_name: m.value
            for m in state.metrics(round_.run_id)
            if m.round_id == round_.round_id and m.client_id is None
        }
        participants = state.participants(round_.run_id, round_.round_id)
        used = sum(p.status is ParticipantStatus.UPDATE_ACCEPTED for p in participants)
        if round_.state is RoundState.FAILED:
            print(f"round {round_.round_id:>3}  FAILED ({round_.failure_reason})")
            return
        print(
            f"round {round_.round_id:>3}  v{round_.output_model_version:<3} "
            f"clients={used}/{len(participants)}  "
            f"f1={values['global_f1']:.3f}  roc_auc={values['global_roc_auc']:.3f}  "
            f"pr_auc={values['global_pr_auc']:.3f}  acc={values['global_accuracy']:.3f}  "
            f"round={values['round_duration_ms']:.0f}ms"
        )

    result = run_simulation(config, repo_root=REPO_ROOT, models=models, on_round_closed=report)

    run = result.run
    rounds = [result.state.get_round(run.run_id, i) for i in range(1, config.rounds.num_rounds + 1)]
    done = sum(r.state is RoundState.DONE for r in rounds)
    print(
        f"\nrun {run.run_id} {run.status.value}: {done}/{len(rounds)} rounds done, "
        f"final model v{run.model_version} sha256={run.model_sha256[:16]}…"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
