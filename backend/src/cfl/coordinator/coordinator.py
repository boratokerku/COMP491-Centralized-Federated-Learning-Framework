"""Coordinator: starts runs and drives their rounds through the state machine.

It never talks to clients directly. Everything it decides is written to the StateStore
(PostgreSQL in distributed mode) and the ModelStore (model files); the gateway, or the in-memory
transport in simulation, reads that state to answer clients. This keeps the coordinator
restartable: all it needs to continue is in the two stores.
"""

from __future__ import annotations

import json
import math
import random
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from cfl import __version__
from cfl.coordinator.aggregator import global_model_key
from cfl.coordinator.config import RunConfig
from cfl.coordinator.state import RoundState, is_terminal
from cfl.coordinator.state_store import (
    EventRecord,
    ParticipantRecord,
    ParticipantStatus,
    RoundRecord,
    RunRecord,
    RunStatus,
    StateStore,
)
from cfl.core.contract import initial_weights
from cfl.core.serialization import sha256, state_dict_to_bytes
from cfl.core.store import ModelStore
from cfl.core.strategy import build_strategy
from cfl.protocol import PROTOCOL_VERSION
from cfl.protocol.messages import TaskAssignment
from cfl.tasks.registry import get_task

Clock = Callable[[], datetime]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def manifest_key(run_id: str) -> str:
    return f"{run_id}/manifest.json"


def select_clients(eligible: list[str], fraction: float, minimum: int, seed: int, round_id: int):
    """Seeded partial participation: the same run seed and round always pick the same clients,
    independent of the order in which clients registered."""
    pool = sorted(eligible)
    k = min(len(pool), max(minimum, math.ceil(fraction * len(pool))))
    return sorted(random.Random(f"{seed}:{round_id}").sample(pool, k))


def round_config(config: RunConfig) -> dict:
    """The `config` object of a TaskAssignment: everything a client needs for local training."""
    strategy = build_strategy(
        config.strategy.name,
        **({"mu": config.strategy.mu} if config.strategy.mu is not None else {}),
    )
    return {
        "strategy": config.strategy.name,
        "local_epochs": config.client.local_epochs,
        "batch_size": config.client.batch_size,
        "learning_rate": config.client.learning_rate,
        "weight_decay": config.client.weight_decay,
        "proximal_mu": 0.0,
        "class_weights": config.client.class_weights,
        **strategy.client_config(),
    }


class Coordinator:
    def __init__(self, state: StateStore, models: ModelStore, clock: Clock = _utcnow):
        self.state = state
        self.models = models
        self.clock = clock

    def start_run(self, config: RunConfig, run_id: str | None = None) -> RunRecord:
        """Create a run: pin its manifest, write the initial global model v0 and record both.

        Files are written before the run row that points to them, so a crash in between leaves
        only an unreferenced file, never a run that points to a missing model.
        """
        run_id = run_id or str(uuid.uuid4())
        task = get_task(config.task.name)
        contract = task.contract()
        manifest = task.build_manifest()
        if manifest["model"]["fingerprint"] != contract.fingerprint():
            raise RuntimeError("manifest and model contract disagree; rebuild the manifest")

        manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
        self.models.put(manifest_key(run_id), manifest_bytes)

        weights = initial_weights(task.build_model, seed=config.run.seed)
        contract.check(weights)
        model_bytes = state_dict_to_bytes(weights)
        model_path = global_model_key(run_id, 0)
        self.models.put(model_path, model_bytes)

        now = self.clock()
        run = RunRecord(
            run_id=run_id,
            name=config.run.name,
            status=RunStatus.RUNNING,
            strategy=config.strategy.name,
            task_name=config.task.name,
            seed=config.run.seed,
            config=config.model_dump(),
            config_sha256=config.fingerprint(),
            protocol_version=PROTOCOL_VERSION,
            manifest_version=manifest["manifest_version"],
            manifest_sha256=sha256(manifest_bytes),
            preprocessing_version=task.preprocessing_version,
            model_architecture=contract.architecture_id,
            model_architecture_version=contract.architecture_version,
            model_fingerprint=contract.fingerprint(),
            model_version=0,
            model_path=model_path,
            model_sha256=sha256(model_bytes),
            created_at=now,
            started_at=now,
        )
        self.state.create_run(
            run,
            EventRecord(
                "RUN_STARTED",
                now,
                run_id=run_id,
                payload={
                    "software_version": __version__,
                    "protocol_version": PROTOCOL_VERSION,
                    "mode": config.run.mode,
                    "task": config.task.name,
                    "strategy": config.strategy.name,
                    "config_sha256": run.config_sha256,
                    "manifest_version": run.manifest_version,
                    "manifest_sha256": run.manifest_sha256,
                    "preprocessing_version": run.preprocessing_version,
                    "model": f"{run.model_architecture}/{run.model_architecture_version}",
                    "model_fingerprint": run.model_fingerprint,
                    "initial_model_sha256": run.model_sha256,
                },
            ),
        )
        return run

    def open_round(self, run_id: str) -> RoundRecord | None:
        """Open the next round of a running run and move it to COLLECTING.

        Returns None when the run already has all its rounds or too few eligible clients are
        available yet (the caller tries again later, e.g. while clients are still registering).
        """
        run = self.state.get_run(run_id)
        if run.status is not RunStatus.RUNNING:
            raise ValueError(f"run {run_id} is {run.status}, not RUNNING")
        config = RunConfig.model_validate(run.config)

        last = self.state.latest_round(run_id)
        if last is not None and not is_terminal(last.state):
            raise RuntimeError(f"round {last.round_id} is still {last.state}")
        round_id = 1 if last is None else last.round_id + 1
        if round_id > config.rounds.num_rounds:
            return None

        eligible = self.state.eligible_clients(run.manifest_version)
        if len(eligible) < config.rounds.min_participants:
            return None
        selected = select_clients(
            eligible,
            config.rounds.client_fraction,
            config.rounds.min_participants,
            run.seed,
            round_id,
        )

        now = self.clock()
        round_ = RoundRecord(
            run_id=run_id,
            round_id=round_id,
            state=RoundState.OPEN,
            input_model_version=run.model_version,
            input_model_path=run.model_path,
            input_model_sha256=run.model_sha256,
            min_participants=config.rounds.min_participants,
            deadline_at=now + timedelta(seconds=config.rounds.round_deadline_s),
            opened_at=now,
        )
        participants = [
            ParticipantRecord(run_id, round_id, cid, ParticipantStatus.SELECTED, now)
            for cid in selected
        ]
        self.state.create_round(
            round_,
            participants,
            EventRecord(
                "ROUND_OPENED",
                now,
                run_id=run_id,
                round_id=round_id,
                payload={
                    "selected": selected,
                    "eligible": len(eligible),
                    "input_model_version": run.model_version,
                    "deadline_at": round_.deadline_at.isoformat(),
                },
            ),
        )
        return self.state.transition_round(
            run_id,
            round_id,
            expected_version=round_.state_version,
            target=RoundState.COLLECTING,
            event=EventRecord("ROUND_COLLECTING", now, run_id=run_id, round_id=round_id),
        )

    def task_for(self, client_id: str, run_id: str) -> TaskAssignment | None:
        """What `GET /task` returns: the active round's assignment if this client was selected
        and has not finished it, otherwise None (HTTP 204)."""
        run = self.state.get_run(run_id)
        round_ = self.state.latest_round(run_id)
        if run.status is not RunStatus.RUNNING or round_ is None:
            return None
        if round_.state is not RoundState.COLLECTING:
            return None
        statuses = {p.client_id: p.status for p in self.state.participants(run_id, round_.round_id)}
        if statuses.get(client_id) is not ParticipantStatus.SELECTED:
            return None
        return TaskAssignment(
            run_id=run_id,
            round_id=round_.round_id,
            model_version=round_.input_model_version,
            deadline=round_.deadline_at,
            config=round_config(RunConfig.model_validate(run.config)),
        )
