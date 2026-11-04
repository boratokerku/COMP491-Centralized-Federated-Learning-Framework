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
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path

import torch

from cfl import __version__
from cfl.coordinator.aggregator import (
    AggregationResult,
    Aggregator,
    RoundInput,
    UpdateRecord,
    global_model_key,
)
from cfl.coordinator.config import RunConfig
from cfl.coordinator.state import RoundState, is_terminal
from cfl.coordinator.state_store import (
    EventRecord,
    MetricRecord,
    ParticipantRecord,
    ParticipantStatus,
    RoundRecord,
    RunRecord,
    RunStatus,
    StateStore,
)
from cfl.core.artifact import (
    ARTIFACT,
    MANIFEST,
    MODEL_ONNX,
    MODEL_SAFETENSORS,
    ArtifactError,
    build_artifact,
    final_key,
)
from cfl.core.contract import initial_weights
from cfl.core.evaluation import HeldOutEvaluator
from cfl.core.serialization import bytes_to_state_dict, sha256, state_dict_to_bytes
from cfl.core.store import ModelStore
from cfl.core.strategy import build_strategy
from cfl.protocol import PROTOCOL_VERSION
from cfl.protocol.messages import TaskAssignment
from cfl.tasks.registry import get_task

Clock = Callable[[], datetime]

REPO_ROOT = Path(__file__).resolve().parents[4]


class Step(StrEnum):
    """What one call of Coordinator.step() did."""

    OPENED = "opened"  # a new round is collecting
    WAITING_FOR_CLIENTS = "waiting_for_clients"  # too few eligible clients to open a round
    COLLECTING = "collecting"  # round open, waiting for updates or the deadline
    DONE = "done"  # round aggregated, global model advanced
    FAILED = "failed"  # round failed (quorum, aggregation or evaluation), model unchanged
    COMPLETED = "completed"  # all rounds attempted, final model delivered, run finished
    RUN_FAILED = "run_failed"  # the run cannot continue (e.g. final model export failed)
    IDLE = "idle"  # run is not running


@dataclass(frozen=True)
class ResumedRun:
    run_id: str
    round_id: int | None
    round_state: RoundState | None
    deadline_passed: bool
    ok: bool  # False: the run's model file is missing or corrupt and the run was failed
    detail: str | None = None


WAITING_STEPS = frozenset(
    {Step.COLLECTING, Step.WAITING_FOR_CLIENTS, Step.COMPLETED, Step.RUN_FAILED, Step.IDLE}
)


def _metric_rows(
    run_id: str, round_id: int, values: dict[str, float], at: datetime, client_id=None
) -> list[MetricRecord]:
    """Seconds become milliseconds; units follow the `metrics` table conventions."""
    rows = []
    for name, value in values.items():
        if name.endswith("_s"):
            name, value, unit = f"{name[:-2]}_ms", value * 1000.0, "ms"
        elif name.endswith("bytes") or name.endswith("_read"):
            unit = "bytes"
        elif name.startswith(("updates_", "total_")):
            unit = "count"
        else:
            unit = "value"
        rows.append(MetricRecord(run_id, name, float(value), unit, at, round_id, client_id))
    return rows


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
    def __init__(
        self,
        state: StateStore,
        models: ModelStore,
        clock: Clock = _utcnow,
        repo_root: Path = REPO_ROOT,
    ):
        self.state = state
        self.models = models
        self.clock = clock
        self.repo_root = repo_root
        self._aggregators: dict[str, Aggregator] = {}

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

    def resume(self) -> list[ResumedRun]:
        """Called once when the coordinator process starts.

        For every RUNNING run: verify that its current global model (and the input model of an
        unfinished round) exists and matches the recorded checksum, then record
        COORDINATOR_RESUMED. The run then simply continues through step(): a COLLECTING round
        keeps the updates that the gateway accepted during the outage and gets the normal
        deadline rule; an AGGREGATING round is aggregated again. A run whose model file is
        missing or corrupt cannot continue and is marked FAILED instead of restarting from v0.
        """
        resumed = []
        now = self.clock()
        for run in self.state.running_runs():
            round_ = self.state.latest_round(run.run_id)
            open_round = round_ is not None and not is_terminal(round_.state)
            files = [(run.model_path, run.model_sha256)]
            if open_round:
                files.append((round_.input_model_path, round_.input_model_sha256))
            problem = next(
                (p for p in (self._file_problem(path, sha) for path, sha in files) if p), None
            )
            info = ResumedRun(
                run_id=run.run_id,
                round_id=round_.round_id if round_ else None,
                round_state=round_.state if round_ else None,
                deadline_passed=bool(open_round and now >= round_.deadline_at),
                ok=problem is None,
                detail=problem,
            )
            if problem is not None:
                self.state.update_run(
                    run.run_id,
                    EventRecord(
                        "RUN_FAILED",
                        now,
                        run.run_id,
                        payload={"reason_code": "model_file_invalid", "detail": problem},
                    ),
                    status=RunStatus.FAILED,
                    failure_reason="model_file_invalid",
                    completed_at=now,
                )
            else:
                self.state.add_event(
                    EventRecord(
                        "COORDINATOR_RESUMED",
                        now,
                        run.run_id,
                        info.round_id,
                        payload={
                            "round_state": info.round_state.value if info.round_state else None,
                            "deadline_passed": info.deadline_passed,
                            "model_version": run.model_version,
                        },
                    )
                )
            resumed.append(info)
        return resumed

    def run_until_idle(self, run_id: str, max_steps: int = 1000) -> Step:
        """Take steps until the run has to wait for clients, the deadline or nothing at all."""
        for _ in range(max_steps):
            step = self.step(run_id)
            if step in WAITING_STEPS:
                return step
        raise RuntimeError(f"run {run_id} did not settle within {max_steps} steps")

    def serve(
        self,
        poll_interval_s: float = 1.0,
        should_stop: Callable[[], bool] = lambda: False,
        on_step: Callable[[str, Step], None] | None = None,
        resume: bool = True,
    ) -> None:
        """The coordinator process in distributed mode: resume, then keep stepping every
        running run. All decisions are persisted, so the process can be killed at any time."""
        if resume:
            self.resume()
        while not should_stop():
            for run in self.state.running_runs():
                step = self.run_until_idle(run.run_id)
                if on_step is not None:
                    on_step(run.run_id, step)
            time.sleep(poll_interval_s)

    def _file_problem(self, key: str, expected_sha256: str) -> str | None:
        try:
            data = self.models.get(key)
        except KeyError:
            return f"{key} is missing"
        if sha256(data) != expected_sha256:
            return f"{key} does not match its recorded checksum"
        return None

    def active_task(self, client_id: str) -> TaskAssignment | None:
        """`GET /task` without a run id: the client's task in whichever run is running."""
        for run in self.state.running_runs():
            task = self.task_for(client_id, run.run_id)
            if task is not None:
                return task
        return None

    def step(self, run_id: str) -> Step:
        """Advance the run by at most one state-machine step, based only on persisted state.

        Call it repeatedly (e.g. every second). Because every decision is re-derived from the
        StateStore, the same call also resumes a run after a coordinator restart: a round left
        in OPEN moves on to COLLECTING, one left in AGGREGATING is aggregated again (which is
        deterministic and writes byte-identical output).
        """
        run = self.state.get_run(run_id)
        if run.status is not RunStatus.RUNNING:
            return Step.IDLE
        config = RunConfig.model_validate(run.config)
        round_ = self.state.latest_round(run_id)

        if round_ is None or is_terminal(round_.state):
            if round_ is not None and round_.round_id >= config.rounds.num_rounds:
                return self._complete_run(run, config)
            opened = self.open_round(run_id)
            return Step.OPENED if opened is not None else Step.WAITING_FOR_CLIENTS

        if round_.state is RoundState.OPEN:  # crashed between creating and opening the round
            self.state.transition_round(
                run_id,
                round_.round_id,
                round_.state_version,
                RoundState.COLLECTING,
                EventRecord("ROUND_COLLECTING", self.clock(), run_id, round_.round_id),
            )
            return Step.OPENED
        if round_.state is RoundState.COLLECTING:
            return self._close_if_ready(run, round_)
        return self._aggregate(run, config, round_)  # AGGREGATING

    def _close_if_ready(self, run: RunRecord, round_: RoundRecord) -> Step:
        """All selected clients reported, or the deadline passed: mark missing clients as
        dropouts, then aggregate if the quorum is met or fail the round."""
        now = self.clock()
        participants = self.state.participants(run.run_id, round_.round_id)
        waiting = [p.client_id for p in participants if p.status is ParticipantStatus.SELECTED]
        deadline_passed = now >= round_.deadline_at
        if waiting and not deadline_passed:
            return Step.COLLECTING

        for client_id in waiting:  # only reached once the deadline has passed
            self.state.set_participant_status(
                run.run_id,
                round_.round_id,
                client_id,
                ParticipantStatus.DROPOUT,
                now,
                EventRecord(
                    "CLIENT_DROPOUT",
                    now,
                    run.run_id,
                    round_.round_id,
                    client_id,
                    {"reason_code": "deadline_no_update"},
                ),
                reason_code="deadline_no_update",
            )
        received = len(self.state.updates(run.run_id, round_.round_id))
        if received < round_.min_participants:
            self._fail_round(
                round_, "quorum_not_met", f"{received} updates, need {round_.min_participants}"
            )
            return Step.FAILED

        self.state.transition_round(
            run.run_id,
            round_.round_id,
            round_.state_version,
            RoundState.AGGREGATING,
            EventRecord(
                "ROUND_AGGREGATING",
                now,
                run.run_id,
                round_.round_id,
                payload={"received": received, "dropouts": len(waiting)},
            ),
        )
        aggregating = self.state.get_round(run.run_id, round_.round_id)
        return self._aggregate(run, RunConfig.model_validate(run.config), aggregating)

    def _aggregate(self, run: RunRecord, config: RunConfig, round_: RoundRecord) -> Step:
        started = self.clock()
        updates = self.state.updates(run.run_id, round_.round_id)
        result = self._aggregator(run, config).aggregate(
            RoundInput(
                run_id=run.run_id,
                round_id=round_.round_id,
                input_version=round_.input_model_version,
                input_key=round_.input_model_path,
                input_sha256=round_.input_model_sha256,
                updates=[
                    UpdateRecord(
                        u.update_id,
                        u.client_id,
                        u.model_version,
                        u.num_samples,
                        u.weights_path,
                        u.checksum_sha256,
                    )
                    for u in updates
                ],
                min_participants=round_.min_participants,
                max_num_samples=config.rounds.max_num_samples,
            )
        )
        now = self.clock()
        by_update = {u.update_id: u for u in updates}
        for update_id, reason in result.rejected.items():
            client_id = by_update[update_id].client_id
            self.state.set_participant_status(
                run.run_id,
                round_.round_id,
                client_id,
                ParticipantStatus.REJECTED,
                now,
                EventRecord(
                    "UPDATE_EXCLUDED",
                    now,
                    run.run_id,
                    round_.round_id,
                    client_id,
                    {"update_id": update_id, "reason_code": reason.value},
                ),
                reason_code=reason.value,
            )

        if result.state is RoundState.FAILED:
            self._fail_round(round_, result.reason.value, result.detail, result)
            return Step.FAILED

        timing = {
            "round_duration_s": (now - round_.opened_at).total_seconds(),
            "waiting_s": (started - round_.opened_at).total_seconds(),
        }
        metrics = _metric_rows(run.run_id, round_.round_id, result.metrics | timing, now)
        for u in updates:
            metrics += _metric_rows(
                run.run_id,
                round_.round_id,
                {f"client_{k}": v for k, v in u.metrics.items()},
                now,
                client_id=u.client_id,
            )
        self.state.transition_round(
            run.run_id,
            round_.round_id,
            round_.state_version,
            RoundState.DONE,
            EventRecord(
                "ROUND_DONE",
                now,
                run.run_id,
                round_.round_id,
                payload={
                    "used": result.used,
                    "rejected": {k: v.value for k, v in result.rejected.items()},
                    "output_model_version": result.output_version,
                    "output_model_sha256": result.output_sha256,
                },
            ),
            run_fields={
                "model_version": result.output_version,
                "model_path": result.output_key,
                "model_sha256": result.output_sha256,
            },
            metrics=metrics,
            output_model_version=result.output_version,
            output_model_path=result.output_key,
            output_model_sha256=result.output_sha256,
            closed_at=now,
        )
        return Step.DONE

    def _fail_round(
        self,
        round_: RoundRecord,
        reason: str,
        detail: str | None,
        result: AggregationResult | None = None,
    ) -> None:
        now = self.clock()
        current = self.state.get_round(round_.run_id, round_.round_id)
        metrics = (
            _metric_rows(round_.run_id, round_.round_id, result.metrics, now) if result else []
        )
        self.state.transition_round(
            round_.run_id,
            round_.round_id,
            current.state_version,
            RoundState.FAILED,
            EventRecord(
                "ROUND_FAILED",
                now,
                round_.run_id,
                round_.round_id,
                payload={"reason_code": reason, "detail": detail},
            ),
            metrics=metrics,
            failure_reason=reason,
            closed_at=now,
        )

    def _complete_run(self, run: RunRecord, config: RunConfig) -> Step:
        """Package the final global model (ONNX verified against PyTorch), write the files,
        then mark the run COMPLETED with a pointer to them. Re-running after a crash rebuilds
        and overwrites the same files atomically."""
        now = self.clock()
        task = get_task(config.task.name)
        try:
            model_bytes = self.models.get(run.model_path)
            if sha256(model_bytes) != run.model_sha256:
                raise ArtifactError(f"{run.model_path} does not match its recorded checksum")
            artifact = build_artifact(
                run_id=run.run_id,
                model_version=run.model_version,
                weights=bytes_to_state_dict(model_bytes),
                build_model=task.build_model,
                contract=task.contract(),
                manifest_bytes=self.models.get(manifest_key(run.run_id)),
                created_at=now,
            )
        except (ArtifactError, KeyError, ValueError) as exc:
            self.state.update_run(
                run.run_id,
                EventRecord(
                    "RUN_FAILED",
                    now,
                    run.run_id,
                    payload={"reason_code": "final_model_export_failed", "detail": str(exc)},
                ),
                status=RunStatus.FAILED,
                failure_reason="final_model_export_failed",
                completed_at=now,
            )
            return Step.RUN_FAILED

        # artifact.json last: its presence means the package is complete.
        for name in (MODEL_ONNX, MODEL_SAFETENSORS, MANIFEST, ARTIFACT):
            self.models.put(final_key(run.run_id, name), artifact.files[name])
        self.state.update_run(
            run.run_id,
            EventRecord(
                "RUN_COMPLETED",
                now,
                run.run_id,
                payload={
                    "model_version": run.model_version,
                    "final_model_sha256": artifact.onnx_sha256,
                    "artifact_sha256": sha256(artifact.files[ARTIFACT]),
                    "onnx_max_abs_diff": artifact.descriptor["verification"][
                        "max_abs_diff_onnx_vs_pytorch"
                    ],
                },
            ),
            status=RunStatus.COMPLETED,
            completed_at=now,
            final_model_path=final_key(run.run_id, MODEL_ONNX),
            final_model_sha256=artifact.onnx_sha256,
            final_model_format="onnx",
            final_model_bytes=len(artifact.files[MODEL_ONNX]),
        )
        return Step.COMPLETED

    def final_artifact(self, run_id: str) -> dict[str, bytes] | None:
        """What `GET /model/{run_id}/final` serves: the artifact files of a completed run."""
        run = self.state.get_run(run_id)
        if run.status is not RunStatus.COMPLETED or run.final_model_path is None:
            return None
        names = (MODEL_ONNX, MODEL_SAFETENSORS, MANIFEST, ARTIFACT)
        return {name: self.models.get(final_key(run_id, name)) for name in names}

    def _aggregator(self, run: RunRecord, config: RunConfig) -> Aggregator:
        """Built once per run: the held-out set is loaded and the strategy configured."""
        if run.run_id not in self._aggregators:
            task = get_task(config.task.name)
            manifest = json.loads(self.models.get(manifest_key(run.run_id)))
            dtype = torch.int64 if task.name == "text" else torch.float32
            sanity = torch.tensor(
                [t["model_input"] for t in manifest["forward_reference_tests"]], dtype=dtype
            )
            held_out = task.held_out_path(self.repo_root / config.task.data_dir)
            strategy_kwargs = {"mu": config.strategy.mu} if config.strategy.mu is not None else {}
            self._aggregators[run.run_id] = Aggregator(
                strategy=build_strategy(config.strategy.name, **strategy_kwargs),
                contract=task.contract(),
                store=self.models,
                build_model=task.build_model,
                evaluator=HeldOutEvaluator(task.build_model, *task.load_split(held_out)),
                sanity_inputs=sanity,
            )
        return self._aggregators[run.run_id]
