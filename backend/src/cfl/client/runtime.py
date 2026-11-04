"""Shared client runtime: everything a participating site does except the modality-specific
work, which it delegates to a TaskSpec (planning/interfaces.md, section 2).

    register -> conformance report -> loop:
        resend a pending update, if any
        poll for a task; when selected: download the model, evaluate on val, train,
            evaluate again, write the update to disk, upload it
        when a run this site took part in has finished: download the final model and verify it

Crash safety. The update (weights + metadata with its update_id) is written to the state
directory before the upload starts and removed only once the gateway has decided. After a
crash at any point the site starts again and resends the same update_id; the gateway counts it
once. A crash during training only costs the training: the task is still assigned and is
trained again with the same seed.

All network errors are retried by the transport; when the gateway stays unreachable the runtime
waits and polls again instead of exiting, so a site survives a server restart.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from cfl.client.http import GatewayHTTPError, GatewayUnavailable
from cfl.core.artifact import verify_artifact
from cfl.core.contract import ModelContract
from cfl.core.serialization import bytes_to_state_dict, sha256, state_dict_to_bytes
from cfl.core.task import TaskSpec
from cfl.core.transport import Transport
from cfl.protocol import PROTOCOL_VERSION
from cfl.protocol.messages import RegisterRequest, TaskAssignment, UpdateMetadata

# Upload answers after which the update must not be sent again.
FINAL_REJECTIONS = frozenset(
    {
        "stale_round",
        "deadline_passed",
        "model_version_mismatch",
        "duplicate_update",
        "not_selected",
        "unknown_round",
        "checksum_mismatch",
        "invalid_update_id",
    }
)


class ConformanceFailed(RuntimeError):
    def __init__(self, reason_code: str, message: str, details: dict):
        super().__init__(f"{reason_code}: {message}")
        self.reason_code = reason_code
        self.details = details


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


@dataclass
class _State:
    """Persisted in <state_dir>/state.json."""

    rounds_done: list[list] = field(default_factory=list)  # [run_id, round_id]
    runs: list[str] = field(default_factory=list)  # runs this site trained in
    finals_verified: list[str] = field(default_factory=list)


class ClientRuntime:
    def __init__(
        self,
        *,
        transport: Transport,
        task: TaskSpec,
        client_id: str,
        client_version: str,
        dataset: str,
        partition_dir: Path,
        state_dir: Path,
        poll_interval_s: float = 3.0,
        seed: int = 0,
        log: Callable[[str], None] = lambda message: print(message, flush=True),
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.transport = transport
        self.task = task
        self.client_id = client_id
        self.client_version = client_version
        self.dataset = dataset
        self.state_dir = Path(state_dir)
        self.poll_interval_s = poll_interval_s
        self.seed = seed
        self.log = log
        self.sleep = sleep
        self.train_data = task.load_split(Path(partition_dir) / "train.csv")
        self.val_data = task.load_split(Path(partition_dir) / "val.csv")
        self.state = self._load_state()
        self.manifest_version: str | None = None

    # -- persistence ------------------------------------------------------------------------------
    @property
    def _state_file(self) -> Path:
        return self.state_dir / "state.json"

    @property
    def _pending_dir(self) -> Path:
        return self.state_dir / "pending"

    def _load_state(self) -> _State:
        try:
            return _State(**json.loads(self._state_file.read_text()))
        except FileNotFoundError:
            return _State()

    def _save_state(self) -> None:
        _atomic_write(self._state_file, json.dumps(self.state.__dict__, indent=2).encode())

    def _write_pending(self, meta: UpdateMetadata, weights: bytes) -> None:
        # Weights first: metadata on disk means the pending update is complete.
        _atomic_write(self._pending_dir / f"{meta.update_id}.safetensors", weights)
        _atomic_write(self._pending_dir / f"{meta.update_id}.json", meta.model_dump_json().encode())

    def _pending(self) -> list[tuple[UpdateMetadata, bytes]]:
        found = []
        for meta_file in sorted(self._pending_dir.glob("*.json")):
            meta = UpdateMetadata.model_validate_json(meta_file.read_text())
            weights = (self._pending_dir / f"{meta.update_id}.safetensors").read_bytes()
            found.append((meta, weights))
        return found

    def _clear_pending(self, update_id: str) -> None:
        for suffix in (".json", ".safetensors"):
            (self._pending_dir / f"{update_id}{suffix}").unlink(missing_ok=True)

    # -- lifecycle --------------------------------------------------------------------------------
    def register(self) -> None:
        response = self.transport.register(
            RegisterRequest(
                client_id=self.client_id,
                client_version=self.client_version,
                protocol_version=PROTOCOL_VERSION,
            )
        )
        self.log(f"registered as {self.client_id} ({self.client_version})")
        if response.manifest_version:
            self.manifest_version = response.manifest_version

    def pass_conformance(self) -> bool:
        """Submit a report for this site's dataset. False while no run uses that dataset yet;
        raises ConformanceFailed when the gateway rejects the report."""
        list_manifests = getattr(self.transport, "list_manifests", None)
        if list_manifests is None:  # in-memory simulation: no conformance step
            return True
        from cfl.conformance.report import build_report

        mine = [m for m in list_manifests() if m.dataset == self.dataset]
        if not mine:
            return False
        manifest_bytes, manifest_sha = self.transport.get_manifest(mine[0].manifest_version)
        manifest = json.loads(manifest_bytes)
        contract = ModelContract.from_model(
            self.task.build_model(),
            manifest["model"]["architecture_id"],
            manifest["model"]["architecture_version"],
        )
        report = build_report(
            manifest=manifest,
            manifest_sha256=manifest_sha,
            client_id=self.client_id,
            client_version=self.client_version,
            preprocess=self.task.preprocess,
            build_model=self.task.build_model,
            contract=contract,
        )
        result = self.transport.submit_report(report)
        if result.status != "PASSED":
            raise ConformanceFailed(result.reason_code, result.message or "", result.details)
        self.manifest_version = mine[0].manifest_version
        self.log(f"conformance passed for {self.manifest_version}")
        return True

    def run(
        self,
        should_stop: Callable[[], bool] = lambda: False,
        exit_after_final: bool = False,
    ) -> None:
        """Take part until stopped. With exit_after_final, return once the final model of the
        latest run this site trained in has been verified, unless it already was before this
        call: older runs fetched on the way (e.g. one that finished while the site was offline)
        do not count, and a site restarted after a crash in the last round still stops."""
        verified_at_start = set(self.state.finals_verified)
        self._call_until_done(self.register)
        while not should_stop() and not self._call_until_done(self.pass_conformance):
            self.log(f"no running run for {self.dataset} yet; waiting")
            self.sleep(self.poll_interval_s)

        while not should_stop():
            try:
                worked = self.step()
            except GatewayUnavailable as exc:
                self.log(f"gateway unreachable, will retry: {exc}")
                worked = False
            latest = self.state.runs[-1] if self.state.runs else None
            if (
                exit_after_final
                and latest in self.state.finals_verified
                and latest not in verified_at_start
            ):
                return
            if not worked:
                self.sleep(self.poll_interval_s)

    def step(self) -> bool:
        """One polling cycle. Returns True when it did something (so the caller polls again
        immediately)."""
        for meta, weights in self._pending():
            return self._upload(meta, weights)  # undecided: wait a poll interval, then resend

        assignment = self.transport.poll_task(self.client_id)
        if assignment is not None and [assignment.run_id, assignment.round_id] not in (
            self.state.rounds_done
        ):
            self._train_and_upload(assignment)
            return True
        return self._fetch_finished_runs()

    # -- one round --------------------------------------------------------------------------------
    def _train_and_upload(self, task: TaskAssignment) -> None:
        self.log(
            f"round {task.round_id} of run {task.run_id[:8]}: training on v{task.model_version}"
        )
        global_bytes = self.transport.download_model(task.run_id, task.model_version)
        global_weights = bytes_to_state_dict(global_bytes)
        model = self.task.build_model()
        model.load_state_dict(global_weights)

        before = self.task.evaluate(model, self.val_data)
        start = time.perf_counter()
        stats = self.task.train(
            model,
            self.train_data,
            task.config,
            global_weights,
            seed=self.seed * 100_003 + task.round_id,
        )
        train_s = time.perf_counter() - start
        after = self.task.evaluate(model, self.val_data)

        metrics = {**stats, "train_s": train_s}
        metrics |= {f"val_{k}_before": v for k, v in before.items()}
        metrics |= {f"val_{k}_after": v for k, v in after.items()}
        metrics |= self._transfer_metrics()
        weights = state_dict_to_bytes(model.state_dict())
        meta = UpdateMetadata(
            run_id=task.run_id,
            round_id=task.round_id,
            client_id=self.client_id,
            update_id=str(uuid.uuid4()),
            model_version=task.model_version,
            num_samples=len(self.train_data[1]),
            checksum_sha256=sha256(weights),
            metrics=metrics,
        )
        self._write_pending(meta, weights)  # from here on a crash resends this exact update
        if task.run_id in self.state.runs:  # keep the latest run last
            self.state.runs.remove(task.run_id)
        self.state.runs.append(task.run_id)
        self._save_state()
        self._upload(meta, weights)

    def _upload(self, meta: UpdateMetadata, weights: bytes) -> bool:
        """Send the update; True once the gateway has decided (accepted or finally rejected)."""
        try:
            ack = self.transport.upload_update(meta, weights)
        except GatewayHTTPError as exc:  # e.g. identity_mismatch: resending cannot help
            self.log(f"update {meta.update_id[:8]} refused: {exc}")
            ack = None
        if ack is not None and ack.accepted:
            note = " (retry acknowledged)" if ack.reason == "duplicate_idempotent" else ""
            self.log(f"round {meta.round_id}: update accepted{note}")
        elif ack is not None and ack.reason not in FINAL_REJECTIONS:
            self.log(f"round {meta.round_id}: upload not decided ({ack.reason}); will resend")
            return False
        elif ack is not None:
            self.log(f"round {meta.round_id}: update rejected ({ack.reason})")
        self._clear_pending(meta.update_id)
        self.state.rounds_done.append([meta.run_id, meta.round_id])
        self._save_state()
        return True

    def _transfer_metrics(self) -> dict[str, float]:
        stats = getattr(self.transport, "stats", None)
        metrics: dict[str, float] = {}
        if stats is not None:
            metrics["retry_count"] = float(sum(stats.retries.values()))
        ping = getattr(self.transport, "ping", None)
        if ping is not None:
            try:
                metrics["rtt_ms"] = ping()
            except (GatewayUnavailable, GatewayHTTPError):
                pass
        return metrics

    # -- final model --------------------------------------------------------------------------
    def _fetch_finished_runs(self) -> bool:
        for run_id in self.state.runs:
            if run_id in self.state.finals_verified:
                continue
            try:
                downloaded = self.transport.download_final_artifact(run_id)
            except GatewayHTTPError as exc:
                if exc.code == "no_final_model":  # the run failed; nothing to fetch
                    self.log(f"run {run_id[:8]} ended without a final model")
                    self.state.finals_verified.append(run_id)
                    self._save_state()
                continue
            if downloaded is None:  # still running
                continue
            files, onnx_sha = downloaded
            descriptor = verify_artifact(files, onnx_sha)
            final_dir = self.state_dir / "final" / run_id
            for name, data in files.items():
                _atomic_write(final_dir / name, data)
            self.state.finals_verified.append(run_id)
            self._save_state()
            self.log(
                f"run {run_id[:8]} finished: final model v{descriptor['model_version']} verified "
                f"(sha256 {onnx_sha[:12]}), saved to {final_dir}"
            )
            return True
        return False

    def _call_until_done(self, action: Callable[[], object]):
        while True:
            try:
                return action()
            except GatewayUnavailable as exc:
                self.log(f"gateway unreachable, will retry: {exc}")
                self.sleep(self.poll_interval_s)
