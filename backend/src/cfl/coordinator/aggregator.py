"""Round aggregation: the AGGREGATING -> DONE | FAILED step of a round.

The coordinator moves a round to AGGREGATING and passes the round's accepted update records
(rows of the `updates` table). The aggregator loads and validates each update, re-checks the
quorum after rejections, aggregates with the run's strategy, evaluates the result on the
server's held-out set and writes the next global model to the store. It never touches the
database: the coordinator persists the returned result (state, output model key/checksum,
rejections, metrics).

Re-running the same round (e.g. after a coordinator crash during AGGREGATING) is safe: the
result depends only on the inputs, updates are combined in a fixed order, and the output file is
replaced atomically, so a rerun writes byte-identical weights.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from cfl.coordinator.state import RoundState
from cfl.core.contract import ContractViolation, ModelContract, StateDict
from cfl.core.privacy import DEFAULT_MODULUS, DEFAULT_SCALE, dequantize_state_dict
from cfl.core.secagg import round_context, sum_masked, unmask_aggregate
from cfl.core.serialization import bytes_to_state_dict, sha256, state_dict_to_bytes
from cfl.core.store import ModelStore
from cfl.core.strategy import ClientUpdate, Strategy, delta_sum

Evaluator = Callable[[StateDict], dict[str, float]]


class Rejection(StrEnum):
    MISSING_FILE = "missing_file"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    CORRUPT_FILE = "corrupt_file"
    MODEL_VERSION_MISMATCH = "model_version_mismatch"
    INVALID_NUM_SAMPLES = "invalid_num_samples"
    DUPLICATE_CLIENT = "duplicate_client"
    MODEL_SHAPE_MISMATCH = "model_shape_mismatch"
    NON_FINITE_WEIGHTS = "non_finite_weights"
    SANITY_CHECK_FAILED = "sanity_check_failed"


class FailureReason(StrEnum):
    INPUT_MODEL_INVALID = "input_model_invalid"
    QUORUM_NOT_MET = "quorum_not_met"
    AGGREGATION_ERROR = "aggregation_error"
    EVALUATION_ERROR = "evaluation_error"


@dataclass(frozen=True)
class UpdateRecord:
    """What the coordinator reads from an accepted `updates` row."""

    update_id: str
    client_id: str
    model_version: int
    num_samples: int
    key: str  # store key of the uploaded safetensors file
    checksum_sha256: str


@dataclass(frozen=True)
class RoundInput:
    run_id: str
    round_id: int
    input_version: int
    input_key: str
    input_sha256: str
    updates: list[UpdateRecord]
    min_participants: int
    max_num_samples: int | None = None  # per-client cap on the reported sample count
    # Privacy extension (SecAgg+ and/or DP delta aggregation)
    privacy_delta: bool = False
    secagg_enabled: bool = False
    secagg_self_seeds: dict[str, bytes] | None = None
    secagg_dropped_keys: dict[str, X25519PrivateKey] | None = None
    secagg_surviving_public_keys: dict[str, bytes] | None = None
    secagg_weight_scale: float = 1.0  # re-normalizes surviving α_i sum back to 1.0 after dropouts
    secagg_modulus: int = DEFAULT_MODULUS
    secagg_scale: float = DEFAULT_SCALE
    cleanup_updates: bool = False


@dataclass
class AggregationResult:
    state: RoundState  # DONE or FAILED
    reason: FailureReason | None = None
    detail: str | None = None
    output_version: int | None = None
    output_key: str | None = None
    output_sha256: str | None = None
    used: list[str] = field(default_factory=list)  # client_ids that went into the average
    rejected: dict[str, Rejection] = field(default_factory=dict)  # update_id -> reason
    metrics: dict[str, float] = field(default_factory=dict)


def global_model_key(run_id: str, version: int) -> str:
    return f"{run_id}/global_model_v{version}.safetensors"


class Aggregator:
    def __init__(
        self,
        strategy: Strategy,
        contract: ModelContract,
        store: ModelStore,
        build_model: Callable[[], torch.nn.Module],
        evaluator: Evaluator | None = None,
        sanity_inputs: torch.Tensor | None = None,
        max_abs_logit: float = 1e4,
    ):
        """`sanity_inputs` are synthetic model inputs (e.g. the manifest's forward reference
        inputs); every update must produce finite logits below `max_abs_logit` on them."""
        self.strategy = strategy
        self.contract = contract
        self.store = store
        self.build_model = build_model
        self.evaluator = evaluator
        self.sanity_inputs = sanity_inputs
        self.max_abs_logit = max_abs_logit

    def aggregate(self, round_input: RoundInput) -> AggregationResult:
        r = round_input
        metrics: dict[str, float] = {"updates_received": float(len(r.updates))}

        try:
            try:
                global_weights = self._load_verified(r.input_key, r.input_sha256)
                self.contract.check(global_weights)
            except Exception as exc:  # missing, corrupt or non-conforming input model
                return AggregationResult(
                    RoundState.FAILED,
                    FailureReason.INPUT_MODEL_INVALID,
                    f"{type(exc).__name__}: {exc}",
                    metrics=metrics,
                )

            start = time.perf_counter()
            accepted, rejected, bytes_read = self._load_updates(r, global_weights)
            metrics["load_s"] = time.perf_counter() - start
            metrics["bytes_read"] = float(bytes_read)
            metrics["updates_used"] = float(len(accepted))
            metrics["updates_rejected"] = float(len(rejected))
            used = [u.client_id for u in accepted]

            if len(accepted) < r.min_participants:
                return AggregationResult(
                    RoundState.FAILED,
                    FailureReason.QUORUM_NOT_MET,
                    f"{len(accepted)} valid updates, need {r.min_participants}",
                    used=used,
                    rejected=rejected,
                    metrics=metrics,
                )

            if r.secagg_enabled and rejected:
                return AggregationResult(
                    RoundState.FAILED,
                    FailureReason.AGGREGATION_ERROR,
                    "cannot unmask SecAgg aggregate: one or more masked updates failed validation",
                    used=used,
                    rejected=rejected,
                    metrics=metrics,
                )

            metrics["total_samples"] = float(sum(u.num_samples for u in accepted))
            start = time.perf_counter()
            try:
                new_weights = self._combine(r, global_weights, accepted)
                self.contract.check(new_weights)
                if r.secagg_enabled and not self._sane(new_weights):
                    raise ValueError("unmasked global model failed sanity check")
            except ValueError as exc:
                return AggregationResult(
                    RoundState.FAILED,
                    FailureReason.AGGREGATION_ERROR,
                    str(exc),
                    used=used,
                    rejected=rejected,
                    metrics=metrics,
                )
            metrics["aggregation_s"] = time.perf_counter() - start

            if self.evaluator is not None:
                start = time.perf_counter()
                try:
                    evaluation = self.evaluator(new_weights)
                except Exception as exc:  # any failure here must fail the round, not crash it
                    return AggregationResult(
                        RoundState.FAILED,
                        FailureReason.EVALUATION_ERROR,
                        f"{type(exc).__name__}: {exc}",
                        used=used,
                        rejected=rejected,
                        metrics=metrics,
                    )
                metrics["evaluation_s"] = time.perf_counter() - start
                metrics.update({f"global_{k}": v for k, v in evaluation.items()})

            start = time.perf_counter()
            data = state_dict_to_bytes(new_weights)
            output_version = r.input_version + 1
            output_key = global_model_key(r.run_id, output_version)
            self.store.put(output_key, data)
            metrics["write_s"] = time.perf_counter() - start
            metrics["model_bytes"] = float(len(data))

            return AggregationResult(
                RoundState.DONE,
                output_version=output_version,
                output_key=output_key,
                output_sha256=sha256(data),
                used=used,
                rejected=rejected,
                metrics=metrics,
            )
        finally:
            if r.cleanup_updates:
                for u in r.updates:
                    self.store.delete(u.key)

    def _combine(
        self, r: RoundInput, global_weights: StateDict, accepted: list[ClientUpdate]
    ) -> StateDict:
        if r.secagg_enabled:
            ctx = round_context(r.run_id, r.round_id)
            agg_masked = sum_masked([u.weights for u in accepted], modulus=r.secagg_modulus)
            unmasked = unmask_aggregate(
                agg_masked,
                ctx,
                surviving_self_seeds=r.secagg_self_seeds or {},
                dropped_private_keys=r.secagg_dropped_keys,
                surviving_public_keys=r.secagg_surviving_public_keys,
                modulus=r.secagg_modulus,
            )
            delta_agg = dequantize_state_dict(
                unmasked, scale=r.secagg_scale, modulus=r.secagg_modulus
            )
            if r.secagg_weight_scale != 1.0:
                delta_agg = {k: v * r.secagg_weight_scale for k, v in delta_agg.items()}
            return delta_sum(global_weights, [delta_agg])
        if r.privacy_delta:
            deltas = [u.weights for u in accepted]
            if r.secagg_weight_scale != 1.0:
                deltas = [{k: v * r.secagg_weight_scale for k, v in d.items()} for d in deltas]
            return delta_sum(global_weights, deltas)
        return self.strategy.aggregate(global_weights, accepted)

    def _load_verified(self, key: str, expected_sha256: str) -> StateDict:
        data = self.store.get(key)
        if sha256(data) != expected_sha256:
            raise ValueError(f"checksum mismatch for {key}")
        return bytes_to_state_dict(data)

    def _load_updates(
        self, r: RoundInput, global_weights: StateDict
    ) -> tuple[list[ClientUpdate], dict[str, Rejection], int]:
        accepted: list[ClientUpdate] = []
        rejected: dict[str, Rejection] = {}
        seen_clients: set[str] = set()
        bytes_read = 0

        # Fixed order, so the floating-point sum (and the output file) never depends on the
        # order in which updates arrived.
        for u in sorted(r.updates, key=lambda u: (u.client_id, u.update_id)):
            # The database allows one accepted update per client and round; if more arrive here
            # anyway, only the first in the fixed order is considered.
            if u.client_id in seen_clients:
                rejected[u.update_id] = Rejection.DUPLICATE_CLIENT
                continue
            seen_clients.add(u.client_id)
            reason = self._precheck(u, r)
            if reason is not None:
                rejected[u.update_id] = reason
                continue
            try:
                data = self.store.get(u.key)
            except KeyError:
                rejected[u.update_id] = Rejection.MISSING_FILE
                continue
            bytes_read += len(data)
            if sha256(data) != u.checksum_sha256:
                rejected[u.update_id] = Rejection.CHECKSUM_MISMATCH
                continue
            try:
                weights = bytes_to_state_dict(data)
            except Exception:  # safetensors raises its own error types on malformed input
                rejected[u.update_id] = Rejection.CORRUPT_FILE
                continue
            try:
                if r.secagg_enabled:
                    self.contract.check_quantized(weights, r.secagg_modulus)
                else:
                    self.contract.check(weights)
            except ContractViolation as exc:
                rejected[u.update_id] = Rejection(exc.code)
                continue
            if not r.secagg_enabled:
                check_weights = delta_sum(global_weights, [weights]) if r.privacy_delta else weights
                if not self._sane(check_weights):
                    rejected[u.update_id] = Rejection.SANITY_CHECK_FAILED
                    continue
            accepted.append(ClientUpdate(u.client_id, weights, u.num_samples))
        return accepted, rejected, bytes_read

    @staticmethod
    def _precheck(u: UpdateRecord, r: RoundInput) -> Rejection | None:
        if u.model_version != r.input_version:
            return Rejection.MODEL_VERSION_MISMATCH
        if u.num_samples <= 0 or (
            r.max_num_samples is not None and u.num_samples > r.max_num_samples
        ):
            return Rejection.INVALID_NUM_SAMPLES
        return None

    def _sane(self, weights: StateDict) -> bool:
        """Run the update through the reference model: a client whose weights pass the shape
        checks but are numerically broken must not reach the average."""
        if self.sanity_inputs is None:
            return True
        model = self.build_model()
        model.load_state_dict(weights)
        model.eval()
        with torch.no_grad():
            logits = model(self.sanity_inputs)
        return bool(torch.isfinite(logits).all() and logits.abs().max() <= self.max_abs_logit)
