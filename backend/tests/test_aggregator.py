import random

import pytest
import torch

from cfl.coordinator.aggregator import (
    Aggregator,
    FailureReason,
    Rejection,
    RoundInput,
    UpdateRecord,
    global_model_key,
)
from cfl.coordinator.state import RoundState
from cfl.core.contract import initial_weights
from cfl.core.serialization import bytes_to_state_dict, sha256, state_dict_to_bytes
from cfl.core.store import InMemoryModelStore
from cfl.core.strategy import FedAvg, FedProx
from cfl.tasks.tabular.model import INPUT_DIM, BankMLP, contract

RUN = "run-1"


def _perturbed(seed: int, scale: float = 0.01) -> dict[str, torch.Tensor]:
    base = initial_weights(BankMLP, seed=0)
    g = torch.Generator().manual_seed(seed)
    return {k: v + scale * torch.randn(v.shape, generator=g) for k, v in base.items()}


class Round:
    """Builds a round in an in-memory store: global model v0 plus client uploads."""

    def __init__(self):
        self.store = InMemoryModelStore()
        data = state_dict_to_bytes(initial_weights(BankMLP, seed=0))
        self.input_key = global_model_key(RUN, 0)
        self.store.put(self.input_key, data)
        self.input_sha = sha256(data)
        self.updates: list[UpdateRecord] = []

    def upload(self, client_id, weights=None, num_samples=100, version=0, data=None, sha=None):
        if data is None:
            if weights is None:
                weights = _perturbed(len(self.updates))
            data = state_dict_to_bytes(weights)
        key = f"{RUN}/updates/r1/{client_id}-{len(self.updates)}.safetensors"
        self.store.put(key, data)
        record = UpdateRecord(
            update_id=f"u-{client_id}-{len(self.updates)}",
            client_id=client_id,
            model_version=version,
            num_samples=num_samples,
            key=key,
            checksum_sha256=sha or sha256(data),
        )
        self.updates.append(record)
        return record

    def input(self, min_participants=2, max_num_samples=None, updates=None):
        return RoundInput(
            run_id=RUN,
            round_id=1,
            input_version=0,
            input_key=self.input_key,
            input_sha256=self.input_sha,
            updates=self.updates if updates is None else updates,
            min_participants=min_participants,
            max_num_samples=max_num_samples,
        )


def _aggregator(store, **kwargs):
    defaults = {
        "strategy": FedAvg(),
        "contract": contract(),
        "store": store,
        "build_model": BankMLP,
        "sanity_inputs": torch.zeros(2, INPUT_DIM),
    }
    return Aggregator(**(defaults | kwargs))


def test_fedavg_round_writes_weighted_average():
    r = Round()
    wa, wb = _perturbed(1), _perturbed(2)
    r.upload("a", wa, num_samples=100)
    r.upload("b", wb, num_samples=300)
    result = _aggregator(r.store).aggregate(r.input())

    assert result.state is RoundState.DONE
    assert result.output_version == 1
    assert result.output_key == global_model_key(RUN, 1)
    data = r.store.get(result.output_key)
    assert sha256(data) == result.output_sha256
    out = bytes_to_state_dict(data)
    for k in wa:
        assert torch.allclose(out[k], 0.25 * wa[k] + 0.75 * wb[k], atol=1e-6)
    assert result.used == ["a", "b"]
    assert result.rejected == {}
    m = result.metrics
    assert (m["updates_received"], m["updates_used"], m["total_samples"]) == (2, 2, 400)
    assert m["model_bytes"] == len(data) and m["bytes_read"] > 0


def test_fedprox_aggregates_like_fedavg():
    r = Round()
    r.upload("a")
    r.upload("b", num_samples=50)
    avg = _aggregator(r.store).aggregate(r.input())
    prox = _aggregator(r.store, strategy=FedProx(mu=0.1)).aggregate(r.input())
    assert avg.output_sha256 == prox.output_sha256


def test_result_is_independent_of_arrival_order_and_rerun_is_identical():
    r = Round()
    for i in range(5):
        r.upload(f"c{i}", num_samples=10 * (i + 1))
    first = _aggregator(r.store).aggregate(r.input())
    shuffled = list(r.updates)
    random.Random(3).shuffle(shuffled)
    again = _aggregator(r.store).aggregate(r.input(updates=shuffled))
    assert first.output_sha256 == again.output_sha256
    assert r.store.get(first.output_key) == r.store.get(again.output_key)


def _bad_weights(kind):
    w = _perturbed(9)
    if kind == "shape":
        w["fc1.weight"] = w["fc1.weight"].T.contiguous()
    elif kind == "nan":
        w["fc2.bias"][0] = float("nan")
    elif kind == "huge":  # finite, right shape, but explodes the logits
        w["fc3.bias"] = torch.full_like(w["fc3.bias"], 1e6)
    return w


@pytest.mark.parametrize(
    "make_bad, reason",
    [
        (lambda r: r.upload("x", _bad_weights("shape")), Rejection.MODEL_SHAPE_MISMATCH),
        (lambda r: r.upload("x", _bad_weights("nan")), Rejection.NON_FINITE_WEIGHTS),
        (lambda r: r.upload("x", _bad_weights("huge")), Rejection.SANITY_CHECK_FAILED),
        (lambda r: r.upload("x", version=3), Rejection.MODEL_VERSION_MISMATCH),
        (lambda r: r.upload("x", num_samples=0), Rejection.INVALID_NUM_SAMPLES),
        (lambda r: r.upload("x", num_samples=10**9), Rejection.INVALID_NUM_SAMPLES),
        (lambda r: r.upload("x", sha="0" * 64), Rejection.CHECKSUM_MISMATCH),
        (lambda r: r.upload("x", data=b"not safetensors"), Rejection.CORRUPT_FILE),
        (lambda r: r.upload("a"), Rejection.DUPLICATE_CLIENT),
    ],
)
def test_invalid_update_is_rejected_and_left_out(make_bad, reason):
    r = Round()
    good_a, good_b = r.upload("a"), r.upload("b")
    bad = make_bad(r)
    result = _aggregator(r.store).aggregate(r.input(max_num_samples=10_000))

    assert result.state is RoundState.DONE
    assert result.rejected == {bad.update_id: reason}
    assert result.used == ["a", "b"]
    assert good_a.update_id not in result.rejected and good_b.update_id not in result.rejected


def test_missing_update_file():
    r = Round()
    r.upload("a")
    r.upload("b")
    ghost = UpdateRecord("u-ghost", "c", 0, 10, f"{RUN}/updates/none.safetensors", "0" * 64)
    result = _aggregator(r.store).aggregate(r.input(updates=r.updates + [ghost]))
    assert result.rejected == {"u-ghost": Rejection.MISSING_FILE}


def test_quorum_is_rechecked_after_rejections():
    r = Round()
    r.upload("a")
    r.upload("b", _bad_weights("nan"))
    r.upload("c", version=7)
    result = _aggregator(r.store).aggregate(r.input(min_participants=2))

    assert result.state is RoundState.FAILED
    assert result.reason is FailureReason.QUORUM_NOT_MET
    assert result.output_key is None
    assert not r.store.exists(global_model_key(RUN, 1))  # global model stays at v0
    assert len(result.rejected) == 2


def test_corrupt_input_model_fails_round():
    r = Round()
    r.upload("a")
    r.upload("b")
    r.store.put(r.input_key, b"tampered")
    result = _aggregator(r.store).aggregate(r.input())
    assert result.state is RoundState.FAILED
    assert result.reason is FailureReason.INPUT_MODEL_INVALID


def test_evaluator_metrics_are_prefixed():
    r = Round()
    r.upload("a")
    r.upload("b")
    seen = []

    def evaluator(weights):
        seen.append(weights)
        return {"accuracy": 0.9, "f1": 0.5}

    result = _aggregator(r.store, evaluator=evaluator).aggregate(r.input())
    assert result.metrics["global_accuracy"] == 0.9
    assert result.metrics["global_f1"] == 0.5
    assert "evaluation_s" in result.metrics
    assert len(seen) == 1


def test_evaluation_error_fails_round_without_writing():
    r = Round()
    r.upload("a")
    r.upload("b")

    def broken(weights):
        raise RuntimeError("test set unavailable")

    result = _aggregator(r.store, evaluator=broken).aggregate(r.input())
    assert result.state is RoundState.FAILED
    assert result.reason is FailureReason.EVALUATION_ERROR
    assert "test set unavailable" in result.detail
    assert not r.store.exists(global_model_key(RUN, 1))
