import json
from pathlib import Path

import pytest
import torch

from cfl.conformance.checker import check_report
from cfl.conformance.report import build_report
from cfl.core.contract import initial_weights
from cfl.core.serialization import sha256
from cfl.tasks.tabular.model import INPUT_DIM, contract
from cfl.tasks.tabular.preprocessing import output_feature_names
from cfl.tasks.tabular.spec import BankTaskSpec

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "planning" / "examples" / "bank_marketing_manifest.json"
MANIFEST = json.loads(MANIFEST_PATH.read_bytes())
RAW_NO_PRIOR_CONTACT, RAW_PRIOR_CONTACT = (
    t["synthetic_raw_input"] for t in MANIFEST["reference_tests"]
)

ROUND_CONFIG = {
    "local_epochs": 2,
    "batch_size": 16,
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "class_weights": [1.0, 7.9],
    "proximal_mu": 0.0,
}


def _toy_split(n: int = 64, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    inputs = torch.randn(n, INPUT_DIM, generator=gen)
    labels = (inputs[:, 0] + 0.5 * torch.randn(n, generator=gen) > 0.8).long()
    labels[:2] = torch.tensor([0, 1])  # both classes present
    return inputs, labels


def _fresh_model(spec: BankTaskSpec, weights):
    model = spec.build_model()
    model.load_state_dict(weights)
    return model


def test_spec_matches_manifest_conformance():
    spec = BankTaskSpec()
    report = build_report(
        manifest=MANIFEST,
        manifest_sha256=sha256(MANIFEST_PATH.read_bytes()),
        client_id="bank-test-client",
        client_version="test",
        preprocess=spec.preprocess,
        build_model=spec.build_model,
        contract=contract(),
    )

    result = check_report(MANIFEST, sha256(MANIFEST_PATH.read_bytes()), report)

    assert result.passed, result
    assert spec.name == "tabular"


def test_model_keeps_architecture_and_parameter_names():
    model = BankTaskSpec().build_model()

    assert [(n, tuple(p.shape)) for n, p in model.named_parameters()] == [
        ("fc1.weight", (64, 65)),
        ("fc1.bias", (64,)),
        ("fc2.weight", (32, 64)),
        ("fc2.bias", (32,)),
        ("fc3.weight", (2, 32)),
        ("fc3.bias", (2,)),
    ]


def test_same_seed_gives_same_weights_and_loss():
    spec = BankTaskSpec()
    data = _toy_split()
    start = initial_weights(spec.build_model, seed=7)
    first, second = _fresh_model(spec, start), _fresh_model(spec, start)

    stats_a = spec.train(first, data, ROUND_CONFIG, None, seed=23)
    stats_b = spec.train(second, data, ROUND_CONFIG, None, seed=23)

    assert "train_loss" in stats_a
    assert stats_a == stats_b
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


def test_different_seed_gives_different_weights():
    spec = BankTaskSpec()
    data = _toy_split()
    start = initial_weights(spec.build_model, seed=7)
    first, second = _fresh_model(spec, start), _fresh_model(spec, start)

    spec.train(first, data, ROUND_CONFIG, None, seed=1)
    spec.train(second, data, ROUND_CONFIG, None, seed=2)

    assert any(
        not torch.equal(value, second.state_dict()[name])
        for name, value in first.state_dict().items()
    )


def test_fedprox_changes_the_result_and_pulls_towards_global_weights():
    spec = BankTaskSpec()
    data = _toy_split()
    global_weights = initial_weights(spec.build_model, seed=11)
    fedavg_model = _fresh_model(spec, global_weights)
    fedprox_model = _fresh_model(spec, global_weights)

    spec.train(fedavg_model, data, ROUND_CONFIG, global_weights, seed=31)
    spec.train(fedprox_model, data, {**ROUND_CONFIG, "proximal_mu": 5.0}, global_weights, seed=31)

    def distance(model):
        return sum(
            torch.sum((p - global_weights[n]) ** 2) for n, p in model.named_parameters()
        ).item()

    assert any(
        not torch.equal(fedprox_model.state_dict()[n], fedavg_model.state_dict()[n])
        for n in global_weights
    )
    assert distance(fedprox_model) < distance(fedavg_model)


def test_fedprox_without_global_weights_is_rejected():
    spec = BankTaskSpec()
    with pytest.raises(ValueError, match="global weights"):
        spec.train(spec.build_model(), _toy_split(), {**ROUND_CONFIG, "proximal_mu": 0.1}, None, 0)


def test_evaluate_returns_all_metrics():
    spec = BankTaskSpec()
    model = _fresh_model(spec, initial_weights(spec.build_model, seed=3))

    metrics = spec.evaluate(model, _toy_split())

    assert set(metrics) == {"loss", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}


def test_pdays_sentinel_is_replaced_by_mean_and_flagged():
    spec = BankTaskSpec()
    names = output_feature_names(spec.config)
    pdays, flag = names.index("pdays"), names.index("pdays_is_sentinel")

    never_contacted = spec.preprocess(RAW_NO_PRIOR_CONTACT)  # pdays = 999
    contacted = spec.preprocess(RAW_PRIOR_CONTACT)  # pdays = 6

    assert RAW_NO_PRIOR_CONTACT["pdays"] == 999
    assert never_contacted[pdays] == pytest.approx(0.0)  # filled with the mean, then scaled
    assert never_contacted[flag] == 1.0
    assert contacted[flag] == 0.0
    assert contacted[pdays] == pytest.approx((6 - 6.01452) / 3.82364, abs=1e-6)


def test_unknown_category_value_is_rejected():
    spec = BankTaskSpec()

    with pytest.raises(ValueError, match="job"):
        spec.preprocess({**RAW_PRIOR_CONTACT, "job": "astronaut"})


def test_missing_feature_column_is_rejected():
    spec = BankTaskSpec()
    raw = {k: v for k, v in RAW_PRIOR_CONTACT.items() if k != "euribor3m"}

    with pytest.raises(ValueError, match="euribor3m"):
        spec.preprocess(raw)


def test_feature_order_follows_manifest_regardless_of_input_key_order():
    spec = BankTaskSpec()
    reordered = dict(reversed(list(RAW_PRIOR_CONTACT.items())))

    row = spec.preprocess(reordered)

    assert len(row) == INPUT_DIM
    assert row == spec.preprocess(RAW_PRIOR_CONTACT)
    assert (
        output_feature_names(spec.config)
        == MANIFEST["input_schema"]["model_input"]["feature_names"]
    )
    expected = MANIFEST["reference_tests"][1]["expected_model_input"]
    assert row == pytest.approx(expected, abs=1e-6)


def test_duration_column_is_ignored():
    spec = BankTaskSpec()

    with_duration = spec.preprocess({**RAW_PRIOR_CONTACT, "duration": 500})

    assert with_duration == spec.preprocess(RAW_PRIOR_CONTACT)
