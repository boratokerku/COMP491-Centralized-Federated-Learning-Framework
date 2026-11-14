"""The independent client (independent-client/) against the gateway's own checks.

The client never imports cfl; these tests do, to confirm that what it produces is accepted or
rejected exactly as the specification says.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("requests")

from cfl.client.reference import ReferenceTaskSpec  # noqa: E402
from cfl.conformance.checker import check_report  # noqa: E402
from cfl.core.contract import initial_weights  # noqa: E402
from cfl.core.serialization import bytes_to_state_dict, state_dict_to_bytes  # noqa: E402
from cfl.protocol.messages import ConformanceReportRequest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "planning" / "examples" / "bank_marketing_manifest.json"


def _load_client():
    path = ROOT / "independent-client" / "independent_client.py"
    spec = importlib.util.spec_from_file_location("independent_client", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["independent_client"] = module
    spec.loader.exec_module(module)
    return module


ic = _load_client()
MANIFEST_BYTES = MANIFEST_PATH.read_bytes()
MANIFEST = json.loads(MANIFEST_BYTES)
MANIFEST_SHA = ic.sha256_hex(MANIFEST_BYTES)


def _decision(variant: str):
    report = ic.build_report(MANIFEST, MANIFEST_SHA, "independent-test", variant)
    return check_report(MANIFEST, MANIFEST_SHA, ConformanceReportRequest.model_validate(report))


def test_compatible_variant_passes_the_gateway_check():
    assert _decision("compatible").passed


@pytest.mark.parametrize(
    ("variant", "code"), [(v, c) for v, c in ic.VARIANTS.items() if c is not None]
)
def test_each_incompatible_variant_is_rejected_with_its_code(variant, code):
    result = _decision(variant)

    assert not result.passed
    assert result.reason_code == code


def test_every_documented_rejection_code_is_covered():
    assert set(filter(None, ic.VARIANTS.values())) == {
        "feature_order_mismatch",
        "unit_mismatch",
        "label_semantics_mismatch",
        "label_range_mismatch",
        "preprocessing_version_mismatch",
        "model_shape_mismatch",
        "encoding_mismatch",
        "forward_output_mismatch",
    }


def test_own_verdict_flags_the_failures_it_can_see_itself():
    report = ic.build_report(MANIFEST, MANIFEST_SHA, "independent-test", "weights-layout")

    assert report["client_passed"] is False
    assert report["client_failure_codes"] == ["forward_output_mismatch"]
    assert ic.build_report(MANIFEST, MANIFEST_SHA, "x")["client_passed"] is True


def test_safetensors_files_are_read_and_written_across_implementations():
    weights = initial_weights(ReferenceTaskSpec("tabular").build_model, seed=5)

    from_cfl = ic.read_safetensors(state_dict_to_bytes(weights))
    back = bytes_to_state_dict(ic.write_safetensors(from_cfl))

    for name, tensor in weights.items():
        assert np.array_equal(from_cfl[name], tensor.numpy().astype(np.float64))
        assert torch.equal(back[name], tensor)


def test_reference_weights_and_forward_pass_match_the_manifest():
    layout = ic.parameter_layout(65, "compatible")
    weights = ic.reference_sin_v1(layout, "compatible")

    for test in MANIFEST["forward_reference_tests"]:
        logits = ic.forward(weights, np.array([test["model_input"]]))[0]
        assert np.abs(logits - np.array(test["expected_logits"])).max() < 1e-5


@pytest.mark.parametrize(("mu", "weight_decay"), [(0.0, 0.0), (0.1, 0.01)])
def test_numpy_training_matches_pytorch_reference(mu, weight_decay, monkeypatch):
    monkeypatch.setattr(ic, "DROPOUT", 0.0)  # dropout masks differ between the two RNGs
    spec = ReferenceTaskSpec("tabular")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(50, 65)).astype(np.float32)
    y = (rng.random(50) < 0.3).astype(np.int64)
    start = initial_weights(spec.build_model, seed=3)
    config = {
        "local_epochs": 5,
        "batch_size": 50,
        "learning_rate": 0.01,
        "weight_decay": weight_decay,
        "class_weights": [1.0, 5.0],
        "proximal_mu": mu,
    }
    model = spec.build_model()
    model.load_state_dict(start)
    model.dropout.p = 0.0
    spec.train(model, (torch.from_numpy(x), torch.from_numpy(y)), config, start, seed=1)

    weights, loss = ic.train(
        {k: v.double().numpy() for k, v in start.items()}, x.astype(np.float64), y, config, seed=1
    )

    assert np.isfinite(loss)
    for name, tensor in model.state_dict().items():
        assert np.abs(weights[name] - tensor.double().numpy()).max() < 1e-4


def test_roc_auc_and_average_precision_match_scikit_learn():
    from sklearn.metrics import average_precision_score, roc_auc_score

    rng = np.random.default_rng(2)
    y = (rng.random(400) < 0.15).astype(int)
    score = np.round(rng.random(400), 1)  # with ties

    assert ic.roc_auc(y, score) == pytest.approx(roc_auc_score(y, score))
    assert ic.average_precision(y, score) == pytest.approx(average_precision_score(y, score))


def test_preprocessing_matches_the_reference_tests_and_rejects_unknown_categories():
    import pandas as pd

    features = ic.schema_features(MANIFEST, "compatible")
    for test in MANIFEST["reference_tests"]:
        row = ic.preprocess(pd.DataFrame([test["synthetic_raw_input"]]), features)[0]
        assert np.abs(row - np.array(test["expected_model_input"])).max() < 1e-6

    bad = {**MANIFEST["reference_tests"][1]["synthetic_raw_input"], "job": "astronaut"}
    with pytest.raises(ic.ClientError, match="job"):
        ic.preprocess(pd.DataFrame([bad]), features)
