import copy
import json
from pathlib import Path

import jsonschema
import pytest
import torch

from cfl.core.contract import forward_reference
from cfl.core.manifest import reference_weights, same_manifest
from cfl.tasks.tabular import model as tabular_model
from cfl.tasks.tabular.manifest import build_manifest as build_tabular
from cfl.tasks.tabular.preprocessing import load_dataset_config
from cfl.tasks.text import model as text_model
from cfl.tasks.text.manifest import build_manifest as build_text

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((REPO_ROOT / "planning" / "conformance-manifest.schema.json").read_text())
EXAMPLES = REPO_ROOT / "planning" / "examples"
BANK_CONFIG = REPO_ROOT / "backend" / "configs" / "datasets" / "bank_marketing.yaml"


@pytest.fixture(scope="module")
def manifests() -> dict[str, dict]:
    return {
        "bank_marketing_manifest.json": build_tabular(load_dataset_config(BANK_CONFIG)),
        "sms_spam_manifest.json": build_text(),
    }


@pytest.mark.parametrize("name", ["bank_marketing_manifest.json", "sms_spam_manifest.json"])
def test_generated_manifest_is_valid_and_checked_in(manifests, name):
    jsonschema.validate(manifests[name], SCHEMA)
    checked_in = json.loads((EXAMPLES / name).read_text())
    assert same_manifest(checked_in, manifests[name]), (
        f"{name} is out of date; run python backend/scripts/build_manifests.py"
    )


def test_fingerprints_match_code(manifests):
    bank, sms = manifests["bank_marketing_manifest.json"], manifests["sms_spam_manifest.json"]
    assert bank["model"]["fingerprint"] == tabular_model.contract().fingerprint()
    assert sms["model"]["fingerprint"] == text_model.contract().fingerprint()


@pytest.mark.parametrize(
    "drop",
    [("model", "fingerprint"), ("forward_reference_tests",), ("validation", "forward_atol")],
)
def test_schema_requires_new_fields(manifests, drop):
    manifest = copy.deepcopy(manifests["bank_marketing_manifest.json"])
    *parents, key = drop
    target = manifest
    for p in parents:
        target = target[p]
    del target[key]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(manifest, SCHEMA)


def test_reference_weights_follow_running_index():
    contract = tabular_model.contract()
    weights = reference_weights(contract)
    contract.check(weights)
    first = weights["fc1.weight"]
    assert first[0, 1].item() == pytest.approx(0.1 * torch.sin(torch.tensor(2.0)).item())
    # fc1.bias continues after the 64 * 65 elements of fc1.weight
    assert weights["fc1.bias"][0].item() == pytest.approx(
        0.1 * torch.sin(torch.tensor(64.0 * 65 + 1, dtype=torch.float64)).item()
    )


@pytest.mark.parametrize(
    "name, build, contract, dtype",
    [
        (
            "bank_marketing_manifest.json",
            tabular_model.BankMLP,
            tabular_model.contract,
            torch.float32,
        ),
        ("sms_spam_manifest.json", text_model.SmsHashEmbed, text_model.contract, torch.int64),
    ],
)
def test_forward_tests_reproducible_and_detect_transpose(manifests, name, build, contract, dtype):
    manifest = manifests[name]
    weights = reference_weights(contract())
    atol = manifest["validation"]["forward_atol"]
    first_layer = "fc1.weight"
    for test in manifest["forward_reference_tests"]:
        x = torch.tensor([test["model_input"]], dtype=dtype)
        logits = forward_reference(build(), weights, x)[0]
        assert logits == pytest.approx(test["expected_logits"], abs=atol)

    # An implementation that reads fc1.weight in the wrong order (column-major) must fail.
    out_dim, in_dim = weights[first_layer].shape
    wrong = dict(weights)
    wrong[first_layer] = weights[first_layer].reshape(in_dim, out_dim).T.contiguous()
    test = manifest["forward_reference_tests"][0]
    logits = forward_reference(build(), wrong, torch.tensor([test["model_input"]], dtype=dtype))[0]
    assert logits != pytest.approx(test["expected_logits"], abs=atol)


def test_same_manifest_tolerance():
    assert same_manifest({"a": [1, 0.5]}, {"a": [1.0, 0.5 + 1e-9]})
    assert not same_manifest({"a": [0.5]}, {"a": [0.51]})
    assert not same_manifest({"a": "x"}, {"a": "y"})
    assert not same_manifest({"a": 1}, {"a": 1, "b": 2})
