from pathlib import Path

import pytest
import torch

from cfl.core.contract import ContractViolation, ModelContract, forward_reference, initial_weights
from cfl.tasks.tabular import model as tabular
from cfl.tasks.tabular.preprocessing import load_dataset_config, output_feature_names
from cfl.tasks.text import model as text

BANK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "datasets" / "bank_marketing.yaml"


def test_tabular_contract_layout():
    assert tabular.INPUT_DIM == len(output_feature_names(load_dataset_config(BANK_CONFIG)))
    assert tabular.contract().manifest_parameters() == [
        {"name": "fc1.weight", "shape": [64, 65], "dtype": "float32"},
        {"name": "fc1.bias", "shape": [64], "dtype": "float32"},
        {"name": "fc2.weight", "shape": [32, 64], "dtype": "float32"},
        {"name": "fc2.bias", "shape": [32], "dtype": "float32"},
        {"name": "fc3.weight", "shape": [2, 32], "dtype": "float32"},
        {"name": "fc3.bias", "shape": [2], "dtype": "float32"},
    ]


def test_text_contract_layout():
    assert [(p.name, p.shape) for p in text.contract().params] == [
        ("emb.weight", (16384, 64)),
        ("fc1.weight", (32, 64)),
        ("fc1.bias", (32,)),
        ("fc2.weight", (2, 32)),
        ("fc2.bias", (2,)),
    ]


def test_fingerprint_is_stable_and_version_sensitive():
    a = tabular.contract()
    assert a.fingerprint() == tabular.contract().fingerprint()
    bumped = ModelContract(a.architecture_id, "1.0.1", a.params)
    assert bumped.fingerprint() != a.fingerprint()
    assert a.fingerprint() != text.contract().fingerprint()


def test_buffers_and_non_float32_rejected():
    with pytest.raises(ContractViolation, match="buffers"):
        ModelContract.from_model(torch.nn.BatchNorm1d(4), "bn", "1.0.0")
    with pytest.raises(ContractViolation, match="float32"):
        ModelContract.from_model(torch.nn.Linear(2, 2).double(), "lin", "1.0.0")


def test_check_accepts_matching_weights():
    tabular.contract().check(initial_weights(tabular.BankMLP, seed=0))


@pytest.mark.parametrize(
    "mutate, code",
    [
        (lambda w: w.pop("fc3.bias"), "model_shape_mismatch"),
        (lambda w: w.update(extra=torch.zeros(1)), "model_shape_mismatch"),
        # transposed non-square layer: [64, 65] sent as [65, 64]
        (lambda w: w.update({"fc1.weight": w["fc1.weight"].T}), "model_shape_mismatch"),
        (lambda w: w.update({"fc1.bias": w["fc1.bias"].double()}), "model_shape_mismatch"),
        (lambda w: w["fc2.weight"].__setitem__((0, 0), float("nan")), "non_finite_weights"),
    ],
)
def test_check_rejects_bad_weights(mutate, code):
    weights = initial_weights(tabular.BankMLP, seed=0)
    mutate(weights)
    with pytest.raises(ContractViolation) as exc:
        tabular.contract().check(weights)
    assert exc.value.code == code


def test_initial_weights_deterministic_and_isolated():
    torch.manual_seed(123)
    expected_next = torch.rand(1)
    torch.manual_seed(123)
    a = initial_weights(text.SmsHashEmbed, seed=7)
    assert torch.equal(torch.rand(1), expected_next)  # global RNG untouched
    b = initial_weights(text.SmsHashEmbed, seed=7)
    assert all(torch.equal(a[k], b[k]) for k in a)
    c = initial_weights(text.SmsHashEmbed, seed=8)
    assert not torch.equal(a["fc1.weight"], c["fc1.weight"])


def test_forward_reference_detects_swapped_square_layer():
    # Shape checks cannot catch a transposed square matrix; the forward reference test can.
    model = torch.nn.Linear(3, 3)
    weights = initial_weights(lambda: torch.nn.Linear(3, 3), seed=0)
    x = torch.tensor([[1.0, 2.0, 3.0]])
    expected = forward_reference(model, weights, x)
    swapped = dict(weights, weight=weights["weight"].T.contiguous())
    assert forward_reference(model, swapped, x)[0] != pytest.approx(expected[0], abs=1e-5)
