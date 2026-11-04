import json
from pathlib import Path

import torch

from cfl.conformance.checker import check_report
from cfl.conformance.report import build_report
from cfl.core.contract import initial_weights
from cfl.core.serialization import sha256
from cfl.tasks.text.model import contract
from cfl.tasks.text.spec import SmsTaskSpec
from cfl.tasks.text.tokenizer import MAX_LEN, PAD_ID

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "planning" / "examples" / "sms_spam_manifest.json"


def test_sms_spec_matches_manifest_conformance():
    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    spec = SmsTaskSpec()
    report = build_report(
        manifest=manifest,
        manifest_sha256=sha256(manifest_bytes),
        client_id="sms-test-client",
        client_version="test",
        preprocess=spec.preprocess,
        build_model=spec.build_model,
        contract=contract(),
    )

    result = check_report(manifest, sha256(manifest_bytes), report)
    assert result.passed, result
    assert spec.name == "text"


def test_spec_preprocess_handles_empty_and_long_messages():
    spec = SmsTaskSpec()

    empty = spec.preprocess({"text": ""})
    long_message = spec.preprocess({"text": " ".join(f"word{i}" for i in range(100))})

    assert len(empty) == MAX_LEN
    assert set(empty) == {PAD_ID}
    assert len(long_message) == MAX_LEN
    assert PAD_ID not in long_message


def test_training_is_deterministic_for_same_seed():
    spec = SmsTaskSpec()
    inputs = torch.tensor([[1, 2, 0], [3, 4, 0], [5, 6, 0], [7, 8, 0]])
    labels = torch.tensor([0, 1, 0, 1])
    config = {
        "local_epochs": 2,
        "batch_size": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "class_weights": None,
        "proximal_mu": 0.0,
    }
    start = initial_weights(spec.build_model, seed=7)
    first, second = spec.build_model(), spec.build_model()
    first.load_state_dict(start)
    second.load_state_dict(start)

    spec.train(first, (inputs, labels), config, None, seed=23)
    spec.train(second, (inputs, labels), config, None, seed=23)

    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])


def test_fedprox_changes_local_training_update():
    spec = SmsTaskSpec()
    inputs = torch.tensor([[1, 2, 0], [3, 4, 0], [5, 6, 0], [7, 8, 0]])
    labels = torch.tensor([0, 1, 0, 1])
    config = {
        "local_epochs": 2,
        "batch_size": 2,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "class_weights": None,
        "proximal_mu": 1.0,
    }
    global_weights = initial_weights(spec.build_model, seed=11)
    fedprox_model = spec.build_model()
    fedprox_model.load_state_dict(global_weights)
    fedavg_config = {**config, "proximal_mu": 0.0}
    fedavg_model = spec.build_model()
    fedavg_model.load_state_dict(global_weights)

    spec.train(fedprox_model, (inputs, labels), config, global_weights, seed=31)
    spec.train(fedavg_model, (inputs, labels), fedavg_config, global_weights, seed=31)

    assert any(
        not torch.equal(fedprox_model.state_dict()[name], fedavg_model.state_dict()[name])
        for name in global_weights
    )
