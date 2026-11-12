import json
import re
import unicodedata
from pathlib import Path

import pytest
import torch

from cfl.conformance.checker import check_report
from cfl.conformance.report import build_report
from cfl.core.contract import ModelContract, initial_weights
from cfl.core.serialization import sha256
from cfl.tasks.text.model import EMBED_DIM, NUM_CLASSES, SmsHashEmbed, contract
from cfl.tasks.text.spec import SmsTaskSpec
from cfl.tasks.text.tokenizer import MAX_LEN, NUM_BUCKETS, PAD_ID, encode, features, fnv1a_32

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


def _wrong_python_hash_preprocess(raw: dict) -> list[int]:
    tokens = re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", raw["text"]).lower())
    ngrams: list[str] = []
    for index, token in enumerate(tokens):
        ngrams.append(token)
        if index:
            ngrams.append(f"{tokens[index - 1]}_{token}")
    ids = [1 + hash(ngram) % (NUM_BUCKETS - 1) for ngram in ngrams[:MAX_LEN]]
    return ids + [PAD_ID] * (MAX_LEN - len(ids))


def _case_sensitive_preprocess(raw: dict) -> list[int]:
    tokens = re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", raw["text"]))
    ngrams: list[str] = []
    for index, token in enumerate(tokens):
        ngrams.append(token)
        if index:
            ngrams.append(f"{tokens[index - 1]}_{token}")
    ids = [
        1 + fnv1a_32(ngram.encode("utf-8")) % (NUM_BUCKETS - 1)
        for ngram in ngrams[:MAX_LEN]
    ]
    return ids + [PAD_ID] * (MAX_LEN - len(ids))


def _wrong_bucket_preprocess(raw: dict) -> list[int]:
    ids = [
        1 + fnv1a_32(ngram.encode("utf-8")) % 8191
        for ngram in features(raw["text"])[:MAX_LEN]
    ]
    return ids + [PAD_ID] * (MAX_LEN - len(ids))


class RenamedParameterSmsModel(torch.nn.Module):
    """Same forward computation, but an incompatible state-dict parameter name."""

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(NUM_BUCKETS, EMBED_DIM, padding_idx=PAD_ID)
        self.fc1 = torch.nn.Linear(EMBED_DIM, 32)
        self.fc2 = torch.nn.Linear(32, NUM_CLASSES)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != PAD_ID).unsqueeze(-1).to(self.embedding.weight.dtype)
        summed = (self.embedding(ids) * mask).sum(dim=1)
        pooled = summed / mask.sum(dim=1).clamp(min=1.0)
        return self.fc2(torch.relu(self.fc1(pooled)))


@pytest.mark.parametrize(
    ("variant", "preprocess", "build_model", "model_contract", "expected_code"),
    [
        (
            "python_hash",
            _wrong_python_hash_preprocess,
            SmsHashEmbed,
            contract,
            "reference_output_mismatch",
        ),
        (
            "case_sensitive",
            _case_sensitive_preprocess,
            SmsHashEmbed,
            contract,
            "reference_output_mismatch",
        ),
        (
            "max_len_128",
            lambda raw: encode(raw["text"], max_len=128),
            SmsHashEmbed,
            contract,
            "invalid_report",
        ),
        (
            "num_buckets_8192",
            _wrong_bucket_preprocess,
            SmsHashEmbed,
            contract,
            "reference_output_mismatch",
        ),
        (
            "renamed_parameter",
            lambda raw: encode(raw["text"]),
            RenamedParameterSmsModel,
            lambda: ModelContract.from_model(
                RenamedParameterSmsModel(), "sms_hash_embed", "1.0.0"
            ),
            "model_shape_mismatch",
        ),
    ],
)
def test_sms_conformance_rejects_incompatible_client_variants(
    variant, preprocess, build_model, model_contract, expected_code
):
    """The checker rejects faulty implementations using the manifest's synthetic fixtures."""
    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    report = build_report(
        manifest=manifest,
        manifest_sha256=sha256(manifest_bytes),
        client_id=f"sms-{variant}-client",
        client_version="test",
        preprocess=preprocess,
        build_model=build_model,
        contract=model_contract(),
    )

    result = check_report(manifest, sha256(manifest_bytes), report)
    assert not result.passed, variant
    assert result.reason_code == expected_code, variant


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
