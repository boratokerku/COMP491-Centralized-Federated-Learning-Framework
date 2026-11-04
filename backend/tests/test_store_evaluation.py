from pathlib import Path

import pandas as pd
import pytest
import torch

from cfl.core.contract import initial_weights
from cfl.core.evaluation import HeldOutEvaluator, classification_metrics
from cfl.core.store import FileModelStore, InMemoryModelStore
from cfl.tasks.tabular.data import load_split as load_tabular
from cfl.tasks.tabular.preprocessing import load_dataset_config
from cfl.tasks.text.data import load_split as load_text
from cfl.tasks.text.model import SmsHashEmbed
from cfl.tasks.text.tokenizer import MAX_LEN

BANK_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "datasets" / "bank_marketing.yaml"


@pytest.mark.parametrize("make", [InMemoryModelStore, None])
def test_store_roundtrip_and_bad_keys(tmp_path, make):
    store = make() if make else FileModelStore(tmp_path)
    store.put("run/global_model_v0.safetensors", b"v0")
    store.put("run/global_model_v0.safetensors", b"v0-replaced")
    assert store.get("run/global_model_v0.safetensors") == b"v0-replaced"
    assert store.exists("run/global_model_v0.safetensors")
    assert not store.exists("run/other")
    with pytest.raises(KeyError):
        store.get("run/other")
    for key in ["", "/etc/passwd", "../outside", "a/../../b", "a\\b"]:
        with pytest.raises(ValueError):
            store.put(key, b"x")


def test_file_store_leaves_no_temp_files(tmp_path):
    store = FileModelStore(tmp_path)
    store.put("r/m.safetensors", b"data")
    assert [p.name for p in (tmp_path / "r").iterdir()] == ["m.safetensors"]


def test_classification_metrics():
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 1.0], [1.0, 0.0]])
    labels = torch.tensor([0, 1, 0, 1])
    m = classification_metrics(logits, labels)
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert 0 <= m["roc_auc"] <= 1 and 0 <= m["pr_auc"] <= 1
    one_class = classification_metrics(logits, torch.zeros(4, dtype=torch.long))
    assert "roc_auc" not in one_class


def test_held_out_evaluator_on_text_split(tmp_path):
    csv = tmp_path / "test.csv"
    pd.DataFrame(
        {"label": ["spam", "ham", "ham"], "text": ['WIN "free", now', "NA", "see you"]}
    ).to_csv(csv, index=False)
    ids, labels = load_text(csv)
    assert ids.shape == (3, MAX_LEN) and labels.tolist() == [1, 0, 0]
    assert (ids[1] != 0).any()  # "NA" is text, not a missing value

    evaluator = HeldOutEvaluator(SmsHashEmbed, ids, labels, batch_size=2)
    m = evaluator(initial_weights(SmsHashEmbed, seed=0))
    assert {"loss", "accuracy", "f1", "roc_auc", "pr_auc"} <= m.keys()


def test_load_tabular_split(tmp_path):
    config = load_dataset_config(BANK_CONFIG)
    row = {f.name: (f.mean if f.type == "numeric" else f.categories[0]) for f in config.features}
    csv = tmp_path / "val.csv"
    pd.DataFrame([row | {"y": "yes"}, row | {"y": "no"}]).to_csv(csv, sep=";", index=False)
    x, y = load_tabular(csv, config)
    assert x.shape == (2, 65) and x.dtype == torch.float32
    assert y.tolist() == [1, 0]
