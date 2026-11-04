import pandas as pd
import pytest

from cfl.tasks.partition import iid_partition
from cfl.tasks.text.data import deduplicate, encode_labels, read_raw


def test_read_raw_keeps_quotes_and_tabs_layout(tmp_path):
    raw = tmp_path / "SMSSpamCollection"
    raw.write_text('ham\tHe said "ok", then left\nspam\tWIN NA prize\n', encoding="utf-8")
    df = read_raw(raw)
    assert list(df.columns) == ["label", "text"]
    assert df["text"].tolist() == ['He said "ok", then left', "WIN NA prize"]


def test_deduplicate_keeps_first_and_drops_conflicts():
    df = pd.DataFrame(
        {
            "label": ["spam", "ham", "spam", "ham", "spam"],
            "text": ["free prize", "hi", "free prize", "both", "both"],
        }
    )
    out = deduplicate(df)
    assert out.to_dict("records") == [
        {"label": "spam", "text": "free prize"},
        {"label": "ham", "text": "hi"},
    ]


def test_encode_labels():
    assert encode_labels(pd.DataFrame({"label": ["ham", "spam"]})).tolist() == [0, 1]
    with pytest.raises(ValueError):
        encode_labels(pd.DataFrame({"label": ["Spam"]}))


def test_partition_after_dedupe_has_no_shared_text():
    texts = [f"message {i}" for i in range(400)]
    labels = ["spam" if i % 8 == 0 else "ham" for i in range(400)]
    df = deduplicate(pd.DataFrame({"label": labels * 2, "text": texts * 2}))  # every text twice
    p = iid_partition(df, encode_labels, num_clients=4, seed=0)
    parts = [p.test] + [getattr(c, s) for c in p.clients for s in ("train", "val", "test")]
    seen = [t for part in parts for t in part["text"]]
    assert len(seen) == len(set(seen)) == 400
