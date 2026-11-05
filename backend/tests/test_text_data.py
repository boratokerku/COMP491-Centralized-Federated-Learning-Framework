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


def test_text_partition_files_are_byte_identical_for_the_same_seed(tmp_path):
    from cfl.tasks.partition import format_summary, summarize, write_partitions

    texts = [f'message {i}, with "quotes" and NA' for i in range(300)]
    labels = ["spam" if i % 7 == 0 else "ham" for i in range(300)]
    df = pd.DataFrame({"label": labels, "text": texts})
    outs = []
    for name in ("a", "b"):
        p = iid_partition(df, encode_labels, num_clients=3, seed=11)
        write_partitions(p, tmp_path / name, summarize(p, encode_labels, "sms_spam", seed=11))
        outs.append(tmp_path / name)
    files = sorted(f.relative_to(outs[0]) for f in outs[0].rglob("*") if f.is_file())
    assert len(files) == 11
    for f in files:
        assert (outs[0] / f).read_bytes() == (outs[1] / f).read_bytes(), f
    assert "sms_spam partition" in format_summary(summarize(p, encode_labels, "sms_spam"))
