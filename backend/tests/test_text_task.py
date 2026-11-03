import numpy as np
import torch

from cfl.core.contract import initial_weights
from cfl.tasks.text.model import SmsHashEmbed
from cfl.tasks.text.tokenizer import (
    MAX_LEN,
    NUM_BUCKETS,
    PAD_ID,
    encode,
    encode_batch,
    feature_id,
    features,
    fnv1a_32,
)


def test_fnv1a_32_known_vectors():
    # Published FNV-1a 32-bit test vectors; independent implementations must agree.
    assert fnv1a_32(b"") == 0x811C9DC5
    assert fnv1a_32(b"a") == 0xE40C292C
    assert fnv1a_32(b"foobar") == 0xBF9CF968


def test_features_normalize_and_interleave_bigrams():
    assert features("FREE call, now!") == ["free", "call", "free_call", "now", "call_now"]
    # NFKC folds the full-width digit and letter before lower-casing.
    assert features("ＷＩＮ １") == ["win", "1", "win_1"]
    assert features("!!!") == []


def test_encode_pads_truncates_and_reserves_zero():
    ids = encode("hello world")
    assert len(ids) == MAX_LEN
    assert ids[:3] == [feature_id("hello"), feature_id("world"), feature_id("hello_world")]
    assert all(i == PAD_ID for i in ids[3:])
    long_ids = encode(" ".join(f"w{i}" for i in range(200)))
    assert len(long_ids) == MAX_LEN and PAD_ID not in long_ids
    assert all(1 <= feature_id(f"tok{i}") < NUM_BUCKETS for i in range(1000))


def test_encode_batch_shape_and_dtype():
    batch = encode_batch(["a b", "c"])
    assert batch.shape == (2, MAX_LEN) and batch.dtype == np.int64
    assert encode_batch([]).shape == (0, MAX_LEN)


def test_model_ignores_padding():
    model = SmsHashEmbed().eval()
    model.load_state_dict(initial_weights(SmsHashEmbed, seed=0))
    ids = torch.tensor([encode("win a free prize")])
    short = torch.tensor([encode("win a free prize", max_len=8)])
    empty = torch.tensor([encode("")])
    with torch.no_grad():
        assert torch.allclose(model(ids), model(short), atol=1e-6)
        assert model(empty).shape == (1, 2)
        assert torch.isfinite(model(empty)).all()
