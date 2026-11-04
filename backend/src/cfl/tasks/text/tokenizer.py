"""Hashing tokenizer for the text task. Runs on the client side.

Federated text models need every client to map a token to the same embedding row, but no
vocabulary can be built centrally because the server never sees client data. Tokens are
therefore hashed into a fixed number of buckets. The procedure below is the contract that
independent implementations must reproduce exactly (it is simple enough for any language):

1. Unicode NFKC normalization, then lower-case.
2. Tokens are the matches of [a-z0-9]+ in order.
3. Features interleave unigrams and bigrams: t0, t1, t0_t1, t2, t1_t2, ...
4. id = 1 + FNV-1a-32(utf8(feature)) mod (NUM_BUCKETS - 1); id 0 is padding.
5. Keep the first MAX_LEN ids and right-pad with 0.

Never use Python's built-in hash(): it is randomized per process.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np

TOKENIZER_VERSION = "1.0.0"
NUM_BUCKETS = 16384
MAX_LEN = 64
PAD_ID = 0

TOKEN_PATTERN = r"[a-z0-9]+"

_TOKEN_RE = re.compile(TOKEN_PATTERN)
_FNV_OFFSET = 0x811C9DC5
_FNV_PRIME = 0x01000193


def fnv1a_32(data: bytes) -> int:
    h = _FNV_OFFSET
    for byte in data:
        h ^= byte
        h = (h * _FNV_PRIME) & 0xFFFFFFFF
    return h


def features(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(unicodedata.normalize("NFKC", text).lower())
    out: list[str] = []
    for i, token in enumerate(tokens):
        out.append(token)
        if i > 0:
            out.append(f"{tokens[i - 1]}_{token}")
    return out


def feature_id(feature: str) -> int:
    return 1 + fnv1a_32(feature.encode("utf-8")) % (NUM_BUCKETS - 1)


def encode(text: str, max_len: int = MAX_LEN) -> list[int]:
    ids = [feature_id(f) for f in features(text)[:max_len]]
    return ids + [PAD_ID] * (max_len - len(ids))


def encode_batch(texts: list[str], max_len: int = MAX_LEN) -> np.ndarray:
    """int64 array of shape [len(texts), max_len]: the model input."""
    return np.array([encode(t, max_len) for t in texts], dtype=np.int64).reshape(-1, max_len)
