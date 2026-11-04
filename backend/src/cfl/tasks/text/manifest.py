"""Conformance manifest for the text task (SMS Spam), built from the tokenizer and model."""

from __future__ import annotations

import torch

from cfl.core.manifest import VALIDATION, forward_tests, model_section
from cfl.tasks.text import model as text_model
from cfl.tasks.text import tokenizer

MANIFEST_VERSION = "1.0.0"

TOKENIZER = {
    "algorithm": "hashed_ngrams",
    "version": tokenizer.TOKENIZER_VERSION,
    "normalization": "unicode NFKC, then lower-case",
    "token_pattern": tokenizer.TOKEN_PATTERN,
    "ngrams": [1, 2],
    "ngram_order": "interleaved: t0, t1, t0_t1, t2, t1_t2, ...",
    "ngram_joiner": "_",
    "hash": "fnv1a_32 over utf-8 bytes",
    "id_formula": "1 + hash(feature) mod (num_buckets - 1)",
    "num_buckets": tokenizer.NUM_BUCKETS,
    "pad_id": tokenizer.PAD_ID,
    "max_len": tokenizer.MAX_LEN,
    "padding": "keep the first max_len ids, right-pad with pad_id",
}

# Synthetic messages covering the tokenizer's edge cases.
REFERENCE_TEXTS: list[tuple[str, str]] = [
    ("sms-spam-reference-001", "WINNER!! Claim your FREE prize now: call 09061701461"),
    ("sms-spam-reference-002", "Ok lar... see you at 6?"),
    ("sms-spam-reference-003", "ＦＲＥＥ ｅｎｔｒｙ ２ wkly comp"),  # full-width, folded by NFKC
    ("sms-spam-reference-004", "café naïve résumé"),  # non-ASCII letters split tokens
    ("sms-spam-reference-005", " ".join(f"word{i}" for i in range(50))),  # truncated
    ("sms-spam-reference-006", ""),  # all padding
]


def build_manifest() -> dict:
    contract = text_model.contract()
    reference_tests = [
        {
            "test_id": test_id,
            "synthetic_raw_input": {"text": text},
            "expected_model_input": tokenizer.encode(text),
        }
        for test_id, text in REFERENCE_TEXTS
    ]

    return {
        "manifest_version": MANIFEST_VERSION,
        "dataset": {
            "name": "sms_spam",
            "dataset_version": "sms-spam-collection-uci-228",
            "source_id": "UCI ML Repository dataset 228; CC BY 4.0",
            "raw_label_column": "label",
        },
        "model": model_section(
            contract,
            task="binary_classification",
            input_dim=tokenizer.MAX_LEN,
            output_dim=text_model.NUM_CLASSES,
        ),
        "preprocessing": {
            "version": tokenizer.TOKENIZER_VERSION,
            "implementation_id": "cfl.tasks.text.tokenizer.encode",
            "encoding": "hashed unigram and bigram ids, see tokenizer",
            "scaling": "none",
            "fit_policy": "nothing_fitted",
            "tokenizer": TOKENIZER,
        },
        "input_schema": {
            "features": [{"name": "text", "type": "text", "encoding": "utf-8"}],
            "model_input": {
                "feature_names": [f"token_{i}" for i in range(tokenizer.MAX_LEN)],
                "shape": [1, tokenizer.MAX_LEN],
                "dtype": "int64",
            },
        },
        "label": {
            "column": "label",
            "positive_value": "spam",
            "negative_value": "ham",
            "encoded_values": [0, 1],
            "meaning": "SMS message is spam; 1=spam (positive), 0=ham (negative)",
        },
        "reference_tests": reference_tests,
        "forward_reference_tests": forward_tests(
            text_model.SmsHashEmbed(),
            contract,
            [
                (t["test_id"].replace("reference", "forward"), t["expected_model_input"])
                for t in reference_tests
            ],
            torch.int64,
        ),
        "validation": VALIDATION,
    }
