"""Server-side re-check of a conformance report (planning/conformance-manifest.md).

The gateway never trusts `client_passed`: it compares every declared value and every synthetic
output against the manifest itself, in the fixed order below, and stops at the first failure.
The same function lets a client check its own report before sending it.

    1  protocol_version_mismatch        8  encoding_mismatch
    2  manifest_version_mismatch        9  preprocessing_version_mismatch
    3  model_version_mismatch          10  label_semantics_mismatch
    4  model_shape_mismatch            11  label_range_mismatch
    5  feature_schema_mismatch         12  reference_output_mismatch
    6  feature_order_mismatch          13  forward_output_mismatch
    7  unit_mismatch                   14  invalid_report

`invalid_report` is reported where the problem is found (missing or unknown test ids,
non-finite numbers, wrong lengths), since a malformed report cannot be checked further.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from cfl.core.manifest import manifest_id
from cfl.protocol import PROTOCOL_VERSION
from cfl.protocol.messages import ConformanceReportRequest

NON_NUMERIC_UNITS = {"categorical": "category", "text": "text"}


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    reason_code: str | None = None
    message: str = "conformant"
    details: dict[str, Any] = field(default_factory=dict)


class _Fail(Exception):
    def __init__(self, code: str, message: str, **details: Any):
        super().__init__(message)
        self.result = CheckResult(False, code, message, details)


def expected_units(manifest: dict) -> list[str]:
    """The `feature_units` a conforming client declares: numeric unit, "category" or "text"."""
    return [
        f["unit"] if f["type"] == "numeric" else NON_NUMERIC_UNITS[f["type"]]
        for f in manifest["input_schema"]["features"]
    ]


def check_report(
    manifest: dict, manifest_sha256: str, report: ConformanceReportRequest
) -> CheckResult:
    try:
        _check(manifest, manifest_sha256, report)
    except _Fail as failure:
        return failure.result
    return CheckResult(True)


def _check(manifest: dict, manifest_sha256: str, r: ConformanceReportRequest) -> None:
    if r.protocol_version != PROTOCOL_VERSION:
        raise _Fail(
            "protocol_version_mismatch",
            f"protocol {r.protocol_version} is not supported; the server speaks {PROTOCOL_VERSION}",
            expected=PROTOCOL_VERSION,
            received=r.protocol_version,
        )

    if r.manifest_version != manifest_id(manifest) or r.manifest_sha256 != manifest_sha256:
        raise _Fail(
            "manifest_version_mismatch",
            "the report was made for a different manifest than the run's",
            expected={"manifest_version": manifest_id(manifest), "sha256": manifest_sha256},
            received={"manifest_version": r.manifest_version, "sha256": r.manifest_sha256},
        )

    model, declared = manifest["model"], r.declared_model
    expected_model = (model["architecture_id"], model["architecture_version"])
    if (declared.architecture_id, declared.architecture_version) != expected_model:
        raise _Fail(
            "model_version_mismatch",
            "model architecture or version differs from the manifest",
            expected=list(expected_model),
            received=[declared.architecture_id, declared.architecture_version],
        )

    expected_params = [(p["name"], list(p["shape"]), p["dtype"]) for p in model["parameters"]]
    received_params = [(p.name, list(p.shape), p.dtype) for p in declared.parameters]
    if received_params != expected_params:
        first = next(
            (
                i
                for i, (a, b) in enumerate(zip(expected_params, received_params, strict=False))
                if a != b
            ),
            min(len(expected_params), len(received_params)),
        )
        raise _Fail(
            "model_shape_mismatch",
            "parameter names, shapes or dtypes differ from the manifest",
            index=first,
            expected=expected_params[first] if first < len(expected_params) else None,
            received=received_params[first] if first < len(received_params) else None,
        )
    if declared.fingerprint is not None and declared.fingerprint != model["fingerprint"]:
        raise _Fail(
            "model_shape_mismatch",
            "model fingerprint differs from the manifest",
            expected=model["fingerprint"],
            received=declared.fingerprint,
        )

    features = manifest["input_schema"]["features"]
    names = [f["name"] for f in features]
    types = {f["name"]: f["type"] for f in features}
    d = r.declared_input
    if not (len(d.feature_names) == len(d.feature_types) == len(d.feature_units)):
        raise _Fail(
            "invalid_report", "feature_names, feature_types and feature_units differ in length"
        )
    if len(d.feature_names) != len(names) or set(d.feature_names) != set(names):
        raise _Fail(
            "feature_schema_mismatch",
            "declared features differ from the manifest",
            missing=sorted(set(names) - set(d.feature_names)),
            unexpected=sorted(set(d.feature_names) - set(names)),
        )
    declared_types = dict(zip(d.feature_names, d.feature_types, strict=True))
    wrong_types = {n: declared_types[n] for n in names if declared_types[n] != types[n]}
    if wrong_types:
        raise _Fail(
            "feature_schema_mismatch",
            "feature types differ from the manifest",
            expected={n: types[n] for n in wrong_types},
            received=wrong_types,
        )
    if d.feature_names != names:
        pairs = zip(names, d.feature_names, strict=True)
        index = next(i for i, (a, b) in enumerate(pairs) if a != b)
        raise _Fail(
            "feature_order_mismatch",
            "declared feature order does not match the manifest ordering",
            index=index,
            expected=names[index],
            received=d.feature_names[index],
        )
    units = expected_units(manifest)
    if d.feature_units != units:
        triples = zip(names, units, d.feature_units, strict=True)
        wrong = {n: (e, g) for n, e, g in triples if e != g}
        raise _Fail(
            "unit_mismatch",
            "feature units differ from the manifest",
            features={n: {"expected": e, "received": g} for n, (e, g) in wrong.items()},
        )

    for f in features:
        if f["type"] == "categorical" and d.categories.get(f["name"]) != f["categories"]:
            raise _Fail(
                "encoding_mismatch",
                f"categories of {f['name']} differ from the manifest (values or order)",
                feature=f["name"],
                expected=f["categories"],
                received=d.categories.get(f["name"]),
            )
    expected_tokenizer = manifest["preprocessing"].get("tokenizer")
    if expected_tokenizer is not None and d.tokenizer != expected_tokenizer:
        raise _Fail(
            "encoding_mismatch",
            "tokenizer differs from the manifest",
            expected=expected_tokenizer,
            received=d.tokenizer,
        )

    if d.preprocessing_version != manifest["preprocessing"]["version"]:
        raise _Fail(
            "preprocessing_version_mismatch",
            "preprocessing version differs from the manifest",
            expected=manifest["preprocessing"]["version"],
            received=d.preprocessing_version,
        )

    label, dl = manifest["label"], r.declared_label
    if dl is None:
        raise _Fail("invalid_report", "declared_label is missing")
    expected_label = (label["column"], label["positive_value"], label["negative_value"])
    if (dl.column, dl.positive_value, dl.negative_value) != expected_label:
        raise _Fail(
            "label_semantics_mismatch",
            "label column or meaning differs from the manifest",
            expected={
                "column": label["column"],
                "positive_value": label["positive_value"],
                "negative_value": label["negative_value"],
            },
            received={
                "column": dl.column,
                "positive_value": dl.positive_value,
                "negative_value": dl.negative_value,
            },
        )
    if dl.encoded_values != label["encoded_values"]:
        raise _Fail(
            "label_range_mismatch",
            "encoded label values differ from the manifest",
            expected=label["encoded_values"],
            received=dl.encoded_values,
        )

    validation = manifest["validation"]
    _compare_outputs(
        expected={t["test_id"]: t["expected_model_input"] for t in manifest["reference_tests"]},
        received={o.test_id: o.output_features for o in r.reference_outputs},
        atol=validation["reference_atol"],
        code="reference_output_mismatch",
        what="reference output",
    )
    _compare_outputs(
        expected={t["test_id"]: t["expected_logits"] for t in manifest["forward_reference_tests"]},
        received={o.test_id: o.logits for o in r.forward_outputs},
        atol=validation["forward_atol"],
        code="forward_output_mismatch",
        what="forward output",
    )


def _compare_outputs(
    expected: dict[str, list], received: dict[str, list], atol: float, code: str, what: str
) -> None:
    if set(received) != set(expected):
        raise _Fail(
            "invalid_report",
            f"{what} test ids do not match the manifest",
            missing=sorted(set(expected) - set(received)),
            unknown=sorted(set(received) - set(expected)),
        )
    for test_id, want in expected.items():
        got = received[test_id]
        if len(got) != len(want) or not all(math.isfinite(v) for v in got):
            raise _Fail(
                "invalid_report",
                f"{what} {test_id} has the wrong length or non-finite values",
                test_id=test_id,
                expected_length=len(want),
                received_length=len(got),
            )
        diffs = [abs(a - b) for a, b in zip(want, got, strict=True)]
        if max(diffs, default=0.0) > atol:
            index = max(range(len(diffs)), key=diffs.__getitem__)
            raise _Fail(
                code,
                f"{what} {test_id} differs from the expected values",
                test_id=test_id,
                index=index,
                expected=want[index],
                received=got[index],
                atol=atol,
            )
