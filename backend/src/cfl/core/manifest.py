"""Shared pieces of the conformance manifest (planning/conformance-manifest.md).

Each task builds its manifest from code (cfl.tasks.<task>.manifest) so the published JSON can
never drift from the model and pre-processing that clients actually run.
"""

from __future__ import annotations

from math import prod

import numpy as np
import torch

from cfl.core.contract import ModelContract, StateDict, forward_reference

REFERENCE_WEIGHTS_ID = "reference_sin_v1"

VALIDATION = {
    "check_order": [
        "protocol_version",
        "manifest_version",
        "model_version",
        "model_shape",
        "feature_schema",
        "feature_order",
        "units",
        "encoding",
        "preprocessing_version",
        "label_semantics",
        "label_range",
        "reference_outputs",
        "forward_outputs",
        "report_schema",
    ],
    "reference_atol": 1e-6,
    "forward_atol": 1e-5,
    "failure_codes": [
        "protocol_version_mismatch",
        "manifest_version_mismatch",
        "model_version_mismatch",
        "model_shape_mismatch",
        "feature_schema_mismatch",
        "feature_order_mismatch",
        "unit_mismatch",
        "encoding_mismatch",
        "preprocessing_version_mismatch",
        "label_semantics_mismatch",
        "label_range_mismatch",
        "reference_output_mismatch",
        "forward_output_mismatch",
        "invalid_report",
    ],
}


def manifest_id(manifest: dict) -> str:
    """The identity a run pins and clients pass conformance for: "<dataset>@<version>".

    The version alone is not enough: the tabular and the text manifest are both 1.0.0, and a
    client that passed one must not count as conformant for the other.
    """
    return f"{manifest['dataset']['name']}@{manifest['manifest_version']}"


def reference_weights(contract: ModelContract) -> StateDict:
    """Language-independent weights for the forward reference test (REFERENCE_WEIGHTS_ID).

    Walk the parameters in manifest order and each tensor in row-major order, keeping one
    running index j across all parameters (starting at 0). Element j is
    float32(0.1 * sin(j + 1)), computed in float64. Because the values depend on the position,
    a transposed or reordered parameter produces different logits.
    """
    weights: StateDict = {}
    offset = 0
    for spec in contract.params:
        n = prod(spec.shape)
        j = torch.arange(offset + 1, offset + n + 1, dtype=torch.float64)
        weights[spec.name] = (0.1 * torch.sin(j)).to(torch.float32).reshape(spec.shape)
        offset += n
    return weights


def as_float32(values) -> list[float]:
    """Shortest decimal that round-trips to the same float32, for readable JSON."""
    return [float(str(np.float32(v))) for v in values]


def same_manifest(a, b, atol: float = 1e-6) -> bool:
    """Structural equality that tolerates last-bit float differences (e.g. BLAS-dependent
    logits), so regenerating a manifest on another machine does not count as a change."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_manifest(a[k], b[k], atol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(same_manifest(x, y, atol) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) or isinstance(b, float):
        both_numbers = all(isinstance(v, int | float) and not isinstance(v, bool) for v in (a, b))
        return both_numbers and abs(a - b) <= atol
    return type(a) is type(b) and a == b


def model_section(
    contract: ModelContract, *, task: str, input_dim: int, output_dim: int
) -> dict:
    return {
        "architecture_id": contract.architecture_id,
        "architecture_version": contract.architecture_version,
        "framework": "pytorch",
        "task": task,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "fingerprint": contract.fingerprint(),
        "parameters": contract.manifest_parameters(),
    }


def forward_tests(
    model: torch.nn.Module,
    contract: ModelContract,
    cases: list[tuple[str, list]],
    dtype: torch.dtype,
) -> list[dict]:
    """Expected logits of `model` with reference_weights() on each (test_id, model_input)."""
    weights = reference_weights(contract)
    tests = []
    for test_id, model_input in cases:
        logits = forward_reference(model, weights, torch.tensor([model_input], dtype=dtype))[0]
        tests.append(
            {
                "test_id": test_id,
                "weights": REFERENCE_WEIGHTS_ID,
                "model_input": model_input,
                "expected_logits": as_float32(logits),
            }
        )
    return tests
