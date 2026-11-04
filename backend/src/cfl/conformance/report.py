"""Build a conformance report from a site's own code (client side).

The client runs its pre-processing on the manifest's synthetic raw inputs and its model on the
forward reference inputs with the reference weights, and declares what it implements. Nothing
from the site's private data is included.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from cfl.conformance.checker import check_report, expected_units
from cfl.core.contract import ModelContract, forward_reference
from cfl.core.manifest import manifest_id, reference_weights
from cfl.protocol import PROTOCOL_VERSION
from cfl.protocol.messages import (
    ConformanceCheck,
    ConformanceReportRequest,
    DeclaredInput,
    DeclaredLabel,
    DeclaredModel,
    ForwardTestOutput,
    ModelParameterSpec,
    ReferenceTestOutput,
)


def build_report(
    *,
    manifest: dict,
    manifest_sha256: str,
    client_id: str,
    client_version: str,
    preprocess: Callable[[dict], list],
    build_model: Callable[[], torch.nn.Module],
    contract: ModelContract,
) -> ConformanceReportRequest:
    features = manifest["input_schema"]["features"]
    label = manifest["label"]
    input_dtype = (
        torch.int64
        if manifest["input_schema"]["model_input"]["dtype"] == "int64"
        else torch.float32
    )
    weights = reference_weights(contract)
    forward = []
    for test in manifest["forward_reference_tests"]:
        x = torch.tensor([test["model_input"]], dtype=input_dtype)
        forward.append(
            ForwardTestOutput(
                test_id=test["test_id"], logits=forward_reference(build_model(), weights, x)[0]
            )
        )

    report = ConformanceReportRequest(
        manifest_version=manifest_id(manifest),
        manifest_sha256=manifest_sha256,
        client_id=client_id,
        client_version=client_version,
        protocol_version=PROTOCOL_VERSION,
        declared_model=DeclaredModel(
            architecture_id=contract.architecture_id,
            architecture_version=contract.architecture_version,
            fingerprint=contract.fingerprint(),
            parameters=[
                ModelParameterSpec(name=p.name, shape=list(p.shape), dtype=p.dtype)
                for p in contract.params
            ],
        ),
        declared_input=DeclaredInput(
            feature_names=[f["name"] for f in features],
            feature_types=[f["type"] for f in features],
            feature_units=expected_units(manifest),
            preprocessing_version=manifest["preprocessing"]["version"],
            categories={f["name"]: f["categories"] for f in features if f["type"] == "categorical"},
            tokenizer=manifest["preprocessing"].get("tokenizer"),
        ),
        declared_label=DeclaredLabel(
            column=label["column"],
            positive_value=label["positive_value"],
            negative_value=label["negative_value"],
            encoded_values=label["encoded_values"],
        ),
        client_passed=False,
        reference_outputs=[
            ReferenceTestOutput(
                test_id=t["test_id"], output_features=list(preprocess(t["synthetic_raw_input"]))
            )
            for t in manifest["reference_tests"]
        ],
        forward_outputs=forward,
    )
    # The client's own verdict, for its log; the gateway re-checks everything.
    result = check_report(manifest, manifest_sha256, report)
    return report.model_copy(
        update={
            "client_passed": result.passed,
            "checks": [
                ConformanceCheck(code=result.reason_code or "conformant", passed=result.passed)
            ],
            "client_failure_codes": [] if result.passed else [result.reason_code],
        }
    )
