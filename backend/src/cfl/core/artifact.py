"""Final model artifact: the trained global model delivered to every client (WP8).

The coordinator builds it when a run completes; clients download it and verify it. Files:

    model.onnx         the model as ONNX, the format clients run
    model.safetensors  the same weights for PyTorch users and for inspection
    manifest.json      the run's conformance manifest: how to turn raw input into model input
    artifact.json      versions, checksums of the files above, and reference predictions

Before anything is written, the ONNX model is run with ONNX Runtime and must reproduce the
PyTorch logits on the manifest's synthetic reference inputs. A client repeats that check on its
own machine with verify_artifact(). The reference inputs are synthetic; no client or held-out
data is included.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import onnxruntime as ort
import torch

from cfl.core.contract import ModelContract, StateDict
from cfl.core.serialization import sha256, state_dict_to_bytes

ARTIFACT_FORMAT_VERSION = 1
ONNX_OPSET = 17
ATOL = 1e-5
INPUT_NAME, OUTPUT_NAME = "input", "logits"

MODEL_ONNX = "model.onnx"
MODEL_SAFETENSORS = "model.safetensors"
MANIFEST = "manifest.json"
ARTIFACT = "artifact.json"


class ArtifactError(RuntimeError):
    pass


@dataclass(frozen=True)
class FinalArtifact:
    files: dict[str, bytes]  # file name -> bytes, including artifact.json
    descriptor: dict

    @property
    def onnx_sha256(self) -> str:
        return self.descriptor["files"][MODEL_ONNX]["sha256"]


def final_key(run_id: str, name: str) -> str:
    return f"{run_id}/final/{name}"


def _reference_inputs(manifest: dict) -> tuple[list[str], torch.Tensor]:
    dtype = (
        torch.int64
        if manifest["input_schema"]["model_input"]["dtype"] == "int64"
        else (torch.float32)
    )
    tests = manifest["reference_tests"]
    return (
        [t["test_id"] for t in tests],
        torch.tensor([t["expected_model_input"] for t in tests], dtype=dtype),
    )


def _run_onnx(onnx_bytes: bytes, inputs: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    return session.run([OUTPUT_NAME], {INPUT_NAME: inputs})[0]


def export_onnx(model: torch.nn.Module, example: torch.Tensor) -> bytes:
    """Export with a dynamic batch dimension (and dynamic sequence length for token inputs)."""
    axes = {0: "batch"} if example.dtype.is_floating_point else {0: "batch", 1: "sequence"}
    buf = io.BytesIO()
    torch.onnx.export(
        model,
        (example,),
        buf,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: axes, OUTPUT_NAME: {0: "batch"}},
        opset_version=ONNX_OPSET,
        dynamo=False,  # the dynamo exporter needs onnxscript, which is not a dependency yet
    )
    return buf.getvalue()


def build_artifact(
    *,
    run_id: str,
    model_version: int,
    weights: StateDict,
    build_model: Callable[[], torch.nn.Module],
    contract: ModelContract,
    manifest_bytes: bytes,
    created_at: datetime,
) -> FinalArtifact:
    contract.check(weights)
    manifest = json.loads(manifest_bytes)
    if manifest["model"]["fingerprint"] != contract.fingerprint():
        raise ArtifactError("manifest does not describe this model")

    model = build_model()
    model.load_state_dict(weights)
    model.eval()
    test_ids, inputs = _reference_inputs(manifest)
    with torch.no_grad():
        expected = model(inputs).numpy()

    onnx_bytes = export_onnx(model, inputs)
    produced = _run_onnx(onnx_bytes, inputs.numpy())
    max_diff = float(np.abs(produced - expected).max())
    if not np.allclose(produced, expected, atol=ATOL):
        raise ArtifactError(f"ONNX output differs from PyTorch by {max_diff:.2e} (> {ATOL})")

    files = {
        MODEL_ONNX: onnx_bytes,
        MODEL_SAFETENSORS: state_dict_to_bytes(weights),
        MANIFEST: manifest_bytes,
    }
    descriptor = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "run_id": run_id,
        "model_version": model_version,
        "created_at": created_at.isoformat(),
        "architecture_id": contract.architecture_id,
        "architecture_version": contract.architecture_version,
        "model_fingerprint": contract.fingerprint(),
        "manifest_version": manifest["manifest_version"],
        "files": {
            name: {"sha256": sha256(data), "bytes": len(data)} for name, data in files.items()
        },
        "onnx": {
            "opset": ONNX_OPSET,
            "producer": f"torch {torch.__version__}",
            "input": {
                "name": INPUT_NAME,
                "dtype": manifest["input_schema"]["model_input"]["dtype"],
                "shape": ["batch", manifest["model"]["input_dim"]],
            },
            "output": {
                "name": OUTPUT_NAME,
                "dtype": "float32",
                "shape": ["batch", manifest["model"]["output_dim"]],
                "meaning": manifest["label"]["meaning"],
            },
        },
        "preprocessing": manifest["preprocessing"],
        "reference_predictions": [
            {
                "test_id": test_id,
                "model_input": manifest["reference_tests"][i]["expected_model_input"],
                "logits": [float(v) for v in expected[i]],
                "predicted_class": int(expected[i].argmax()),
            }
            for i, test_id in enumerate(test_ids)
        ],
        "verification": {"atol": ATOL, "max_abs_diff_onnx_vs_pytorch": max_diff},
    }
    files[ARTIFACT] = (json.dumps(descriptor, indent=2, ensure_ascii=False) + "\n").encode()
    return FinalArtifact(files, descriptor)


def verify_artifact(files: dict[str, bytes], expected_onnx_sha256: str) -> dict:
    """Client-side check of a downloaded artifact. `expected_onnx_sha256` comes from the
    gateway (runs.final_model_sha256), not from the artifact itself. Raises ArtifactError on any
    mismatch; returns the descriptor on success."""
    missing = {MODEL_ONNX, ARTIFACT} - files.keys()
    if missing:
        raise ArtifactError(f"missing files: {sorted(missing)}")
    if sha256(files[MODEL_ONNX]) != expected_onnx_sha256:
        raise ArtifactError("model.onnx does not match the checksum published by the server")
    descriptor = json.loads(files[ARTIFACT])
    for name, info in descriptor["files"].items():
        if name in files and sha256(files[name]) != info["sha256"]:
            raise ArtifactError(f"{name} does not match artifact.json")

    dtype = np.int64 if descriptor["onnx"]["input"]["dtype"] == "int64" else np.float32
    refs = descriptor["reference_predictions"]
    inputs = np.array([r["model_input"] for r in refs], dtype=dtype)
    produced = _run_onnx(files[MODEL_ONNX], inputs)
    expected = np.array([r["logits"] for r in refs], dtype=np.float32)
    if not np.allclose(produced, expected, atol=descriptor["verification"]["atol"]):
        raise ArtifactError("model.onnx does not reproduce the reference predictions")
    if [int(c) for c in produced.argmax(axis=1)] != [r["predicted_class"] for r in refs]:
        raise ArtifactError("predicted classes differ from the reference predictions")
    return descriptor
