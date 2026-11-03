"""Model contract: the exact parameter layout every client update must match.

Aggregation averages parameters element by element, so every client (managed or independent)
must use the same architecture: identical parameter names, shapes and dtypes. The contract is
generated from the reference PyTorch model, never written by hand, and summarized by a
fingerprint (SHA-256 of its canonical JSON) that the gateway and aggregator compare.

Rules enforced by from_model():
- only float32 parameters, no buffers (no BatchNorm running statistics or integer counters),
  so every entry in the state dict can be averaged;
- weights use the PyTorch layout (Linear.weight is [out, in], row-major).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass

import torch

StateDict = dict[str, torch.Tensor]

ALLOWED_DTYPES = {torch.float32: "float32"}


class ContractViolation(ValueError):
    """A model or update does not match the contract. `code` matches the manifest's codes."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class ParamSpec:
    name: str
    shape: tuple[int, ...]
    dtype: str

    def to_dict(self) -> dict:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


@dataclass(frozen=True)
class ModelContract:
    architecture_id: str
    architecture_version: str
    params: tuple[ParamSpec, ...]

    @classmethod
    def from_model(
        cls, model: torch.nn.Module, architecture_id: str, architecture_version: str
    ) -> ModelContract:
        buffers = [name for name, _ in model.named_buffers()]
        if buffers:
            raise ContractViolation("model_shape_mismatch", f"buffers are not allowed: {buffers}")
        specs = []
        for name, param in model.named_parameters():
            if param.dtype not in ALLOWED_DTYPES:
                raise ContractViolation(
                    "model_shape_mismatch", f"{name} has dtype {param.dtype}; only float32"
                )
            specs.append(ParamSpec(name, tuple(param.shape), ALLOWED_DTYPES[param.dtype]))
        return cls(architecture_id, architecture_version, tuple(specs))

    def manifest_parameters(self) -> list[dict]:
        """The `model.parameters` list of the conformance manifest."""
        return [p.to_dict() for p in self.params]

    def fingerprint(self) -> str:
        canonical = json.dumps(
            {
                "architecture_id": self.architecture_id,
                "architecture_version": self.architecture_version,
                "parameters": self.manifest_parameters(),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def check(self, weights: StateDict) -> None:
        """Raise ContractViolation unless `weights` matches the contract exactly."""
        expected = {p.name: p for p in self.params}
        missing = sorted(expected.keys() - weights.keys())
        extra = sorted(weights.keys() - expected.keys())
        if missing or extra:
            raise ContractViolation(
                "model_shape_mismatch", f"missing parameters {missing}, extra parameters {extra}"
            )
        for name, spec in expected.items():
            tensor = weights[name]
            if tuple(tensor.shape) != spec.shape:
                raise ContractViolation(
                    "model_shape_mismatch",
                    f"{name} has shape {tuple(tensor.shape)}, expected {spec.shape}",
                )
            if ALLOWED_DTYPES.get(tensor.dtype) != spec.dtype:
                raise ContractViolation(
                    "model_shape_mismatch",
                    f"{name} has dtype {tensor.dtype}, expected {spec.dtype}",
                )
            if not torch.isfinite(tensor).all():
                raise ContractViolation("non_finite_weights", f"{name} contains NaN or Inf")


def initial_weights(build_model: Callable[[], torch.nn.Module], seed: int) -> StateDict:
    """Global model v0. Created once by the server with a fixed seed; clients never initialize."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = build_model()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def forward_reference(model: torch.nn.Module, weights: StateDict, inputs: torch.Tensor) -> list:
    """Logits of `model` loaded with `weights` on `inputs`: the expected outputs of the forward
    reference test that independent clients must reproduce."""
    model.load_state_dict(weights)
    model.eval()
    with torch.no_grad():
        return model(inputs).tolist()
