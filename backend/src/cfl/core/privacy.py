"""Client-side privacy primitives: gradient clipping, calibrated Local DP noise and
fixed-point quantization for Secure Aggregation.

The pipeline on the client side is:
    1. clip_delta(w_i, w_global, C)        -> Δw_i  (L2-bounded update delta)
    2. add_dp_noise(Δw_i, σ_t, rng)        -> Δw_i' (differentially private delta)
    3. scale_by_trust(Δw_i', α_i)          -> w̃_i   (trust-weighted delta)
    4. quantize_state_dict(w̃_i, scale, M)  -> ŵ_i   (fixed-point, ready for SecAgg masking)

After blind aggregation the server calls:
    dequantize_state_dict(Σ ŵ_i, scale, M) -> Σ w̃_i  (back to float)
    w^{t+1} = w^t + Σ w̃_i
"""

from __future__ import annotations

import math

import torch

StateDict = dict[str, torch.Tensor]

# ---------------------------------------------------------------------------
# 1. Gradient Clipping
# ---------------------------------------------------------------------------

DEFAULT_CLIP_NORM: float = 1.0


def clip_delta(
    weights: StateDict,
    global_weights: StateDict,
    clip_norm: float = DEFAULT_CLIP_NORM,
) -> StateDict:
    """Compute Δw_i = w_i − w^t and clip its L2 norm to *clip_norm*.

    If ‖Δw_i‖₂ > C the entire delta is scaled down so that ‖Δw_i‖₂ = C.
    Returns a **new** state dict; both inputs are left unchanged.
    """
    if clip_norm <= 0:
        raise ValueError(f"clip_norm must be positive, got {clip_norm}")

    # Compute delta and its global L2 norm across all parameters.
    delta: StateDict = {}
    sq_sum = torch.zeros((), dtype=torch.float64)
    for name in weights:
        d = weights[name].to(torch.float64) - global_weights[name].to(torch.float64)
        delta[name] = d
        sq_sum += d.square().sum()
    global_norm = sq_sum.sqrt().item()

    # Scale down when exceeding the bound.
    if global_norm > clip_norm:
        scale = clip_norm / global_norm
        delta = {name: (d * scale).to(weights[name].dtype) for name, d in delta.items()}
    else:
        delta = {name: d.to(weights[name].dtype) for name, d in delta.items()}
    return delta


# ---------------------------------------------------------------------------
# 2. Calibrated Local DP Noise
# ---------------------------------------------------------------------------

DEFAULT_BASE_SIGMA: float = 0.01


def calibrate_noise_std(base_sigma: float, num_participants: int) -> float:
    """σ_t = σ₀ / √N_t — more participants allow less noise per client."""
    if num_participants < 1:
        raise ValueError(f"num_participants must be >= 1, got {num_participants}")
    if base_sigma < 0:
        raise ValueError(f"base_sigma must be non-negative, got {base_sigma}")
    return base_sigma / math.sqrt(num_participants)


def add_dp_noise(
    delta: StateDict,
    sigma: float,
    rng: torch.Generator | None = None,
) -> StateDict:
    """Add calibrated Gaussian noise η_i ∼ N(0, σ²I) to every parameter tensor.

    When *sigma* is 0 the delta is returned as-is (no copy, for efficiency in the
    maximum-accuracy regime).  A *rng* generator can be passed for reproducibility.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if sigma == 0.0:
        return delta
    noisy: StateDict = {}
    for name, tensor in delta.items():
        noise = torch.normal(
            mean=0.0,
            std=sigma,
            size=tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            generator=rng,
        )
        noisy[name] = tensor + noise
    return noisy


# ---------------------------------------------------------------------------
# 3. Trust-weight scaling
# ---------------------------------------------------------------------------


def scale_by_trust(delta: StateDict, alpha: float) -> StateDict:
    """Multiply every parameter by the coordinator-assigned trust weight α_i^t.

    The coordinator normalises Σ α_i = 1, so after blind summation the server gets
    Σ α_i Δw_i' directly.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    return {name: tensor * alpha for name, tensor in delta.items()}


# ---------------------------------------------------------------------------
# 4. Fixed-point quantization for SecAgg
# ---------------------------------------------------------------------------

DEFAULT_SCALE: float = 2**24
DEFAULT_MODULUS: int = 2**48  # large enough to hold clipped + noised values


def quantize_state_dict(
    delta: StateDict,
    scale: float = DEFAULT_SCALE,
    modulus: int = DEFAULT_MODULUS,
) -> dict[str, torch.Tensor]:
    """Convert float32 tensors to non-negative integers in [0, modulus).

    Mapping: q = round(x * scale) mod modulus
    Negative values are wrapped to [modulus/2, modulus) so that modular addition
    preserves the signed sum.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if modulus < 2:
        raise ValueError(f"modulus must be >= 2, got {modulus}")
    quantized: dict[str, torch.Tensor] = {}
    for name, tensor in delta.items():
        # int64 can hold values up to ~9.2e18, more than enough for 2^48.
        q = torch.round(tensor.to(torch.float64) * scale).to(torch.int64)
        q = q % modulus  # wraps negatives into [modulus/2 .. modulus)
        quantized[name] = q
    return quantized


def dequantize_state_dict(
    quantized: dict[str, torch.Tensor],
    scale: float = DEFAULT_SCALE,
    modulus: int = DEFAULT_MODULUS,
) -> StateDict:
    """Inverse of quantize_state_dict: int64 (mod M) -> float32.

    Values in [modulus/2, modulus) are interpreted as negative.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if modulus < 2:
        raise ValueError(f"modulus must be >= 2, got {modulus}")
    half = modulus // 2
    result: StateDict = {}
    for name, tensor in quantized.items():
        signed = tensor.clone()
        signed[signed >= half] -= modulus
        result[name] = (signed.to(torch.float64) / scale).to(torch.float32)
    return result
