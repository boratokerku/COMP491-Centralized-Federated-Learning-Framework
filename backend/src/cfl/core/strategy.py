"""Aggregation strategies (FedAvg, FedProx) and dynamic trust weighting.

The privacy extension adds two capabilities:
- **Delta-based aggregation:** when SecAgg is active the server receives masked
  *deltas* (Δw_i, already trust-weighted by α_i on the client side), so aggregation
  becomes ``w^{t+1} = w^t + Σ ŵ_i`` instead of a weighted average of full weights.
- **Metadata-based dynamic trust weights (α_i^t):** computed from behavioural
  signals only (participation rate, on-time delivery, sample-count consistency,
  validation-metric stability), never from the raw weights, so SecAgg blindness is
  preserved.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch

StateDict = dict[str, torch.Tensor]


@dataclass(frozen=True)
class ClientUpdate:
    client_id: str
    weights: StateDict
    num_samples: int


class Strategy(ABC):
    """Server-side aggregation plus the config each client receives for local training."""

    name: str

    @abstractmethod
    def aggregate(self, global_weights: StateDict, updates: list[ClientUpdate]) -> StateDict: ...

    def client_config(self) -> dict:
        return {}


def weighted_average(updates: list[ClientUpdate]) -> StateDict:
    if not updates:
        raise ValueError("cannot aggregate an empty list of updates")
    total = sum(u.num_samples for u in updates)
    if total <= 0:
        raise ValueError("total number of samples must be positive")

    keys = updates[0].weights.keys()
    for u in updates[1:]:
        if u.weights.keys() != keys:
            raise ValueError(f"update from {u.client_id} has mismatching parameter names")

    result: StateDict = {}
    for key in keys:
        acc = torch.zeros_like(updates[0].weights[key], dtype=torch.float64)
        for u in updates:
            acc += u.weights[key].to(torch.float64) * (u.num_samples / total)
        result[key] = acc.to(updates[0].weights[key].dtype)
    return result


def delta_sum(global_weights: StateDict, deltas: list[StateDict]) -> StateDict:
    """SecAgg-compatible aggregation: w^{t+1} = w^t + Σ Δw̃_i.

    Each delta is assumed to be **already trust-weighted** (α_i × Δw_i') on the client
    side before masking, so the server just adds the dequantized/unmasked sum to the
    current global model.
    """
    if not deltas:
        raise ValueError("cannot aggregate an empty list of deltas")

    result: StateDict = {}
    for key in global_weights:
        acc = global_weights[key].to(torch.float64).clone()
        for d in deltas:
            acc += d[key].to(torch.float64)
        result[key] = acc.to(global_weights[key].dtype)
    return result


class FedAvg(Strategy):
    name = "fedavg"

    def aggregate(self, global_weights: StateDict, updates: list[ClientUpdate]) -> StateDict:
        return weighted_average(updates)


class FedProx(Strategy):
    """Same server-side aggregation as FedAvg; clients add a proximal term with coefficient mu."""

    name = "fedprox"

    def __init__(self, mu: float = 0.01):
        if mu < 0:
            raise ValueError("mu must be non-negative")
        self.mu = mu

    def aggregate(self, global_weights: StateDict, updates: list[ClientUpdate]) -> StateDict:
        return weighted_average(updates)

    def client_config(self) -> dict:
        return {"proximal_mu": self.mu}


def proximal_term(model: torch.nn.Module, global_weights: StateDict, mu: float) -> torch.Tensor:
    """(mu / 2) * ||w - w_global||^2, added to the client's local loss in FedProx."""
    term = torch.zeros((), device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        term = term + torch.sum((param - global_weights[name].to(param.device)) ** 2)
    return 0.5 * mu * term


def build_strategy(name: str, **kwargs) -> Strategy:
    strategies = {"fedavg": FedAvg, "fedprox": FedProx}
    if name not in strategies:
        raise ValueError(f"unknown strategy {name!r}; expected one of {sorted(strategies)}")
    return strategies[name](**kwargs)


# ---------------------------------------------------------------------------
# Dynamic trust weighting (metadata-based, SecAgg-blind)
# ---------------------------------------------------------------------------


@dataclass
class ClientStats:
    """Behavioural statistics the coordinator keeps per client across rounds.

    **None** of these require seeing the actual model weights — SecAgg blindness is
    preserved.
    """

    client_id: str
    rounds_participated: int = 0  # rounds where the client delivered an update
    rounds_selected: int = 0  # rounds where the client was selected
    on_time_deliveries: int = 0  # updates delivered before the deadline
    total_num_samples: int = 0  # sum of reported num_samples over all rounds
    val_metrics: list[float] = field(default_factory=list)  # client-reported val loss history


def compute_dynamic_trust_weights(
    client_stats: list[ClientStats],
    max_num_samples: int | None = None,
) -> dict[str, float]:
    """Compute normalised trust weights α_i^t from metadata only.

    Signals used (each in [0, 1]):
    - **Participation rate** = rounds_participated / rounds_selected
    - **On-time ratio** = on_time_deliveries / rounds_participated
    - **Sample-count consistency** = min(1, total_samples / (rounds × max_num_samples))
      (penalises clients that report implausibly many samples)
    - **Validation stability** = 1 − normalised std-dev of recent val metrics
      (erratic metrics suggest poisoning or extreme non-IID)

    The raw score is the product of the signals; the final α_i^t is the softmax
    normalisation so that Σ α_i = 1.
    """
    if not client_stats:
        raise ValueError("need at least one client")

    raw_scores: dict[str, float] = {}
    for cs in client_stats:
        # Participation rate
        p_rate = cs.rounds_participated / max(cs.rounds_selected, 1)

        # On-time ratio
        ot_rate = cs.on_time_deliveries / max(cs.rounds_participated, 1)

        # Sample-count consistency (1.0 when within expected range)
        if max_num_samples and cs.rounds_participated > 0:
            expected = max_num_samples * cs.rounds_participated
            sc_rate = min(1.0, cs.total_num_samples / expected)
        else:
            sc_rate = 1.0

        # Validation stability (1.0 when perfectly stable)
        if len(cs.val_metrics) >= 2:
            mean = sum(cs.val_metrics) / len(cs.val_metrics)
            if mean != 0:
                std = (sum((v - mean) ** 2 for v in cs.val_metrics) / len(cs.val_metrics)) ** 0.5
                val_stab = max(0.0, 1.0 - std / abs(mean))
            else:
                val_stab = 1.0
        else:
            val_stab = 1.0  # not enough data to judge; assume good

        raw_scores[cs.client_id] = p_rate * ot_rate * sc_rate * val_stab

    # Softmax-style normalisation to Σ = 1.
    total = sum(raw_scores.values())
    if total <= 0:
        # Fallback: uniform weights.
        n = len(raw_scores)
        return {cid: 1.0 / n for cid in raw_scores}
    return {cid: score / total for cid, score in raw_scores.items()}
