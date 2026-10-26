"""Aggregation strategies (FedAvg, FedProx)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

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
