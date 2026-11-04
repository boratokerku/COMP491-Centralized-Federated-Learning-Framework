"""Run configuration: one YAML file defines a run completely (validated here), including all
seeds, so any reported result can be rerun. See backend/configs/sim_*.yaml."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RunSection(_Strict):
    name: str = Field(min_length=1)
    seed: int
    mode: Literal["simulation", "distributed"]


class TaskSection(_Strict):
    name: Literal["tabular", "text"]
    data_dir: str  # holds partitions/<dataset>/test.csv, the server's held-out set
    num_clients: int = Field(ge=1)
    partition: Literal["iid"] = "iid"


class StrategySection(_Strict):
    name: Literal["fedavg", "fedprox"]
    mu: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _mu_only_for_fedprox(self) -> StrategySection:
        if self.name == "fedprox" and self.mu is None:
            raise ValueError("fedprox needs mu")
        if self.name == "fedavg" and self.mu is not None:
            raise ValueError("mu is only used by fedprox")
        return self


class RoundsSection(_Strict):
    num_rounds: int = Field(ge=1)
    client_fraction: float = Field(gt=0, le=1)
    min_participants: int = Field(ge=1)
    round_deadline_s: float = Field(gt=0)
    max_num_samples: int | None = Field(default=None, ge=1)  # per-client cap on reported samples


class ClientSection(_Strict):
    """Local training settings sent to every client in the round config."""

    local_epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    class_weights: list[float] | None = None  # fixed for the federation, never per client

    @model_validator(mode="after")
    def _two_classes(self) -> ClientSection:
        if self.class_weights is not None and (
            len(self.class_weights) != 2 or min(self.class_weights) <= 0
        ):
            raise ValueError("class_weights must be two positive numbers")
        return self


class RunConfig(_Strict):
    run: RunSection
    task: TaskSection
    strategy: StrategySection
    rounds: RoundsSection
    client: ClientSection

    @model_validator(mode="after")
    def _participants_fit(self) -> RunConfig:
        if self.rounds.min_participants > self.task.num_clients:
            raise ValueError("min_participants cannot exceed task.num_clients")
        return self

    def fingerprint(self) -> str:
        """SHA-256 of the canonical JSON, recorded with the run for reproducibility."""
        canonical = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_run_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        return RunConfig.model_validate(yaml.safe_load(f))
