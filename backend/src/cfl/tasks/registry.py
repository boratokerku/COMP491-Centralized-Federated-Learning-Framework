"""Server-side view of each task: what the coordinator needs to start and run it.

Clients implement training and evaluation (TaskSpec); the coordinator only needs to build the
reference model, know its contract and manifest, and find the server's held-out test set.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from cfl.core.contract import ModelContract

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"


@dataclass(frozen=True)
class TaskDefinition:
    name: str
    dataset: str
    build_model: Callable[[], torch.nn.Module]
    contract: Callable[[], ModelContract]
    build_manifest: Callable[[], dict]
    preprocessing_version: str
    # (held-out test.csv path) -> (inputs, labels) for server-side evaluation
    load_split: Callable[[Path], tuple[torch.Tensor, torch.Tensor]]

    def held_out_path(self, data_dir: Path) -> Path:
        return data_dir / "partitions" / self.dataset / "test.csv"


def _tabular() -> TaskDefinition:
    from cfl.tasks.tabular import model
    from cfl.tasks.tabular.data import load_split
    from cfl.tasks.tabular.manifest import build_manifest
    from cfl.tasks.tabular.preprocessing import load_dataset_config

    config = load_dataset_config(CONFIGS_DIR / "datasets" / "bank_marketing.yaml")
    return TaskDefinition(
        name="tabular",
        dataset=config.name,
        build_model=model.BankMLP,
        contract=model.contract,
        build_manifest=lambda: build_manifest(config),
        preprocessing_version=config.preprocessing_version,
        load_split=lambda path: load_split(path, config),
    )


def _text() -> TaskDefinition:
    from cfl.tasks.text import model, tokenizer
    from cfl.tasks.text.data import DATASET_NAME, load_split
    from cfl.tasks.text.manifest import build_manifest

    return TaskDefinition(
        name="text",
        dataset=DATASET_NAME,
        build_model=model.SmsHashEmbed,
        contract=model.contract,
        build_manifest=build_manifest,
        preprocessing_version=tokenizer.TOKENIZER_VERSION,
        load_split=load_split,
    )


_TASKS: dict[str, Callable[[], TaskDefinition]] = {"tabular": _tabular, "text": _text}


def get_task(name: str) -> TaskDefinition:
    if name not in _TASKS:
        raise ValueError(f"unknown task {name!r}; expected one of {sorted(_TASKS)}")
    return _TASKS[name]()
