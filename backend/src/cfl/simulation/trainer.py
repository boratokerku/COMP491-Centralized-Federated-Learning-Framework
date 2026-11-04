"""Reference local training for simulated clients.

The real client implementations (tabular and text) are owned by the client teams and implement
TaskSpec; this is the minimal behaviour they must match so the simulation can run end to end:
the round config keys from Coordinator.round_config, class-weighted cross-entropy, and the
FedProx proximal term when `proximal_mu` > 0.
"""

from __future__ import annotations

import torch

from cfl.core.contract import StateDict
from cfl.core.evaluation import classification_metrics
from cfl.core.strategy import proximal_term


def train_local(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    config: dict,
    global_weights: StateDict | None = None,
    seed: int = 0,
) -> dict[str, float]:
    mu = float(config.get("proximal_mu") or 0.0)
    if mu > 0 and global_weights is None:
        raise ValueError("FedProx needs the global weights")
    class_weights = config.get("class_weights")
    loss_fn = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    generator = torch.Generator().manual_seed(seed)

    model.train()
    total, batches = 0.0, 0
    with torch.random.fork_rng(devices=[]):  # dropout masks are reproducible per seed too
        torch.manual_seed(seed)
        for _ in range(config["local_epochs"]):
            order = torch.randperm(len(labels), generator=generator)
            for batch in torch.split(order, config["batch_size"]):
                optimizer.zero_grad()
                loss = loss_fn(model(inputs[batch]), labels[batch])
                if mu > 0:
                    loss = loss + proximal_term(model, global_weights, mu)
                loss.backward()
                optimizer.step()
                total += loss.item()
                batches += 1
    return {"train_loss": total / max(batches, 1)}


def evaluate(model: torch.nn.Module, inputs: torch.Tensor, labels: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        return classification_metrics(model(inputs), labels)
