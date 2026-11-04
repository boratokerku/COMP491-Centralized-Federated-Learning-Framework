"""Run a complete federated training in one process: coordinator, update intake, in-memory
transport and one simulated client per partition, all driven by the run config."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from cfl.coordinator.config import RunConfig
from cfl.coordinator.coordinator import REPO_ROOT, Coordinator, Step
from cfl.coordinator.intake import UpdateIntake
from cfl.coordinator.state_store import InMemoryStateStore, RoundRecord, RunRecord, StateStore
from cfl.core.store import InMemoryModelStore, ModelStore
from cfl.db.state_store import PostgresStateStore
from cfl.simulation.client import SimulatedClient
from cfl.simulation.transport import InMemoryTransport
from cfl.tasks.registry import get_task


@dataclass
class SimulationResult:
    run: RunRecord
    state: StateStore
    models: ModelStore
    verified_by: list[str]  # clients that downloaded and verified the final model


def run_simulation(
    config: RunConfig,
    repo_root: Path = REPO_ROOT,
    models: ModelStore | None = None,
    state: InMemoryStateStore | PostgresStateStore | None = None,
    on_round_closed: Callable[[RoundRecord, StateStore], None] | None = None,
    max_steps: int = 100_000,
) -> SimulationResult:
    if config.run.mode != "simulation":
        raise ValueError("run_simulation needs run.mode = simulation")
    state = state if state is not None else InMemoryStateStore()
    models = models if models is not None else InMemoryModelStore()
    coordinator = Coordinator(state, models, repo_root=repo_root)
    intake = UpdateIntake(state, models, coordinator.clock)
    transport = InMemoryTransport(coordinator, intake, state, models)

    run = coordinator.start_run(config)
    task = get_task(config.task.name)
    partitions = repo_root / config.task.data_dir / "partitions" / task.dataset
    clients = [
        SimulatedClient(
            f"client_{i}", transport, task, partitions / f"client_{i}", seed=config.run.seed + i
        )
        for i in range(config.task.num_clients)
    ]
    for client in clients:
        client.register()

    for _ in range(max_steps):
        step = coordinator.step(run.run_id)
        if step in (Step.DONE, Step.FAILED) and on_round_closed is not None:
            on_round_closed(state.latest_round(run.run_id), state)
        if step in (Step.COMPLETED, Step.IDLE):
            break
        for client in clients:
            client.tick()
    else:
        raise RuntimeError(f"run did not finish within {max_steps} steps")
    verified_by = [c.client_id for c in clients if c.fetch_final_model(run.run_id) is not None]
    return SimulationResult(state.get_run(run.run_id), state, models, verified_by)
