"""A simulated client site: polls for a task, trains on its own partition, uploads the update.

It only talks to the Transport, so the same loop works against the in-memory transport here
and against the gateway over HTTPS.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from cfl.core.serialization import bytes_to_state_dict, sha256, state_dict_to_bytes
from cfl.core.transport import Transport
from cfl.protocol.messages import RegisterRequest, UpdateAck, UpdateMetadata
from cfl.simulation.trainer import evaluate, train_local
from cfl.tasks.registry import TaskDefinition

CLIENT_VERSION = "cfl-sim-client/0.1.0"


class SimulatedClient:
    def __init__(
        self,
        client_id: str,
        transport: Transport,
        task: TaskDefinition,
        partition_dir: Path,
        seed: int,
    ):
        self.client_id = client_id
        self.transport = transport
        self.task = task
        self.seed = seed
        self.train_data = task.load_split(partition_dir / "train.csv")
        self.val_data = task.load_split(partition_dir / "val.csv")
        self._finished: set[tuple[str, int]] = set()

    def register(self) -> bool:
        response = self.transport.register(
            RegisterRequest(client_id=self.client_id, client_version=CLIENT_VERSION)
        )
        return response.accepted

    def tick(self) -> UpdateAck | None:
        """One polling cycle. Returns the upload acknowledgement, or None if there was no task."""
        task = self.transport.poll_task(self.client_id)
        if task is None or (task.run_id, task.round_id) in self._finished:
            return None

        global_weights = bytes_to_state_dict(
            self.transport.download_model(task.run_id, task.model_version)
        )
        model = self.task.build_model()
        model.load_state_dict(global_weights)

        before = evaluate(model, *self.val_data)  # federated evaluation of the received model
        start = time.perf_counter()
        stats = train_local(
            model,
            *self.train_data,
            task.config,
            global_weights=global_weights,
            seed=self.seed * 1000 + task.round_id,
        )
        train_s = time.perf_counter() - start
        after = evaluate(model, *self.val_data)

        weights = state_dict_to_bytes(model.state_dict())
        metrics = {**stats, "train_s": train_s}
        metrics |= {f"val_{k}_before": v for k, v in before.items()}
        metrics |= {f"val_{k}_after": v for k, v in after.items()}
        ack = self.transport.upload_update(
            UpdateMetadata(
                run_id=task.run_id,
                round_id=task.round_id,
                client_id=self.client_id,
                update_id=str(uuid.uuid4()),
                model_version=task.model_version,
                num_samples=len(self.train_data[1]),
                checksum_sha256=sha256(weights),
                metrics=metrics,
            ),
            weights,
        )
        self._finished.add((task.run_id, task.round_id))
        return ack
