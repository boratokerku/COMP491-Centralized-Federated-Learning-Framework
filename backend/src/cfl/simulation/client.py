"""A simulated client site: polls for a task, trains on its own partition, uploads the update.

It only talks to the Transport, so the same loop works against the in-memory transport here
and against the gateway over HTTPS.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import torch

from cfl.core.artifact import verify_artifact
from cfl.core.privacy import add_dp_noise, clip_delta, quantize_state_dict, scale_by_trust
from cfl.core.secagg import (
    KeyPair,
    ShamirShare,
    ShareKind,
    apply_masks,
    decrypt_peer_shares,
    encrypt_peer_shares,
    generate_self_mask_seed,
    private_key_to_bytes,
    round_context,
    share_to_reveal,
    split_secret,
)
from cfl.core.serialization import bytes_to_state_dict, sha256, state_dict_to_bytes
from cfl.core.transport import Transport
from cfl.protocol.messages import (
    RegisterRequest,
    RevealedShare,
    RoundKeySubmit,
    ShamirSharePayload,
    TaskAssignment,
    UnmaskSharesRequest,
    UpdateAck,
    UpdateMetadata,
)
from cfl.simulation.trainer import evaluate, train_local
from cfl.tasks.registry import TaskDefinition

CLIENT_VERSION = "cfl-sim-client/0.1.0"


@dataclass
class _SimSecAggSession:
    task: TaskAssignment
    metrics: dict[str, float]
    quantized: dict[str, torch.Tensor]
    keypair: KeyPair
    self_seed: bytes
    stage: str = "keys_submitted"
    peer_ids: list[str] = field(default_factory=list)
    peer_pks: dict[str, bytes] = field(default_factory=dict)
    peer_shares: dict[str, tuple[ShamirShare, ShamirShare]] = field(default_factory=dict)


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
        self._secagg: _SimSecAggSession | None = None

    def register(self) -> bool:
        response = self.transport.register(
            RegisterRequest(client_id=self.client_id, client_version=CLIENT_VERSION)
        )
        return response.accepted

    def _apply_privacy(
        self,
        task: TaskAssignment,
        local_weights: dict[str, torch.Tensor],
        global_weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        privacy_active = any(
            v is not None
            for v in (task.clip_norm, task.sigma_t, task.trust_weight, task.secagg_threshold)
        )
        if not privacy_active:
            return local_weights
        if task.clip_norm is not None:
            delta = clip_delta(local_weights, global_weights, task.clip_norm)
        else:
            delta = {k: local_weights[k] - global_weights[k] for k in local_weights}
        if task.sigma_t is not None and task.sigma_t > 0:
            rng = torch.Generator().manual_seed(self.seed * 1000 + task.round_id + 17)
            delta = add_dp_noise(delta, task.sigma_t, rng)
        if task.trust_weight is not None:
            delta = scale_by_trust(delta, task.trust_weight)
        return delta

    def _advance_secagg(self) -> UpdateAck | None:
        s = self._secagg
        if s is None:
            return None
        ctx = round_context(s.task.run_id, s.task.round_id)
        try:
            if s.stage == "keys_submitted":
                resp = self.transport.get_round_keys(s.task.run_id, s.task.round_id, self.client_id)
                if not resp.ready:
                    return None
                s.peer_ids = sorted(p.client_id for p in resp.peers)
                s.peer_pks = {p.client_id: p.public_key_bytes() for p in resp.peers}
                n, k = len(s.peer_ids), resp.secagg_threshold
                self_shares = split_secret(s.self_seed, n, k)
                key_shares = split_secret(private_key_to_bytes(s.keypair.private_key), n, k)
                encrypted: dict[str, str] = {}
                for idx, peer_id in enumerate(s.peer_ids):
                    if peer_id == self.client_id:
                        s.peer_shares[self.client_id] = (self_shares[idx], key_shares[idx])
                    else:
                        encrypted[peer_id] = encrypt_peer_shares(
                            s.keypair.private_key,
                            s.peer_pks[peer_id],
                            ctx,
                            self_shares[idx],
                            key_shares[idx],
                        )
                self.transport.submit_round_keys(
                    RoundKeySubmit(
                        run_id=s.task.run_id,
                        round_id=s.task.round_id,
                        client_id=self.client_id,
                        public_key_hex=s.keypair.public_bytes.hex(),
                        encrypted_shares=encrypted,
                    )
                )
                s.stage = "shares_submitted"
                return None

            if s.stage == "shares_submitted":
                resp = self.transport.get_round_keys(s.task.run_id, s.task.round_id, self.client_id)
                if len(resp.incoming_shares) < len(s.peer_ids) - 1:
                    return None
                for sender_id, blob_hex in resp.incoming_shares.items():
                    s.peer_shares[sender_id] = decrypt_peer_shares(
                        s.keypair.private_key, s.peer_pks[sender_id], ctx, blob_hex
                    )
                masked = apply_masks(
                    s.quantized,
                    self.client_id,
                    s.keypair,
                    s.peer_ids,
                    s.peer_pks,
                    ctx,
                    s.self_seed,
                )
                weights = state_dict_to_bytes(masked)
                ack = self.transport.upload_update(
                    UpdateMetadata(
                        run_id=s.task.run_id,
                        round_id=s.task.round_id,
                        client_id=self.client_id,
                        update_id=str(uuid.uuid4()),
                        model_version=s.task.model_version,
                        num_samples=len(self.train_data[1]),
                        checksum_sha256=sha256(weights),
                        metrics=s.metrics,
                    ),
                    weights,
                )
                if not ack.accepted:
                    self._finished.add((s.task.run_id, s.task.round_id))
                    self._secagg = None
                    return ack
                s.stage = "unmasking"
                return ack

            if s.stage == "unmasking":
                status = self.transport.get_unmask_status(
                    s.task.run_id, s.task.round_id, self.client_id
                )
                if not status.ready:
                    return None
                surviving_set, dropped_set = set(status.surviving), set(status.dropped)
                revealed = [
                    RevealedShare(
                        target_client_id=pid,
                        kind=(kind := share_to_reveal(pid, surviving_set, dropped_set)).value,
                        share=ShamirSharePayload.from_share(
                            s.peer_shares[pid][0]
                            if kind is ShareKind.SELF_MASK
                            else s.peer_shares[pid][1]
                        ),
                    )
                    for pid in sorted(s.peer_shares)
                ]
                self.transport.submit_unmask_shares(
                    UnmaskSharesRequest(
                        run_id=s.task.run_id,
                        round_id=s.task.round_id,
                        client_id=self.client_id,
                        shares=revealed,
                    )
                )
                self._finished.add((s.task.run_id, s.task.round_id))
                self._secagg = None
                return None
        except (ValueError, KeyError):
            self._finished.add((s.task.run_id, s.task.round_id))
            self._secagg = None
        return None

    def tick(self) -> UpdateAck | None:
        """One polling cycle. Returns the upload acknowledgement, or None if there was no task."""
        if self._secagg is not None:
            return self._advance_secagg()

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

        metrics = {**stats, "train_s": train_s}
        metrics |= {f"val_{k}_before": v for k, v in before.items()}
        metrics |= {f"val_{k}_after": v for k, v in after.items()}
        payload = self._apply_privacy(task, model.state_dict(), global_weights)

        if task.secagg_threshold is not None:
            kp = KeyPair.generate()
            self_seed = generate_self_mask_seed()
            self.transport.submit_round_keys(
                RoundKeySubmit(
                    run_id=task.run_id,
                    round_id=task.round_id,
                    client_id=self.client_id,
                    public_key_hex=kp.public_bytes.hex(),
                )
            )
            self._secagg = _SimSecAggSession(
                task=task,
                metrics=metrics,
                quantized=quantize_state_dict(payload),
                keypair=kp,
                self_seed=self_seed,
            )
            return self._advance_secagg()

        weights = state_dict_to_bytes(payload)
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

    def fetch_final_model(self, run_id: str) -> dict | None:
        """Download the final artifact and verify it locally (checksums, then the reference
        predictions with ONNX Runtime). Returns its descriptor, or None if not ready yet."""
        downloaded = self.transport.download_final_artifact(run_id)
        if downloaded is None:
            return None
        files, onnx_sha256 = downloaded
        return verify_artifact(files, onnx_sha256)
