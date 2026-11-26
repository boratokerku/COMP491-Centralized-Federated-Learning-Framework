"""Tests for Faz 4: ClientRuntime DP/SecAgg+ flow, peer share encryption, and HttpTransport."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import torch
from cryptography.exceptions import InvalidTag

from cfl.client.http import HttpTransport, RetryPolicy
from cfl.client.runtime import ClientRuntime
from cfl.coordinator.aggregator import Aggregator, global_model_key
from cfl.coordinator.config import RunConfig
from cfl.coordinator.coordinator import Coordinator, Step
from cfl.coordinator.intake import UpdateIntake
from cfl.coordinator.state_store import EventRecord, InMemoryStateStore, RunRecord, RunStatus
from cfl.core.contract import ModelContract, initial_weights
from cfl.core.secagg import (
    KeyPair,
    ShamirShare,
    decrypt_peer_shares,
    encrypt_peer_shares,
    generate_self_mask_seed,
    round_context,
    split_secret,
)
from cfl.core.serialization import bytes_to_state_dict, sha256, state_dict_to_bytes
from cfl.core.store import InMemoryModelStore
from cfl.core.strategy import FedAvg
from cfl.core.task import TaskSpec
from cfl.gateway.app import create_app
from cfl.gateway.clients import AuthError, ClientRecord, ClientStatus
from cfl.protocol.messages import (
    RevealedShare,
    RoundKeySubmit,
    ShamirSharePayload,
    UnmaskSharesRequest,
)
from cfl.simulation.transport import InMemoryTransport


class _TinyTask(TaskSpec):
    """Minimal TaskSpec for fast end-to-end ClientRuntime privacy & SecAgg+ tests."""

    name = "tabular"
    preprocessing_version = "1.0"

    def build_model(self) -> torch.nn.Module:
        return torch.nn.Linear(3, 2)

    def contract(self) -> ModelContract:
        return ModelContract.from_model(self.build_model(), "tiny-linear", "1.0")

    def build_manifest(self) -> dict:
        return {
            "dataset": {"name": "tiny"},
            "model": {"fingerprint": self.contract().fingerprint()},
        }

    def load_split(self, path: Path):
        return torch.ones(8, 3), torch.zeros(8, dtype=torch.long)

    def preprocess(self, rows):
        return rows

    def train(self, model, train_data, config, global_weights, *, seed: int) -> dict[str, float]:
        # Deterministic small weight step that depends on seed so each client differs.
        shift = 0.02 * ((seed % 7) + 1)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(shift)
        return {"train_loss": 0.42}

    def evaluate(self, model, eval_data) -> dict[str, float]:
        return {"loss": 0.35, "f1": 0.8}

    def held_out_path(self, data_dir: Path) -> Path:
        return data_dir / "test.csv"


class _ManualClock:
    def __init__(self):
        self.now = datetime(2026, 11, 26, 15, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class _StubRegistry:
    def __init__(self, client_ids: list[str]):
        now = datetime.now(UTC)
        self._tokens = {
            f"tok-{cid}": ClientRecord(
                client_id=cid,
                status=ClientStatus.REGISTERED,
                client_version="cfl-client/0.1.0",
                protocol_version="0.1",
                created_at=now,
                registered_at=now,
                last_seen_at=now,
            )
            for cid in client_ids
        }

    def authenticate(self, token: str) -> ClientRecord:
        rec = self._tokens.get(token)
        if rec is None:
            raise AuthError(401, "invalid_token", "unknown token")
        return rec


def _privacy_config(
    num_clients: int,
    min_participants: int,
    *,
    secagg_enabled: bool,
    base_noise_std: float = 0.01,
) -> RunConfig:
    return RunConfig.model_validate(
        {
            "run": {"name": "client-privacy-test", "seed": 5, "mode": "simulation"},
            "task": {"name": "tabular", "data_dir": "data/tabular", "num_clients": num_clients},
            "strategy": {"name": "fedprox", "mu": 0.01},
            "rounds": {
                "num_rounds": 1,
                "client_fraction": 1.0,
                "min_participants": min_participants,
                "round_deadline_s": 60,
                "max_num_samples": 100,
            },
            "client": {"local_epochs": 1, "batch_size": 8, "learning_rate": 0.01},
            "privacy": {
                "enabled": True,
                "clip_norm": 2.0,
                "base_noise_std": base_noise_std,
                "secagg_enabled": secagg_enabled,
                "min_recovery_ratio": 0.67,
            },
        }
    )


def _start_run(
    coord: Coordinator, state: InMemoryStateStore, task: _TinyTask, cfg: RunConfig, cids: list[str]
) -> str:
    run_id = str(uuid.uuid4())
    g0 = initial_weights(task.build_model, seed=cfg.run.seed)
    g0_bytes = state_dict_to_bytes(g0)
    g0_key = global_model_key(run_id, 0)
    coord.models.put(g0_key, g0_bytes)
    manifest_bytes = b'{"dataset": {"name": "tiny"}}'
    coord.models.put(f"{run_id}/manifest.json", manifest_bytes)
    now = coord.clock()
    contract = task.contract()
    state.create_run(
        RunRecord(
            run_id=run_id,
            name=cfg.run.name,
            status=RunStatus.RUNNING,
            strategy=cfg.strategy.name,
            task_name=cfg.task.name,
            seed=cfg.run.seed,
            config=cfg.model_dump(),
            config_sha256=cfg.fingerprint(),
            protocol_version="0.1",
            manifest_version="tiny@1.0.0",
            manifest_sha256=sha256(manifest_bytes),
            preprocessing_version="1.0",
            model_architecture=contract.architecture_id,
            model_architecture_version=contract.architecture_version,
            model_fingerprint=contract.fingerprint(),
            model_version=0,
            model_path=g0_key,
            model_sha256=sha256(g0_bytes),
            created_at=now,
            started_at=now,
        ),
        EventRecord("RUN_STARTED", now, run_id=run_id),
    )
    state.mark_eligible("tiny@1.0.0", cids)
    coord._aggregators[run_id] = Aggregator(
        strategy=FedAvg(),
        contract=contract,
        store=coord.models,
        build_model=task.build_model,
        sanity_inputs=torch.zeros(2, 3),
    )
    return run_id


def _make_runtime(tmp_path: Path, transport, task: _TinyTask, cid: str, seed: int) -> ClientRuntime:
    return ClientRuntime(
        transport=transport,
        task=task,
        client_id=cid,
        client_version="cfl-client/0.1.0",
        dataset="tiny",
        partition_dir=tmp_path / f"part_{cid}",
        state_dir=tmp_path / f"state_{cid}",
        poll_interval_s=0.0,
        seed=seed,
        log=lambda _: None,
        sleep=lambda _: None,
    )


# ========================== 1. Peer Share Encryption ==========================


class TestPeerShareEncryption:
    def test_encrypt_decrypt_roundtrip(self):
        kp_a, kp_b = KeyPair.generate(), KeyPair.generate()
        ctx = round_context("run-enc", 2)
        s_self = split_secret(generate_self_mask_seed(), 3, 2)[0]
        s_key = split_secret(generate_self_mask_seed(), 3, 2)[0]

        blob = encrypt_peer_shares(kp_a.private_key, kp_b.public_bytes, ctx, s_self, s_key)
        dec_self, dec_key = decrypt_peer_shares(kp_b.private_key, kp_a.public_bytes, ctx, blob)
        assert dec_self == s_self
        assert dec_key == s_key

    def test_wrong_context_or_wrong_recipient_fails(self):
        kp_a, kp_b, kp_eve = KeyPair.generate(), KeyPair.generate(), KeyPair.generate()
        ctx = round_context("run-enc", 2)
        s1, s2 = ShamirShare(1, 12345), ShamirShare(1, 67890)
        blob = encrypt_peer_shares(kp_a.private_key, kp_b.public_bytes, ctx, s1, s2)

        with pytest.raises(InvalidTag):
            decrypt_peer_shares(
                kp_b.private_key, kp_a.public_bytes, round_context("run-enc", 3), blob
            )
        with pytest.raises(InvalidTag):
            decrypt_peer_shares(kp_eve.private_key, kp_a.public_bytes, ctx, blob)


# ========================== 2. End-to-End ClientRuntime ==========================


class TestClientRuntimePrivacy:
    def test_dp_only_round_end_to_end(self, tmp_path):
        clock = _ManualClock()
        state, models = InMemoryStateStore(), InMemoryModelStore()
        coord = Coordinator(state, models, clock=clock)
        intake = UpdateIntake(state, models, clock=clock)
        task = _TinyTask()
        cids = ["c1", "c2", "c3"]
        cfg = _privacy_config(3, 3, secagg_enabled=False, base_noise_std=0.0)
        run_id = _start_run(coord, state, task, cfg, cids)

        transport = InMemoryTransport(coord, intake, state, models)
        clients = [
            _make_runtime(tmp_path, transport, task, cid, i + 1) for i, cid in enumerate(cids)
        ]

        assert coord.step(run_id) is Step.OPENED
        g0 = bytes_to_state_dict(models.get(global_model_key(run_id, 0)))

        for c in clients:
            assert c.step() is True
            assert [run_id, 1] in c.state.rounds_done

        assert coord.step(run_id) is Step.DONE
        g1 = bytes_to_state_dict(models.get(global_model_key(run_id, 1)))
        # Global model moved in the positive direction from clipped + trust-scaled deltas
        for k in g0:
            assert (g1[k] - g0[k]).mean().item() > 0.0
        for cid in cids:
            assert not models.exists(f"{run_id}/round_1/{cid}.safetensors")

    def test_secagg_four_clients_with_dropout_end_to_end(self, tmp_path):
        """4 real ClientRuntimes exchange keys + encrypted Shamir shares; c4 drops out before
        upload; the 3 survivors unmask via Coordinator and recover the exact aggregate."""
        clock = _ManualClock()
        state, models = InMemoryStateStore(), InMemoryModelStore()
        coord = Coordinator(state, models, clock=clock)
        intake = UpdateIntake(state, models, clock=clock)
        task = _TinyTask()
        cids = ["c1", "c2", "c3", "c4"]
        cfg = _privacy_config(4, 3, secagg_enabled=True, base_noise_std=0.01)
        run_id = _start_run(coord, state, task, cfg, cids)

        transport = InMemoryTransport(coord, intake, state, models)
        clients = {
            cid: _make_runtime(tmp_path, transport, task, cid, i + 1) for i, cid in enumerate(cids)
        }

        assert coord.step(run_id) is Step.OPENED
        g0 = bytes_to_state_dict(models.get(global_model_key(run_id, 0)))

        # Step 1: all 4 clients train locally and submit their X25519 public keys.
        for cid in cids:
            clients[cid].step()
        # Step 2: bring every client to "shares_submitted" (c4 already advanced when it posted
        # the 4th public key and saw ready=True).
        for cid in cids:
            if clients[cid]._secagg.stage == "keys_submitted":
                assert clients[cid].step() is True
            assert clients[cid]._secagg.stage == "shares_submitted"

        # Step 3: c1, c2, c3 decrypt shares, mask and upload; c4 drops out here!
        survivors = ["c1", "c2", "c3"]
        for cid in survivors:
            assert clients[cid].step() is True
            assert clients[cid]._secagg is not None
            assert clients[cid]._secagg.stage == "unmasking"

        # Before deadline: coordinator still COLLECTING, survivors wait in unmasking.
        assert coord.step(run_id) is Step.COLLECTING
        assert clients["c1"].step() is False

        # Past deadline: c4 marked DROPOUT, coordinator enters UNMASKING.
        clock.advance(61)
        assert coord.step(run_id) is Step.UNMASKING

        # Survivors poll unmask status, reveal self_mask for survivors & mask_key for c4.
        for cid in survivors:
            assert clients[cid].step() is True
            assert clients[cid]._secagg is None
            assert [run_id, 1] in clients[cid].state.rounds_done

        # Coordinator reconstructs secrets, unmasks aggregate, advances global model to v1.
        assert coord.step(run_id) is Step.DONE
        g1 = bytes_to_state_dict(models.get(global_model_key(run_id, 1)))
        for k in g0:
            assert torch.isfinite(g1[k]).all()
            assert not torch.equal(g1[k], g0[k])

        # Server cleaned up all masked uploads, ephemeral keys, and Shamir shares.
        for cid in cids:
            assert not models.exists(f"{run_id}/round_1/{cid}.safetensors")
            assert not models.exists(f"{run_id}/round_1/secagg/key_{cid}.json")
            assert not models.exists(f"{run_id}/round_1/secagg/unmask_{cid}.json")

    def test_secagg_upload_retry_reuses_same_update_id(self, tmp_path):
        """A transient upload failure during SecAgg must resend the exact same update_id."""
        clock = _ManualClock()
        state, models = InMemoryStateStore(), InMemoryModelStore()
        coord = Coordinator(state, models, clock=clock)
        intake = UpdateIntake(state, models, clock=clock)
        task = _TinyTask()
        cids = ["c1", "c2"]
        cfg = _privacy_config(2, 2, secagg_enabled=True, base_noise_std=0.0)
        run_id = _start_run(coord, state, task, cfg, cids)

        seen_update_ids: list[str] = []

        class _FlakyUploadTransport(InMemoryTransport):
            def upload_update(self, metadata, weights):
                seen_update_ids.append(metadata.update_id)
                if len(seen_update_ids) == 1:
                    from cfl.protocol.messages import UpdateAck

                    return UpdateAck(accepted=False, reason="temporary_busy")
                return super().upload_update(metadata, weights)

        t1 = _FlakyUploadTransport(coord, intake, state, models)
        t2 = InMemoryTransport(coord, intake, state, models)
        c1 = _make_runtime(tmp_path, t1, task, "c1", 1)
        c2 = _make_runtime(tmp_path, t2, task, "c2", 2)

        assert coord.step(run_id) is Step.OPENED
        c1.step()
        c2.step()
        if c1._secagg.stage == "keys_submitted":
            c1.step()
        assert c1._secagg.stage == "shares_submitted"
        assert c2._secagg.stage == "shares_submitted"

        # First upload attempt returns non-final rejection -> stays in "uploading"
        assert c1.step() is False
        assert c1._secagg.stage == "uploading"
        # Retry resends the exact same update_id and advances to "unmasking"
        assert c1.step() is True
        assert c1._secagg.stage == "unmasking"
        assert len(seen_update_ids) == 2
        assert seen_update_ids[0] == seen_update_ids[1]


# ========================== 3. HttpTransport SecAgg+ Methods ==========================


class TestHttpTransportSecAgg:
    def test_http_transport_secagg_with_retry(self):
        clock = _ManualClock()
        state, models = InMemoryStateStore(), InMemoryModelStore()
        cids = ["c1", "c2", "c3"]
        app = create_app(
            models=models,
            clock=clock,
            state_store=state,
            client_registry=_StubRegistry(cids),
        )
        coord = Coordinator(state, models, clock=clock)
        task = _TinyTask()
        cfg = _privacy_config(3, 2, secagg_enabled=True)
        run_id = _start_run(coord, state, task, cfg, cids)
        coord.step(run_id)

        # Wrap ASGI app with a one-shot 503 on first POST /rounds/1/keys to verify retry works.
        from fastapi.testclient import TestClient

        test_client = TestClient(app)
        attempts = {"keys_post": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST" and request.url.path == "/rounds/1/keys":
                attempts["keys_post"] += 1
                if attempts["keys_post"] == 1:
                    return httpx.Response(503)
            resp = test_client.request(
                request.method,
                str(request.url.raw_path.decode()),
                headers=dict(request.headers),
                content=request.content,
            )
            return httpx.Response(resp.status_code, headers=resp.headers, content=resp.content)

        def make_http(cid: str) -> HttpTransport:
            client = httpx.Client(
                base_url="https://gateway.test", transport=httpx.MockTransport(handler)
            )
            return HttpTransport(
                "https://gateway.test",
                f"tok-{cid}",
                client=client,
                retry=RetryPolicy(max_attempts=3),
                sleep=lambda _: None,
            )

        transports = {cid: make_http(cid) for cid in cids}
        kps = {cid: KeyPair.generate() for cid in cids}
        for cid in cids:
            resp = transports[cid].submit_round_keys(
                RoundKeySubmit(
                    run_id=run_id,
                    round_id=1,
                    client_id=cid,
                    public_key_hex=kps[cid].public_bytes.hex(),
                )
            )
            assert resp.secagg_threshold == 3
        assert attempts["keys_post"] == 4  # 1 transient 503 + 3 successes

        keys = transports["c1"].get_round_keys(run_id, 1, "c1")
        assert keys.ready is True
        assert len(keys.peers) == 3

        status = transports["c1"].get_unmask_status(run_id, 1, "c1")
        assert status.ready is False

        # Trigger unmasking state directly and verify submit_unmask_shares over HTTP.
        from cfl.coordinator.secagg_store import UnmaskState

        coord.secagg.save_unmask_state(
            run_id,
            1,
            UnmaskState(
                surviving=cids, dropped=[], unmask_deadline=clock() + timedelta(seconds=60)
            ),
        )
        status2 = transports["c1"].get_unmask_status(run_id, 1, "c1")
        assert status2.ready is True
        ack = transports["c1"].submit_unmask_shares(
            UnmaskSharesRequest(
                run_id=run_id,
                round_id=1,
                client_id="c1",
                shares=[
                    RevealedShare(
                        target_client_id="c2",
                        kind="self_mask",
                        share=ShamirSharePayload(x=1, y_hex="abcd"),
                    )
                ],
            )
        )
        assert ack.accepted is True
