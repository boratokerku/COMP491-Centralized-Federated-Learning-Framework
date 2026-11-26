"""Tests for Faz 3: Blind Aggregator, Coordinator SecAgg+ orchestration, and Gateway endpoints."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import torch
from fastapi.testclient import TestClient

from cfl.coordinator.aggregator import (
    Aggregator,
    FailureReason,
    Rejection,
    RoundInput,
    UpdateRecord,
    global_model_key,
)
from cfl.coordinator.config import RunConfig
from cfl.coordinator.coordinator import Coordinator, Step
from cfl.coordinator.intake import UpdateIntake
from cfl.coordinator.state import RoundState
from cfl.coordinator.state_store import InMemoryStateStore
from cfl.core.contract import ModelContract, initial_weights
from cfl.core.privacy import (
    DEFAULT_MODULUS,
    add_dp_noise,
    calibrate_noise_std,
    clip_delta,
    quantize_state_dict,
    scale_by_trust,
)
from cfl.core.secagg import (
    KeyPair,
    apply_masks,
    generate_self_mask_seed,
    private_key_to_bytes,
    round_context,
    split_secret,
)
from cfl.core.serialization import bytes_to_state_dict, sha256, state_dict_to_bytes
from cfl.core.store import InMemoryModelStore
from cfl.core.strategy import FedAvg
from cfl.gateway.app import create_app
from cfl.gateway.clients import AuthError, ClientRecord, ClientStatus
from cfl.protocol.messages import (
    RevealedShare,
    RoundKeySubmit,
    ShamirSharePayload,
    UnmaskSharesRequest,
    UpdateMetadata,
)


def _tiny_model() -> torch.nn.Module:
    return torch.nn.Linear(3, 2)


def _contract() -> ModelContract:
    return ModelContract.from_model(_tiny_model(), "tiny-linear", "1.0")


class _ManualClock:
    def __init__(self, start: datetime | None = None):
        self.now = start or datetime(2026, 11, 26, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class _StubRegistry:
    """In-memory bearer-token registry for testing Gateway SecAgg HTTP routes."""

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


def _secagg_config(
    num_clients: int = 4,
    min_participants: int = 3,
    num_rounds: int = 2,
    secagg_enabled: bool = True,
    base_noise_std: float = 0.0,
) -> RunConfig:
    return RunConfig.model_validate(
        {
            "run": {"name": "secagg-test", "seed": 7, "mode": "simulation"},
            "task": {"name": "tabular", "data_dir": "data/tabular", "num_clients": num_clients},
            "strategy": {"name": "fedprox", "mu": 0.01},
            "rounds": {
                "num_rounds": num_rounds,
                "client_fraction": 1.0,
                "min_participants": min_participants,
                "round_deadline_s": 60,
                "max_num_samples": 1000,
            },
            "client": {"local_epochs": 1, "batch_size": 32, "learning_rate": 0.01},
            "privacy": {
                "enabled": True,
                "clip_norm": 5.0,
                "base_noise_std": base_noise_std,
                "secagg_enabled": secagg_enabled,
                "min_recovery_ratio": 0.67,
            },
        }
    )


def _setup_aggregator_round(
    store: InMemoryModelStore,
    run_id: str = "run-agg",
    round_id: int = 1,
) -> tuple[Aggregator, dict[str, torch.Tensor], str, str]:
    contract = _contract()
    g0 = initial_weights(_tiny_model, seed=11)
    g0_bytes = state_dict_to_bytes(g0)
    g0_key = global_model_key(run_id, 0)
    store.put(g0_key, g0_bytes)
    agg = Aggregator(
        strategy=FedAvg(),
        contract=contract,
        store=store,
        build_model=_tiny_model,
        sanity_inputs=torch.zeros(2, 3),
    )
    return agg, g0, g0_key, sha256(g0_bytes)


# ========================== 1. Blind Aggregator ==========================


class TestBlindAggregator:
    def test_dp_delta_mode_aggregates_and_cleans_updates(self):
        store = InMemoryModelStore()
        agg, g0, g0_key, g0_sha = _setup_aggregator_round(store)

        d1 = {k: torch.full_like(v, 0.05) for k, v in g0.items()}
        d2 = {k: torch.full_like(v, -0.02) for k, v in g0.items()}
        records = []
        for idx, (cid, delta) in enumerate([("c1", d1), ("c2", d2)]):
            raw = state_dict_to_bytes(delta)
            key = f"run-agg/round_1/{cid}.safetensors"
            store.put(key, raw)
            records.append(UpdateRecord(f"u{idx}", cid, 0, 100, key, sha256(raw)))

        res = agg.aggregate(
            RoundInput(
                run_id="run-agg",
                round_id=1,
                input_version=0,
                input_key=g0_key,
                input_sha256=g0_sha,
                updates=records,
                min_participants=2,
                privacy_delta=True,
                cleanup_updates=True,
            )
        )
        assert res.state is RoundState.DONE
        out = bytes_to_state_dict(store.get(res.output_key))
        for k in g0:
            torch.testing.assert_close(out[k], g0[k] + d1[k] + d2[k], atol=1e-6, rtol=1e-6)
        # Temporary update files are removed immediately after aggregation.
        for rec in records:
            assert not store.exists(rec.key)

    def test_secagg_blind_summation_and_cleanup(self):
        store = InMemoryModelStore()
        agg, g0, g0_key, g0_sha = _setup_aggregator_round(store)

        ids = ["c1", "c2", "c3"]
        ctx = round_context("run-agg", 1)
        kps = {cid: KeyPair.generate() for cid in ids}
        pubs = {cid: kp.public_bytes for cid, kp in kps.items()}
        seeds = {cid: generate_self_mask_seed() for cid in ids}

        deltas = {
            "c1": {k: torch.full_like(v, 0.04) for k, v in g0.items()},
            "c2": {k: torch.full_like(v, -0.01) for k, v in g0.items()},
            "c3": {k: torch.full_like(v, 0.02) for k, v in g0.items()},
        }
        records = []
        for cid in ids:
            q = quantize_state_dict(deltas[cid])
            masked = apply_masks(q, cid, kps[cid], ids, pubs, ctx, seeds[cid])
            raw = state_dict_to_bytes(masked)
            key = f"run-agg/round_1/{cid}.safetensors"
            store.put(key, raw)
            records.append(UpdateRecord(f"u-{cid}", cid, 0, 50, key, sha256(raw)))

        res = agg.aggregate(
            RoundInput(
                run_id="run-agg",
                round_id=1,
                input_version=0,
                input_key=g0_key,
                input_sha256=g0_sha,
                updates=records,
                min_participants=3,
                privacy_delta=True,
                secagg_enabled=True,
                secagg_self_seeds=seeds,
                secagg_surviving_public_keys=pubs,
                cleanup_updates=True,
            )
        )
        assert res.state is RoundState.DONE
        out = bytes_to_state_dict(store.get(res.output_key))
        for k in g0:
            expected = g0[k] + deltas["c1"][k] + deltas["c2"][k] + deltas["c3"][k]
            torch.testing.assert_close(out[k], expected, atol=1e-5, rtol=1e-5)
        for rec in records:
            assert not store.exists(rec.key)

    def test_secagg_rejects_out_of_modulus_upload_and_fails_cleanly(self):
        store = InMemoryModelStore()
        agg, g0, g0_key, g0_sha = _setup_aggregator_round(store)

        good = {k: torch.zeros_like(v, dtype=torch.int64) for k, v in g0.items()}
        bad = {k: torch.full_like(v, DEFAULT_MODULUS + 5, dtype=torch.int64) for k, v in g0.items()}
        records = []
        for cid, tensor_dict in [("c1", good), ("c2", good), ("c3", bad)]:
            raw = state_dict_to_bytes(tensor_dict)
            key = f"run-agg/round_1/{cid}.safetensors"
            store.put(key, raw)
            records.append(UpdateRecord(f"u-{cid}", cid, 0, 10, key, sha256(raw)))

        res = agg.aggregate(
            RoundInput(
                run_id="run-agg",
                round_id=1,
                input_version=0,
                input_key=g0_key,
                input_sha256=g0_sha,
                updates=records,
                min_participants=2,
                privacy_delta=True,
                secagg_enabled=True,
                secagg_self_seeds={},
                cleanup_updates=True,
            )
        )
        assert res.state is RoundState.FAILED
        assert res.reason is FailureReason.AGGREGATION_ERROR
        assert res.rejected["u-c3"] is Rejection.SANITY_CHECK_FAILED
        for rec in records:
            assert not store.exists(rec.key)


# ========================== 2. Coordinator SecAgg+ Lifecycle ==========================


def _make_coordinator(
    clock: _ManualClock,
) -> tuple[Coordinator, InMemoryStateStore, InMemoryModelStore, UpdateIntake]:
    state = InMemoryStateStore()
    models = InMemoryModelStore()
    coord = Coordinator(state, models, clock=clock)
    intake = UpdateIntake(state, models, clock=clock)
    return coord, state, models, intake


def _patch_lightweight_aggregator(coord: Coordinator, run_id: str) -> None:
    """Avoid needing held-out CSV files on disk when testing coordinator state transitions."""
    task_contract = _contract()
    coord._aggregators[run_id] = Aggregator(
        strategy=FedAvg(),
        contract=task_contract,
        store=coord.models,
        build_model=_tiny_model,
        sanity_inputs=torch.zeros(2, 3),
    )


def _start_tiny_run(
    coord: Coordinator, state: InMemoryStateStore, config: RunConfig, clients: list[str]
) -> str:
    run_id = str(uuid.uuid4())
    # Bypass real tabular task model size: pin tiny model weights directly for fast unit tests.
    g0 = initial_weights(_tiny_model, seed=config.run.seed)
    g0_bytes = state_dict_to_bytes(g0)
    g0_key = global_model_key(run_id, 0)
    coord.models.put(g0_key, g0_bytes)
    manifest_bytes = b'{"dataset": {"name": "tiny"}, "model": {"fingerprint": "abc"}}'
    coord.models.put(f"{run_id}/manifest.json", manifest_bytes)

    from cfl.coordinator.state_store import EventRecord, RunRecord, RunStatus

    now = coord.clock()
    contract = _contract()
    run = RunRecord(
        run_id=run_id,
        name=config.run.name,
        status=RunStatus.RUNNING,
        strategy=config.strategy.name,
        task_name=config.task.name,
        seed=config.run.seed,
        config=config.model_dump(),
        config_sha256=config.fingerprint(),
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
    )
    state.create_run(run, EventRecord("RUN_STARTED", now, run_id=run_id))
    state.mark_eligible("tiny@1.0.0", clients)
    _patch_lightweight_aggregator(coord, run_id)
    return run_id


class TestCoordinatorSecAggLifecycle:
    def test_full_secagg_round_with_dropout_and_trust_update(self):
        clock = _ManualClock()
        coord, state, models, intake = _make_coordinator(clock)
        clients = ["c1", "c2", "c3", "c4"]
        cfg = _secagg_config(num_clients=4, min_participants=3, num_rounds=2, base_noise_std=0.04)
        run_id = _start_tiny_run(coord, state, cfg, clients)

        # Open round 1
        assert coord.step(run_id) is Step.OPENED

        # Every selected client gets calibrated sigma_t, k_t, clip_norm and uniform alpha on R1
        tasks = {cid: coord.task_for(cid, run_id) for cid in clients}
        assert all(t is not None for t in tasks.values())
        expected_sigma = calibrate_noise_std(0.04, 4)
        for t in tasks.values():
            assert t.clip_norm == 5.0
            assert t.sigma_t == pytest.approx(expected_sigma)
            assert t.trust_weight == pytest.approx(0.25)
            assert t.secagg_threshold == 3  # ceil(0.67 * 4) = 3

        # 1) Key exchange + Shamir secret splitting by all 4 clients
        kps = {cid: KeyPair.generate() for cid in clients}
        seeds = {cid: generate_self_mask_seed() for cid in clients}
        k_t = 3
        self_shares = {cid: split_secret(seeds[cid], 4, k_t) for cid in clients}
        key_shares = {
            cid: split_secret(private_key_to_bytes(kps[cid].private_key), 4, k_t) for cid in clients
        }

        for idx, cid in enumerate(clients):
            resp = coord.submit_round_keys(
                RoundKeySubmit(
                    run_id=run_id,
                    round_id=1,
                    client_id=cid,
                    public_key_hex=kps[cid].public_bytes.hex(),
                    encrypted_shares={
                        peer: f"enc-{cid}-to-{peer}" for peer in clients if peer != cid
                    },
                )
            )
            assert resp.ready is (idx == len(clients) - 1)

        keys_resp = coord.get_round_keys(run_id, 1, "c1")
        assert keys_resp.ready is True
        assert set(keys_resp.incoming_shares.keys()) == {"c2", "c3", "c4"}
        pubs = {p.client_id: p.public_key_bytes() for p in keys_resp.peers}

        # 2) c1, c2, c3 upload masked updates; c4 drops out
        g0 = bytes_to_state_dict(models.get(global_model_key(run_id, 0)))
        ctx = round_context(run_id, 1)
        survivors = ["c1", "c2", "c3"]
        expected_deltas = {}
        for i, cid in enumerate(survivors):
            local_w = {k: v + 0.05 * (i + 1) for k, v in g0.items()}
            delta = clip_delta(local_w, g0, tasks[cid].clip_norm)
            delta = add_dp_noise(delta, tasks[cid].sigma_t, torch.Generator().manual_seed(i + 10))
            weighted = scale_by_trust(delta, tasks[cid].trust_weight)
            expected_deltas[cid] = weighted
            masked = apply_masks(
                quantize_state_dict(weighted), cid, kps[cid], clients, pubs, ctx, seeds[cid]
            )
            raw = state_dict_to_bytes(masked)
            ack = intake.submit(
                UpdateMetadata(
                    run_id=run_id,
                    round_id=1,
                    client_id=cid,
                    update_id=str(uuid.uuid4()),
                    model_version=0,
                    num_samples=200,
                    checksum_sha256=sha256(raw),
                    metrics={"val_loss": 0.4},
                ),
                raw,
            )
            assert ack.accepted is True

        # Before deadline -> still COLLECTING
        assert coord.step(run_id) is Step.COLLECTING
        assert coord.get_unmask_status(run_id, 1, "c1").ready is False

        # Advance past deadline -> c4 marked DROPOUT, round enters UNMASKING
        clock.advance(61)
        assert coord.step(run_id) is Step.UNMASKING
        unmask_status = coord.get_unmask_status(run_id, 1, "c1")
        assert unmask_status.ready is True
        assert unmask_status.surviving == ["c1", "c2", "c3"]
        assert unmask_status.dropped == ["c4"]

        # 3) Surviving clients submit Shamir shares (self_mask for survivors, mask_key for c4)
        for share_idx, cid in enumerate(survivors):
            revealed = [
                RevealedShare(
                    target_client_id=s,
                    kind="self_mask",
                    share=ShamirSharePayload.from_share(self_shares[s][share_idx]),
                )
                for s in survivors
            ] + [
                RevealedShare(
                    target_client_id="c4",
                    kind="mask_key",
                    share=ShamirSharePayload.from_share(key_shares["c4"][share_idx]),
                )
            ]
            ack = coord.submit_unmask_shares(
                UnmaskSharesRequest(run_id=run_id, round_id=1, client_id=cid, shares=revealed)
            )
            assert ack.accepted is True

        # Coordinator completes blind aggregation (re-normalized over the 3 survivors: scale = 4/3)
        assert coord.step(run_id) is Step.DONE
        g1 = bytes_to_state_dict(models.get(global_model_key(run_id, 1)))
        alpha_sum = sum(tasks[c].trust_weight for c in survivors)
        for k in g0:
            expected_g1 = g0[k] + sum(expected_deltas[c][k] for c in survivors) / alpha_sum
            torch.testing.assert_close(g1[k], expected_g1, atol=1e-5, rtol=1e-5)

        # Ephemeral keys, shares, and masked uploads are all cleaned up
        for cid in clients:
            assert not models.exists(f"{run_id}/round_1/{cid}.safetensors")
            assert not models.exists(f"{run_id}/round_1/secagg/key_{cid}.json")
            assert not models.exists(f"{run_id}/round_1/secagg/unmask_{cid}.json")

        # 4) Round 2: dynamic trust weights downweight c4 (who dropped out in round 1)
        assert coord.step(run_id) is Step.OPENED
        t_c1 = coord.task_for("c1", run_id)
        t_c4 = coord.task_for("c4", run_id)
        assert t_c1 is not None and t_c4 is not None
        assert t_c1.trust_weight > t_c4.trust_weight
        assert 0.0 < t_c4.trust_weight < 1.0

    def test_unmasking_timeout_fails_round_safely(self):
        clock = _ManualClock()
        coord, state, models, intake = _make_coordinator(clock)
        clients = ["c1", "c2", "c3"]
        cfg = _secagg_config(num_clients=3, min_participants=2, num_rounds=1)
        run_id = _start_tiny_run(coord, state, cfg, clients)

        assert coord.step(run_id) is Step.OPENED
        kps = {cid: KeyPair.generate() for cid in clients}
        for cid in clients:
            coord.submit_round_keys(
                RoundKeySubmit(
                    run_id=run_id,
                    round_id=1,
                    client_id=cid,
                    public_key_hex=kps[cid].public_bytes.hex(),
                )
            )
        g0 = bytes_to_state_dict(models.get(global_model_key(run_id, 0)))
        zeros = state_dict_to_bytes(
            {k: torch.zeros_like(v, dtype=torch.int64) for k, v in g0.items()}
        )
        for cid in clients:
            intake.submit(
                UpdateMetadata(
                    run_id=run_id,
                    round_id=1,
                    client_id=cid,
                    update_id=str(uuid.uuid4()),
                    model_version=0,
                    num_samples=10,
                    checksum_sha256=sha256(zeros),
                ),
                zeros,
            )

        assert coord.step(run_id) is Step.UNMASKING
        # Nobody submits unmask shares before the unmask deadline -> fails safely
        clock.advance(61)
        assert coord.step(run_id) is Step.FAILED
        round_ = state.get_round(run_id, 1)
        assert round_.failure_reason == "unmasking_quorum_not_met"
        for cid in clients:
            assert not models.exists(f"{run_id}/round_1/{cid}.safetensors")


# ========================== 3. Gateway SecAgg+ HTTP Endpoints ==========================


class TestGatewaySecAggEndpoints:
    def test_keys_and_unmask_http_routes(self):
        clock = _ManualClock()
        state = InMemoryStateStore()
        models = InMemoryModelStore()
        clients = ["c1", "c2", "c3"]
        registry = _StubRegistry(clients + ["outsider"])
        app = create_app(
            models=models,
            clock=clock,
            state_store=state,
            client_registry=registry,
        )
        coord = Coordinator(state, models, clock=clock)
        intake = UpdateIntake(state, models, clock=clock)
        cfg = _secagg_config(num_clients=3, min_participants=2, num_rounds=1)
        run_id = _start_tiny_run(coord, state, cfg, clients)
        coord.step(run_id)

        http = TestClient(app)
        hdr = lambda cid: {"Authorization": f"Bearer tok-{cid}"}  # noqa: E731

        # GET /task includes privacy parameters
        task_resp = http.get("/task", headers=hdr("c1"))
        assert task_resp.status_code == 200
        task_body = task_resp.json()
        assert task_body["clip_norm"] == 5.0
        assert task_body["secagg_threshold"] == 3
        assert task_body["trust_weight"] == pytest.approx(1 / 3)

        # POST /rounds/1/keys by all 3 selected clients
        kps = {cid: KeyPair.generate() for cid in clients}
        for cid in clients:
            r = http.post(
                "/rounds/1/keys",
                headers=hdr(cid),
                json={
                    "run_id": run_id,
                    "round_id": 1,
                    "client_id": cid,
                    "public_key_hex": kps[cid].public_bytes.hex(),
                    "encrypted_shares": {p: f"blob-{cid}-{p}" for p in clients if p != cid},
                },
            )
            assert r.status_code == 200

        # Outsider gets 403
        r_out = http.get(f"/rounds/1/keys?run_id={run_id}", headers=hdr("outsider"))
        assert r_out.status_code == 403

        # Selected client gets ready=True + incoming shares
        r_keys = http.get(f"/rounds/1/keys?run_id={run_id}", headers=hdr("c2"))
        assert r_keys.status_code == 200
        body = r_keys.json()
        assert body["ready"] is True
        assert len(body["peers"]) == 3
        assert body["incoming_shares"] == {"c1": "blob-c1-c2", "c3": "blob-c3-c2"}

        # Before unmasking starts: GET /rounds/1/unmask -> ready=False, POST -> 409
        r_unmask_status = http.get(f"/rounds/1/unmask?run_id={run_id}", headers=hdr("c1"))
        assert r_unmask_status.status_code == 200
        assert r_unmask_status.json()["ready"] is False

        r_early = http.post(
            "/rounds/1/unmask",
            headers=hdr("c1"),
            json={"run_id": run_id, "round_id": 1, "client_id": "c1", "shares": []},
        )
        assert r_early.status_code == 409

        # Upload masked tensors for c1, c2, c3 and enter UNMASKING
        g0 = bytes_to_state_dict(models.get(global_model_key(run_id, 0)))
        zeros = state_dict_to_bytes(
            {k: torch.zeros_like(v, dtype=torch.int64) for k, v in g0.items()}
        )
        for cid in clients:
            intake.submit(
                UpdateMetadata(
                    run_id=run_id,
                    round_id=1,
                    client_id=cid,
                    update_id=str(uuid.uuid4()),
                    model_version=0,
                    num_samples=10,
                    checksum_sha256=sha256(zeros),
                ),
                zeros,
            )
        assert coord.step(run_id) is Step.UNMASKING

        r_status = http.get(f"/rounds/1/unmask?run_id={run_id}", headers=hdr("c1"))
        assert r_status.status_code == 200
        assert r_status.json()["ready"] is True
        assert r_status.json()["surviving"] == clients

        # Wrong share kind (mask_key for a survivor) rejected with 409
        r_wrong = http.post(
            "/rounds/1/unmask",
            headers=hdr("c1"),
            json={
                "run_id": run_id,
                "round_id": 1,
                "client_id": "c1",
                "shares": [
                    {
                        "target_client_id": "c2",
                        "kind": "mask_key",
                        "share": {"x": 1, "y_hex": "1234"},
                    }
                ],
            },
        )
        assert r_wrong.status_code == 409

        # Valid self_mask share accepted with 200
        r_ok = http.post(
            "/rounds/1/unmask",
            headers=hdr("c1"),
            json={
                "run_id": run_id,
                "round_id": 1,
                "client_id": "c1",
                "shares": [
                    {
                        "target_client_id": "c2",
                        "kind": "self_mask",
                        "share": {"x": 1, "y_hex": "1234"},
                    }
                ],
            },
        )
        assert r_ok.status_code == 200
        assert r_ok.json()["accepted"] is True
