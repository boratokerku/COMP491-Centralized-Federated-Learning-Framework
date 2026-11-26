"""Tests for Faz 6: poisoning resilience (gradient scaling attack) and privacy reporting."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import torch

from cfl.coordinator.state import RoundState
from cfl.coordinator.state_store import (
    EventRecord,
    ParticipantRecord,
    ParticipantStatus,
    RoundRecord,
    RunRecord,
    RunStatus,
)
from cfl.core.privacy import (
    DEFAULT_MODULUS,
    add_dp_noise,
    calibrate_noise_std,
    clip_delta,
    dequantize_state_dict,
    quantize_state_dict,
    scale_by_trust,
)
from cfl.core.secagg import (
    KeyPair,
    apply_masks,
    generate_self_mask_seed,
    round_context,
    sum_masked,
    unmask_aggregate,
)
from cfl.core.strategy import (
    ClientStats,
    ClientUpdate,
    FedAvg,
    compute_dynamic_trust_weights,
    delta_sum,
)
from cfl.reporting.data import RunData
from cfl.reporting.markdown import render_markdown
from cfl.reporting.summary import build_summary


class TestPoisoningResilience:
    """Threat 2: a malicious client sends a 1000x scaled gradient update to poison the model."""

    def test_clipping_and_trust_weights_bound_gradient_scaling_attack(self):
        torch.manual_seed(0)
        g = {"w": torch.zeros(32)}

        # 3 honest clients with small updates (~0.05 norm); 1 attacker scaling by 1000x.
        honest = [{"w": torch.randn(32) * 0.01} for _ in range(3)]
        attacker = {"w": torch.full((32,), 1000.0)}  # extreme poisoning vector

        # Baseline (unprotected FedAvg): global model explodes.
        fedavg_out = FedAvg().aggregate(
            g,
            [ClientUpdate(f"h{i}", w, 100) for i, w in enumerate(honest)]
            + [ClientUpdate("attacker", attacker, 100)],
        )
        assert torch.linalg.vector_norm(fedavg_out["w"]).item() > 1000.0

        # Protected (Clip C=1.0 + Dynamic Trust α_i + SecAgg+):
        # Attacker has erratic validation loss and dropped previously -> low trust weight α.
        trust = compute_dynamic_trust_weights(
            [
                ClientStats("h0", 5, 5, 5, 500, [0.40, 0.38, 0.36]),
                ClientStats("h1", 5, 5, 5, 500, [0.41, 0.39, 0.37]),
                ClientStats("h2", 5, 5, 5, 500, [0.39, 0.37, 0.35]),
                ClientStats("attacker", 2, 5, 1, 200, [0.2, 5.0, 0.1, 8.0]),
            ],
            max_num_samples=200,
        )
        assert trust["attacker"] < 0.05 * trust["h0"]

        ids = ["h0", "h1", "h2", "attacker"]
        all_weights = {"h0": honest[0], "h1": honest[1], "h2": honest[2], "attacker": attacker}
        clip_norm = 1.0
        sigma_t = calibrate_noise_std(0.002, len(ids))

        ctx = round_context("run-poison", 1)
        kps = {cid: KeyPair.generate() for cid in ids}
        pubs = {cid: kp.public_bytes for cid, kp in kps.items()}
        seeds = {cid: generate_self_mask_seed() for cid in ids}

        masked_uploads = []
        for idx, cid in enumerate(ids):
            d = clip_delta(all_weights[cid], g, clip_norm)
            d = add_dp_noise(d, sigma_t, torch.Generator().manual_seed(idx + 50))
            d = scale_by_trust(d, trust[cid])
            q = quantize_state_dict(d)
            masked_uploads.append(apply_masks(q, cid, kps[cid], ids, pubs, ctx, seeds[cid]))

        agg = sum_masked(masked_uploads, DEFAULT_MODULUS)
        unmasked = unmask_aggregate(agg, ctx, surviving_self_seeds=seeds)
        protected_out = delta_sum(g, [dequantize_state_dict(unmasked)])

        # Deviation stays tightly bounded (< 0.1) instead of exploding to > 1000.
        protected_norm = torch.linalg.vector_norm(protected_out["w"]).item()
        assert protected_norm < 0.1

        # Furthermore, the attacker's contribution is tiny compared to honest clients.
        honest_avg = sum(trust[f"h{i}"] * honest[i]["w"] for i in range(3))
        diff_from_honest = torch.linalg.vector_norm(protected_out["w"] - honest_avg).item()
        assert diff_from_honest < 0.05


class TestPrivacyReporting:
    def test_summary_and_markdown_include_active_privacy_parameters(self):
        now = datetime(2026, 11, 26, 18, 0, tzinfo=UTC)
        sha = "a" * 64
        run = RunRecord(
            run_id="run-priv-1",
            name="privacy-demo",
            status=RunStatus.COMPLETED,
            strategy="fedprox",
            task_name="tabular",
            seed=42,
            config={
                "strategy": {"name": "fedprox", "mu": 0.01},
                "privacy": {
                    "enabled": True,
                    "clip_norm": 2.0,
                    "base_noise_std": 0.01,
                    "secagg_enabled": True,
                    "min_recovery_ratio": 0.67,
                },
            },
            config_sha256=sha,
            protocol_version="0.1",
            manifest_version="bank_marketing@1.0.0",
            manifest_sha256=sha,
            preprocessing_version="1.0",
            model_architecture="mlp",
            model_architecture_version="1.0",
            model_fingerprint=sha,
            model_version=2,
            model_path="run-priv-1/global_model_v2.safetensors",
            model_sha256=sha,
            created_at=now,
            started_at=now,
            completed_at=now,
        )
        rounds = [
            RoundRecord(
                "run-priv-1",
                r,
                RoundState.DONE,
                r - 1,
                f"m{r - 1}",
                sha,
                3,
                now,
                now,
                output_model_version=r,
                output_model_path=f"m{r}",
                output_model_sha256=sha,
                closed_at=now,
            )
            for r in (1, 2)
        ]
        events = [
            EventRecord(
                "ROUND_OPENED",
                now,
                "run-priv-1",
                1,
                payload={
                    "privacy": {
                        "clip_norm": 2.0,
                        "sigma_t": 0.005,
                        "secagg_enabled": True,
                        "secagg_threshold": 3,
                    }
                },
            ),
            EventRecord(
                "ROUND_OPENED",
                now,
                "run-priv-1",
                2,
                payload={
                    "privacy": {
                        "clip_norm": 2.0,
                        "sigma_t": 0.007,
                        "secagg_enabled": True,
                        "secagg_threshold": 2,
                    }
                },
            ),
        ]
        data = RunData(
            run=run,
            rounds=rounds,
            participants={
                1: [
                    ParticipantRecord("run-priv-1", 1, "c1", ParticipantStatus.UPDATE_ACCEPTED, now)
                ],
                2: [
                    ParticipantRecord("run-priv-1", 2, "c1", ParticipantStatus.UPDATE_ACCEPTED, now)
                ],
            },
            updates={1: [], 2: []},
            events=events,
            metrics=[],
        )

        summary = build_summary(data)
        priv = summary["privacy"]
        assert priv["enabled"] is True
        assert priv["secagg_enabled"] is True
        assert priv["clip_norm"] == 2.0
        assert priv["base_noise_std"] == 0.01
        assert priv["min_recovery_ratio"] == 0.67
        assert priv["mean_sigma_t"] == pytest.approx(0.006)

        md = render_markdown(summary, plots=[])
        assert "## Privacy" in md
        assert "SecAgg+ masking" in md
        assert "Clip norm (C)" in md
        assert "0.0060" in md
