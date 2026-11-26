"""Tests for the privacy extension — Faz 1 (core layer).

Covers:
- cfl.core.privacy: clip_delta, calibrate_noise_std, add_dp_noise, scale_by_trust,
  quantize_state_dict / dequantize_state_dict
- cfl.core.secagg: key exchange, pairwise masking, mask cancellation, Shamir Secret
  Sharing
- cfl.core.strategy: delta_sum, compute_dynamic_trust_weights, ClientStats
"""

from __future__ import annotations

import pytest
import torch

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
    ShareKind,
    apply_masks,
    generate_self_mask_seed,
    private_key_from_bytes,
    private_key_to_bytes,
    reconstruct_secret,
    round_context,
    share_to_reveal,
    split_secret,
    sum_masked,
    unmask_aggregate,
)
from cfl.core.strategy import (
    ClientStats,
    compute_dynamic_trust_weights,
    delta_sum,
)

# ========================== helpers ==========================


def _make_weights(seed: int = 0) -> dict[str, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    return {
        "layer.weight": torch.randn(4, 3, generator=rng),
        "layer.bias": torch.randn(4, generator=rng),
    }


# ========================== privacy.py ==========================


class TestClipDelta:
    def test_no_clipping_when_within_bound(self):
        w = {"a": torch.tensor([0.1, -0.1])}
        g = {"a": torch.zeros(2)}
        delta = clip_delta(w, g, clip_norm=10.0)
        torch.testing.assert_close(delta["a"], w["a"])

    def test_clipping_scales_to_norm(self):
        w = {"a": torch.tensor([3.0, 4.0])}  # norm = 5
        g = {"a": torch.zeros(2)}
        delta = clip_delta(w, g, clip_norm=1.0)
        norm = torch.linalg.vector_norm(delta["a"]).item()
        assert abs(norm - 1.0) < 1e-5

    def test_negative_clip_norm_raises(self):
        w = {"a": torch.tensor([1.0])}
        g = {"a": torch.zeros(1)}
        with pytest.raises(ValueError, match="positive"):
            clip_delta(w, g, clip_norm=-1.0)

    def test_multi_param_global_norm(self):
        # Delta across two params: [3, 0] and [4] → global norm = 5
        w = {"a": torch.tensor([3.0, 0.0]), "b": torch.tensor([4.0])}
        g = {"a": torch.zeros(2), "b": torch.zeros(1)}
        delta = clip_delta(w, g, clip_norm=2.5)
        global_sq = delta["a"].square().sum() + delta["b"].square().sum()
        assert abs(global_sq.sqrt().item() - 2.5) < 1e-5


class TestCalibrateNoiseStd:
    def test_basic_scaling(self):
        assert calibrate_noise_std(1.0, 4) == pytest.approx(0.5)

    def test_single_participant(self):
        assert calibrate_noise_std(1.0, 1) == pytest.approx(1.0)

    def test_zero_sigma(self):
        assert calibrate_noise_std(0.0, 10) == 0.0

    def test_invalid_participants(self):
        with pytest.raises(ValueError):
            calibrate_noise_std(1.0, 0)


class TestAddDpNoise:
    def test_zero_sigma_returns_same(self):
        d = {"a": torch.tensor([1.0, 2.0])}
        result = add_dp_noise(d, sigma=0.0)
        assert result is d  # exact same object for efficiency

    def test_nonzero_sigma_adds_noise(self):
        d = {"a": torch.ones(100)}
        rng = torch.Generator().manual_seed(42)
        result = add_dp_noise(d, sigma=1.0, rng=rng)
        # Should not be exactly the same
        assert not torch.equal(result["a"], d["a"])

    def test_noise_distribution_roughly_correct(self):
        d = {"a": torch.zeros(50000)}
        rng = torch.Generator().manual_seed(123)
        result = add_dp_noise(d, sigma=0.5, rng=rng)
        # Mean should be near 0, std near 0.5
        assert abs(result["a"].mean().item()) < 0.02
        assert abs(result["a"].std().item() - 0.5) < 0.02


class TestScaleByTrust:
    def test_basic_scaling(self):
        d = {"a": torch.tensor([2.0, 4.0])}
        result = scale_by_trust(d, 0.5)
        torch.testing.assert_close(result["a"], torch.tensor([1.0, 2.0]))

    def test_invalid_alpha(self):
        d = {"a": torch.tensor([1.0])}
        with pytest.raises(ValueError):
            scale_by_trust(d, 0.0)
        with pytest.raises(ValueError):
            scale_by_trust(d, 1.5)


class TestQuantizeDequantize:
    def test_roundtrip_positive(self):
        d = {"a": torch.tensor([0.5, 1.0, -0.5])}
        q = quantize_state_dict(d)
        r = dequantize_state_dict(q)
        # Should be close to original within quantization error
        torch.testing.assert_close(r["a"], d["a"], atol=1e-6, rtol=1e-6)

    def test_modular_addition_preserves_sum(self):
        """The core SecAgg property: quantize(a) + quantize(b) mod M → dequantize → a + b."""
        a = {"x": torch.tensor([0.3, -0.7])}
        b = {"x": torch.tensor([0.5, 0.2])}
        qa = quantize_state_dict(a)
        qb = quantize_state_dict(b)
        qsum = {"x": (qa["x"] + qb["x"]) % DEFAULT_MODULUS}
        result = dequantize_state_dict(qsum)
        expected = {"x": a["x"] + b["x"]}
        torch.testing.assert_close(result["x"], expected["x"], atol=1e-6, rtol=1e-6)

    def test_three_way_sum(self):
        vals = [
            {"x": torch.tensor([0.1, -0.3, 0.5])},
            {"x": torch.tensor([-0.2, 0.4, -0.1])},
            {"x": torch.tensor([0.3, -0.1, 0.0])},
        ]
        quants = [quantize_state_dict(v) for v in vals]
        qsum = {"x": sum(q["x"] for q in quants) % DEFAULT_MODULUS}
        result = dequantize_state_dict(qsum)
        expected = {"x": sum(v["x"] for v in vals)}
        torch.testing.assert_close(result["x"], expected["x"], atol=1e-6, rtol=1e-6)


# ========================== secagg.py ==========================


class TestKeyPair:
    def test_generate_produces_32_byte_public_key(self):
        kp = KeyPair.generate()
        assert len(kp.public_bytes) == 32

    def test_different_keypairs(self):
        kp1 = KeyPair.generate()
        kp2 = KeyPair.generate()
        assert kp1.public_bytes != kp2.public_bytes


class TestPairwiseMasking:
    def test_two_client_masks_cancel(self):
        """Core SecAgg guarantee: masks cancel out when summed."""
        ctx = round_context("run-1", 1)
        kp_a = KeyPair.generate()
        kp_b = KeyPair.generate()

        # Simple quantized deltas
        delta_a = {"w": torch.tensor([100, 200, 300], dtype=torch.int64)}
        delta_b = {"w": torch.tensor([400, 500, 600], dtype=torch.int64)}

        peer_keys = {"alice": kp_a.public_bytes, "bob": kp_b.public_bytes}

        masked_a = apply_masks(delta_a, "alice", kp_a, ["alice", "bob"], peer_keys, ctx)
        masked_b = apply_masks(delta_b, "bob", kp_b, ["alice", "bob"], peer_keys, ctx)

        # Sum mod M should equal the unmasked sum
        m = DEFAULT_MODULUS
        unmasked_sum = (delta_a["w"] + delta_b["w"]) % m
        masked_sum = (masked_a["w"] + masked_b["w"]) % m
        torch.testing.assert_close(masked_sum, unmasked_sum)

    def test_three_client_masks_cancel(self):
        """3 clients: Σ masks = 0 (mod M)."""
        ctx = round_context("run-1", 2)
        ids = ["c1", "c2", "c3"]
        keypairs = {cid: KeyPair.generate() for cid in ids}
        peer_keys = {cid: kp.public_bytes for cid, kp in keypairs.items()}

        deltas = {cid: {"w": torch.randint(0, 1000, (5,), dtype=torch.int64)} for cid in ids}

        masked = {
            cid: apply_masks(deltas[cid], cid, keypairs[cid], ids, peer_keys, ctx) for cid in ids
        }

        m = DEFAULT_MODULUS
        raw_sum = sum(deltas[cid]["w"] for cid in ids) % m
        mask_sum = sum(masked[cid]["w"] for cid in ids) % m
        torch.testing.assert_close(mask_sum, raw_sum)

    def test_individual_masked_values_differ(self):
        """Masked values should NOT equal the original (masks are non-trivial)."""
        ctx = round_context("run-1", 1)
        kp_a = KeyPair.generate()
        kp_b = KeyPair.generate()
        delta_a = {"w": torch.tensor([100, 200, 300], dtype=torch.int64)}
        peer_keys = {"a": kp_a.public_bytes, "b": kp_b.public_bytes}
        masked_a = apply_masks(delta_a, "a", kp_a, ["a", "b"], peer_keys, ctx)
        # With overwhelming probability the mask is non-zero
        assert not torch.equal(masked_a["w"], delta_a["w"])


class TestShamirSecretSharing:
    def test_basic_split_reconstruct(self):
        secret = b"\x42" * 32
        shares = split_secret(secret, n=5, k=3)
        assert len(shares) == 5
        # Any 3 shares should reconstruct
        recovered = reconstruct_secret(shares[:3])
        assert recovered == secret

    def test_different_subsets_same_result(self):
        secret = b"\xab" * 32
        shares = split_secret(secret, n=5, k=3)
        assert reconstruct_secret(shares[:3]) == secret
        assert reconstruct_secret(shares[2:]) == secret
        assert reconstruct_secret([shares[0], shares[2], shares[4]]) == secret

    def test_minimum_threshold(self):
        secret = b"\x01" * 32
        shares = split_secret(secret, n=4, k=2)
        # 2 shares should work
        recovered = reconstruct_secret(shares[:2])
        assert recovered == secret

    def test_private_key_roundtrip(self):
        """Split an X25519 private key, reconstruct it, and verify the public key matches."""
        kp = KeyPair.generate()
        raw = private_key_to_bytes(kp.private_key)
        shares = split_secret(raw, n=5, k=3)
        recovered_raw = reconstruct_secret(shares[1:4])
        recovered_key = private_key_from_bytes(recovered_raw)
        assert recovered_key.public_key().public_bytes_raw() == kp.public_bytes


# ========================== strategy.py ==========================


class TestDeltaSum:
    def test_basic_delta_sum(self):
        g = {"a": torch.tensor([1.0, 2.0])}
        d1 = {"a": torch.tensor([0.1, 0.2])}
        d2 = {"a": torch.tensor([0.3, 0.4])}
        result = delta_sum(g, [d1, d2])
        expected = torch.tensor([1.4, 2.6])
        torch.testing.assert_close(result["a"], expected)

    def test_empty_list_raises(self):
        g = {"a": torch.tensor([1.0])}
        with pytest.raises(ValueError, match="empty"):
            delta_sum(g, [])


class TestDynamicTrustWeights:
    def test_uniform_clients_get_equal_weights(self):
        stats = [
            ClientStats("c1", rounds_participated=10, rounds_selected=10, on_time_deliveries=10),
            ClientStats("c2", rounds_participated=10, rounds_selected=10, on_time_deliveries=10),
        ]
        weights = compute_dynamic_trust_weights(stats)
        assert weights["c1"] == pytest.approx(0.5)
        assert weights["c2"] == pytest.approx(0.5)

    def test_weights_sum_to_one(self):
        stats = [
            ClientStats("c1", rounds_participated=8, rounds_selected=10, on_time_deliveries=6),
            ClientStats("c2", rounds_participated=10, rounds_selected=10, on_time_deliveries=10),
            ClientStats("c3", rounds_participated=5, rounds_selected=10, on_time_deliveries=4),
        ]
        weights = compute_dynamic_trust_weights(stats)
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_better_client_gets_higher_weight(self):
        stats = [
            ClientStats("good", rounds_participated=10, rounds_selected=10, on_time_deliveries=10),
            ClientStats("bad", rounds_participated=3, rounds_selected=10, on_time_deliveries=1),
        ]
        weights = compute_dynamic_trust_weights(stats)
        assert weights["good"] > weights["bad"]

    def test_validation_instability_lowers_weight(self):
        stats = [
            ClientStats(
                "stable",
                rounds_participated=10,
                rounds_selected=10,
                on_time_deliveries=10,
                val_metrics=[0.5, 0.5, 0.5, 0.5],
            ),
            ClientStats(
                "erratic",
                rounds_participated=10,
                rounds_selected=10,
                on_time_deliveries=10,
                val_metrics=[0.1, 0.9, 0.2, 0.8],
            ),
        ]
        weights = compute_dynamic_trust_weights(stats)
        assert weights["stable"] > weights["erratic"]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_dynamic_trust_weights([])


# ========================== End-to-end pipeline ==========================


class TestEndToEndPipeline:
    """Full client-side pipeline: clip → noise → trust-scale → quantize → mask → sum → unmask
    → dequantize → delta_sum → verify correctness."""

    def test_full_pipeline_three_clients(self):
        torch.manual_seed(42)
        global_weights = {"w": torch.randn(10)}
        client_weights = [
            {"w": global_weights["w"] + torch.randn(10) * 0.5},
            {"w": global_weights["w"] + torch.randn(10) * 0.5},
            {"w": global_weights["w"] + torch.randn(10) * 0.5},
        ]
        alphas = [0.4, 0.35, 0.25]
        clip_norm = 2.0
        sigma = calibrate_noise_std(0.001, 3)  # very low noise for test precision

        # Client-side: clip → noise → trust-scale → quantize
        deltas_private = []
        for i, cw in enumerate(client_weights):
            d = clip_delta(cw, global_weights, clip_norm)
            rng = torch.Generator().manual_seed(i + 100)
            d = add_dp_noise(d, sigma, rng)
            d = scale_by_trust(d, alphas[i])
            deltas_private.append(d)

        quantized = [quantize_state_dict(d) for d in deltas_private]

        # SecAgg masking
        ctx = round_context("run-e2e", 1)
        ids = ["c0", "c1", "c2"]
        keypairs = {cid: KeyPair.generate() for cid in ids}
        peer_keys = {cid: kp.public_bytes for cid, kp in keypairs.items()}

        masked = [
            apply_masks(quantized[i], ids[i], keypairs[ids[i]], ids, peer_keys, ctx)
            for i in range(3)
        ]

        # Server-side: sum masked → dequantize → delta_sum
        m = DEFAULT_MODULUS
        qsum = {"w": sum(masked[i]["w"] for i in range(3)) % m}
        delta_agg = dequantize_state_dict(qsum)
        new_global = delta_sum(global_weights, [delta_agg])

        # Verify: the result should match direct computation (within quantization error)
        expected_delta = sum(d["w"] for d in deltas_private)
        expected_global = global_weights["w"] + expected_delta
        torch.testing.assert_close(new_global["w"], expected_global, atol=1e-5, rtol=1e-5)


# ========================== Double masking & dropout recovery ==========================


def _setup_round(ids, size=6, seed=0):
    rng = torch.Generator().manual_seed(seed)
    ctx = round_context("run-sec", 3)
    keypairs = {cid: KeyPair.generate() for cid in ids}
    pub = {cid: kp.public_bytes for cid, kp in keypairs.items()}
    self_seeds = {cid: generate_self_mask_seed() for cid in ids}
    raw = {cid: quantize_state_dict({"w": torch.randn(size, generator=rng) * 0.1}) for cid in ids}
    masked = {
        cid: apply_masks(raw[cid], cid, keypairs[cid], ids, pub, ctx, self_seeds[cid])
        for cid in ids
    }
    return ctx, keypairs, pub, self_seeds, raw, masked


class TestDoubleMasking:
    def test_no_dropout_recovers_exact_sum(self):
        ids = ["c1", "c2", "c3", "c4"]
        ctx, _, _, seeds, raw, masked = _setup_round(ids)
        agg = sum_masked([masked[c] for c in ids])
        result = unmask_aggregate(agg, ctx, surviving_self_seeds=seeds)
        expected = sum_masked([raw[c] for c in ids])
        torch.testing.assert_close(result["w"], expected["w"])

    def test_sum_without_self_seeds_is_still_hidden(self):
        """Pairwise masks cancel, but the self masks keep the aggregate blind until the
        survivors' shares are combined."""
        ids = ["c1", "c2", "c3"]
        _, _, _, _, raw, masked = _setup_round(ids)
        agg = sum_masked([masked[c] for c in ids])
        assert not torch.equal(agg["w"], sum_masked([raw[c] for c in ids])["w"])

    def test_dropout_recovery_with_shamir(self):
        """c3 drops after the key exchange: survivors reconstruct its mask key from k shares
        and the server still gets the exact sum of the survivors."""
        ids = ["c1", "c2", "c3", "c4"]
        ctx, keypairs, pub, seeds, raw, masked = _setup_round(ids)
        survivors, dropped = ["c1", "c2", "c4"], "c3"
        k = 2

        # Round setup: every client shares its secrets (n = len(ids)).
        key_shares = split_secret(private_key_to_bytes(keypairs[dropped].private_key), 4, k)
        seed_shares = {c: split_secret(seeds[c], 4, k) for c in survivors}

        agg = sum_masked([masked[c] for c in survivors])
        recovered_sk = private_key_from_bytes(reconstruct_secret(key_shares[:k]))
        recovered_seeds = {c: reconstruct_secret(seed_shares[c][1 : 1 + k]) for c in survivors}

        result = unmask_aggregate(
            agg,
            ctx,
            surviving_self_seeds=recovered_seeds,
            dropped_private_keys={dropped: recovered_sk},
            surviving_public_keys={c: pub[c] for c in survivors},
        )
        expected = sum_masked([raw[c] for c in survivors])
        torch.testing.assert_close(result["w"], expected["w"])

    def test_dropout_of_lowest_and_highest_id(self):
        ids = ["a", "b", "c", "d", "e"]
        ctx, keypairs, pub, seeds, raw, masked = _setup_round(ids, seed=7)
        survivors = ["b", "c", "d"]
        result = unmask_aggregate(
            sum_masked([masked[c] for c in survivors]),
            ctx,
            surviving_self_seeds={c: seeds[c] for c in survivors},
            dropped_private_keys={d: keypairs[d].private_key for d in ("a", "e")},
            surviving_public_keys={c: pub[c] for c in survivors},
        )
        torch.testing.assert_close(result["w"], sum_masked([raw[c] for c in survivors])["w"])


class TestCompromisedAggregator:
    """Threat model: the aggregator is hacked and tries to see one client's update."""

    def test_single_masked_upload_reveals_nothing_useful(self):
        ids = ["c1", "c2", "c3"]
        _, _, _, _, raw, masked = _setup_round(ids)
        leaked = dequantize_state_dict(masked["c1"])["w"]
        true = dequantize_state_dict(raw["c1"])["w"]
        # The masked value is uniform over a huge range, nowhere near the true update.
        assert (leaked - true).abs().max() > 1.0

    def test_false_dropout_claim_does_not_unmask_victim(self):
        """Server claims c1 dropped (gets c1's mask key) while c1's upload is in hand.
        Without b_1 the update stays hidden behind the self mask."""
        ids = ["c1", "c2", "c3"]
        ctx, keypairs, pub, _, raw, masked = _setup_round(ids)
        attempt = unmask_aggregate(
            masked["c1"],
            ctx,
            surviving_self_seeds={},
            dropped_private_keys={"c1": keypairs["c1"].private_key},
            surviving_public_keys={c: pub[c] for c in ("c2", "c3")},
        )
        # Pairwise masks are removed, but PRG(b_1) remains, so the result is not x_1.
        assert not torch.equal(attempt["w"], raw["c1"]["w"])

    def test_honest_client_refuses_both_shares_for_same_peer(self):
        with pytest.raises(ValueError, match="both"):
            share_to_reveal("c1", surviving={"c1", "c2"}, dropped={"c1"})
        assert share_to_reveal("c1", {"c1"}, set()) is ShareKind.SELF_MASK
        assert share_to_reveal("c2", {"c1"}, {"c2"}) is ShareKind.MASK_KEY
        with pytest.raises(ValueError):
            share_to_reveal("x", {"c1"}, {"c2"})


class TestClientSideInference:
    """Threat model: a client subtracts its own update from the published aggregate."""

    def test_two_clients_without_dp_leak_peer_exactly(self):
        """Documents why DP is required: with N=2 and no noise, subtraction is exact."""
        g = {"w": torch.zeros(5)}
        a, b = {"w": torch.randn(5) * 0.1}, {"w": torch.randn(5) * 0.1}
        agg = delta_sum(g, [clip_delta(a, g, 10.0), clip_delta(b, g, 10.0)])
        inferred_b = agg["w"] - a["w"]
        torch.testing.assert_close(inferred_b, b["w"])

    def test_dp_noise_hides_peer_update(self):
        torch.manual_seed(0)
        g = {"w": torch.zeros(2000)}
        a, b = {"w": torch.randn(2000) * 0.01}, {"w": torch.randn(2000) * 0.01}
        sigma = calibrate_noise_std(0.1, 2)
        da = add_dp_noise(clip_delta(a, g, 10.0), sigma, torch.Generator().manual_seed(1))
        db = add_dp_noise(clip_delta(b, g, 10.0), sigma, torch.Generator().manual_seed(2))
        agg = delta_sum(g, [da, db])
        inferred_b = agg["w"] - da["w"]  # attacker knows its own noisy delta
        error = (inferred_b - b["w"]).std().item()
        # The estimate error is dominated by σ_t, far above the signal scale (0.01).
        assert error > 5 * b["w"].std().item()

    def test_noise_shrinks_with_more_participants(self):
        sigmas = [calibrate_noise_std(1.0, n) for n in (2, 4, 9, 16)]
        assert sigmas == sorted(sigmas, reverse=True)
        assert sigmas[-1] == pytest.approx(0.25)


class TestShamirEdgeCases:
    def test_fewer_than_k_shares_give_wrong_secret(self):
        secret = generate_self_mask_seed()
        shares = split_secret(secret, n=5, k=3)
        assert reconstruct_secret(shares[:2]) != secret

    def test_duplicate_shares_rejected(self):
        shares = split_secret(b"\x07" * 32, n=3, k=2)
        with pytest.raises(ValueError, match="duplicate"):
            reconstruct_secret([shares[0], shares[0]])

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            split_secret(b"\x00" * 32, n=3, k=4)
