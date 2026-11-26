"""Tests for Faz 2: privacy configuration (RunConfig/PrivacySection) and SecAgg+ protocol models."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from cfl.coordinator.config import PrivacySection, RunConfig, load_run_config
from cfl.core.secagg import KeyPair, generate_self_mask_seed, reconstruct_secret, split_secret
from cfl.protocol.messages import (
    PeerKeyEntry,
    RevealedShare,
    RoundKeysResponse,
    RoundKeySubmit,
    ShamirSharePayload,
    TaskAssignment,
    TransferReport,
    UnmaskSharesRequest,
    UnmaskSharesResponse,
)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def _base_config_dict(**overrides) -> dict:
    cfg = {
        "run": {"name": "test-run", "seed": 42, "mode": "simulation"},
        "task": {"name": "tabular", "data_dir": "data/tabular", "num_clients": 4},
        "strategy": {"name": "fedprox", "mu": 0.01},
        "rounds": {
            "num_rounds": 5,
            "client_fraction": 1.0,
            "min_participants": 3,
            "round_deadline_s": 60,
        },
        "client": {"local_epochs": 1, "batch_size": 32, "learning_rate": 0.01},
    }
    cfg.update(overrides)
    return cfg


# ========================== config.py (PrivacySection) ==========================


class TestPrivacySection:
    def test_defaults_when_omitted(self):
        cfg = RunConfig.model_validate(_base_config_dict())
        assert cfg.privacy.enabled is False
        assert cfg.privacy.secagg_enabled is False
        assert cfg.privacy.clip_norm > 0
        assert cfg.privacy.base_noise_std >= 0
        assert 0 < cfg.privacy.min_recovery_ratio <= 1

    def test_existing_yaml_configs_still_load(self):
        for path in sorted(CONFIGS_DIR.glob("*.yaml")):
            cfg = load_run_config(path)
            if path.stem.endswith("_privacy"):
                assert cfg.privacy.enabled is True
                assert cfg.privacy.secagg_enabled is True
            else:
                assert cfg.privacy.enabled is False

    def test_valid_privacy_and_secagg_config(self):
        cfg = RunConfig.model_validate(
            _base_config_dict(
                privacy={
                    "enabled": True,
                    "clip_norm": 1.5,
                    "base_noise_std": 0.05,
                    "secagg_enabled": True,
                    "min_recovery_ratio": 0.67,
                }
            )
        )
        assert cfg.privacy.enabled is True
        assert cfg.privacy.secagg_enabled is True
        assert cfg.privacy.clip_norm == 1.5
        assert cfg.privacy.base_noise_std == 0.05
        assert cfg.privacy.recovery_threshold(3) == 3  # ceil(0.67 * 3) = 3
        assert cfg.privacy.recovery_threshold(5) == 4  # ceil(0.67 * 5) = 4
        assert cfg.privacy.recovery_threshold(2) == 2

    def test_dp_only_without_secagg_allows_single_participant(self):
        data = _base_config_dict(
            privacy={"enabled": True, "secagg_enabled": False, "clip_norm": 2.0}
        )
        data["rounds"]["min_participants"] = 1
        cfg = RunConfig.model_validate(data)
        assert cfg.privacy.enabled is True
        assert cfg.privacy.secagg_enabled is False

    def test_secagg_requires_privacy_enabled(self):
        with pytest.raises(ValidationError, match="privacy.enabled"):
            PrivacySection(enabled=False, secagg_enabled=True)

    def test_secagg_requires_at_least_two_min_participants(self):
        data = _base_config_dict(privacy={"enabled": True, "secagg_enabled": True})
        data["rounds"]["min_participants"] = 1
        with pytest.raises(ValidationError, match="min_participants >= 2"):
            RunConfig.model_validate(data)

    @pytest.mark.parametrize(
        "bad_fields",
        [
            {"clip_norm": 0.0},
            {"clip_norm": -1.0},
            {"base_noise_std": -0.01},
            {"min_recovery_ratio": 0.0},
            {"min_recovery_ratio": 1.5},
            {"unknown_field": True},
        ],
    )
    def test_invalid_privacy_fields_rejected(self, bad_fields):
        with pytest.raises(ValidationError):
            PrivacySection(**bad_fields)

    def test_recovery_threshold_bounds(self):
        sec = PrivacySection(enabled=True, secagg_enabled=True, min_recovery_ratio=0.1)
        assert sec.recovery_threshold(4) == 2  # clamped up to Shamir minimum 2
        with pytest.raises(ValueError, match="at least 2"):
            sec.recovery_threshold(1)


# ========================== messages.py (Protocol) ==========================


class TestTaskAssignmentPrivacyFields:
    def test_backwards_compatible_without_privacy_fields(self):
        task = TaskAssignment(
            run_id="r1",
            round_id=1,
            model_version=0,
            deadline=datetime.now(UTC),
        )
        assert task.clip_norm is None
        assert task.sigma_t is None
        assert task.trust_weight is None
        assert task.secagg_threshold is None

    def test_valid_privacy_fields(self):
        task = TaskAssignment(
            run_id="r1",
            round_id=2,
            model_version=1,
            deadline=datetime.now(UTC),
            clip_norm=1.0,
            sigma_t=0.02,
            trust_weight=0.25,
            secagg_threshold=3,
        )
        assert task.clip_norm == 1.0
        assert task.sigma_t == 0.02
        assert task.trust_weight == 0.25
        assert task.secagg_threshold == 3

    @pytest.mark.parametrize(
        "bad",
        [
            {"clip_norm": 0.0},
            {"sigma_t": -0.1},
            {"trust_weight": 0.0},
            {"trust_weight": 1.1},
            {"secagg_threshold": 1},
        ],
    )
    def test_invalid_privacy_fields_rejected(self, bad):
        with pytest.raises(ValidationError):
            TaskAssignment(
                run_id="r1",
                round_id=1,
                model_version=0,
                deadline=datetime.now(UTC),
                **bad,
            )


class TestSecAggProtocolMessages:
    def test_key_submit_and_response_roundtrip(self):
        kp1, kp2 = KeyPair.generate(), KeyPair.generate()
        submit = RoundKeySubmit(
            run_id="r1",
            round_id=1,
            client_id="c1",
            public_key_hex=kp1.public_bytes.hex().upper(),
            encrypted_shares={"c2": "deadbeef"},
        )
        assert submit.public_key_hex == kp1.public_bytes.hex()  # normalized lowercase

        resp = RoundKeysResponse(
            run_id="r1",
            round_id=1,
            ready=True,
            secagg_threshold=2,
            peers=[
                PeerKeyEntry(client_id="c1", public_key_hex=kp1.public_bytes.hex()),
                PeerKeyEntry(client_id="c2", public_key_hex=kp2.public_bytes.hex()),
            ],
            incoming_shares={"c2": "cafebabe"},
        )
        parsed = RoundKeysResponse.model_validate_json(resp.model_dump_json())
        assert parsed.peers[0].public_key_bytes() == kp1.public_bytes
        assert parsed.peers[1].public_key_bytes() == kp2.public_bytes
        assert parsed.incoming_shares == {"c2": "cafebabe"}

    @pytest.mark.parametrize("bad_key", ["not-hex", "aabbcc", "00" * 31, "00" * 33])
    def test_invalid_public_key_hex_rejected(self, bad_key):
        with pytest.raises(ValidationError):
            RoundKeySubmit(run_id="r1", round_id=1, client_id="c1", public_key_hex=bad_key)
        with pytest.raises(ValidationError):
            PeerKeyEntry(client_id="c1", public_key_hex=bad_key)

    def test_shamir_share_payload_json_roundtrip(self):
        secret = generate_self_mask_seed()
        shares = split_secret(secret, n=4, k=3)
        payloads = [ShamirSharePayload.from_share(s) for s in shares[:3]]
        reloaded = [
            ShamirSharePayload.model_validate_json(p.model_dump_json()).to_share() for p in payloads
        ]
        assert reconstruct_secret(reloaded) == secret

    def test_unmask_shares_request_valid(self):
        secret = generate_self_mask_seed()
        share_payload = ShamirSharePayload.from_share(split_secret(secret, n=3, k=2)[0])
        req = UnmaskSharesRequest(
            run_id="r1",
            round_id=1,
            client_id="c1",
            shares=[
                RevealedShare(target_client_id="c2", kind="self_mask", share=share_payload),
                RevealedShare(target_client_id="c3", kind="mask_key", share=share_payload),
            ],
        )
        parsed = UnmaskSharesRequest.model_validate_json(req.model_dump_json())
        assert len(parsed.shares) == 2
        ack = UnmaskSharesResponse(accepted=True, round_id=1)
        assert ack.accepted is True

    def test_unmask_request_rejects_both_kinds_for_same_peer(self):
        share_payload = ShamirSharePayload(x=1, y_hex="abcd")
        with pytest.raises(ValidationError, match="cannot reveal both"):
            UnmaskSharesRequest(
                run_id="r1",
                round_id=1,
                client_id="c1",
                shares=[
                    RevealedShare(target_client_id="c2", kind="self_mask", share=share_payload),
                    RevealedShare(target_client_id="c2", kind="mask_key", share=share_payload),
                ],
            )

    def test_unmask_request_rejects_duplicate_share_for_same_peer(self):
        share_payload = ShamirSharePayload(x=1, y_hex="abcd")
        with pytest.raises(ValidationError, match="duplicate"):
            UnmaskSharesRequest(
                run_id="r1",
                round_id=1,
                client_id="c1",
                shares=[
                    RevealedShare(target_client_id="c2", kind="self_mask", share=share_payload),
                    RevealedShare(target_client_id="c2", kind="self_mask", share=share_payload),
                ],
            )

    def test_transfer_report_accepts_secagg_message_types(self):
        for msg_type in ("secagg_keys", "secagg_unmask"):
            rep = TransferReport(run_id="r1", round_id=1, message_type=msg_type, duration_ms=12.5)
            assert rep.message_type == msg_type
