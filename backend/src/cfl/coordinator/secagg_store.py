"""Per-round privacy and SecAgg+ ephemeral state stored in the shared ModelStore.

Both the gateway and coordinator mount the same ModelStore (InMemoryModelStore in simulation,
FileModelStore on the shared `/models` volume in Docker), so storing per-client JSON files
here lets both services coordinate key exchange and unmasking without races and without
altering the PostgreSQL schema. All ephemeral key/share files are deleted once the round ends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

from cfl.core.secagg import (
    ShamirShare,
    ShareKind,
    private_key_from_bytes,
    reconstruct_secret,
)
from cfl.core.store import ModelStore
from cfl.protocol.messages import (
    PeerKeyEntry,
    RoundKeysResponse,
    RoundKeySubmit,
    UnmaskSharesRequest,
)


@dataclass(frozen=True)
class RoundPrivacyParams:
    clip_norm: float
    sigma_t: float
    trust_weights: dict[str, float]
    secagg_enabled: bool
    secagg_threshold: int | None
    selected: list[str]


@dataclass(frozen=True)
class UnmaskState:
    surviving: list[str]
    dropped: list[str]
    unmask_deadline: datetime


class SecAggRoundStore:
    """Atomic per-file persistence for a round's privacy parameters, keys and Shamir shares."""

    def __init__(self, models: ModelStore):
        self.models = models

    @staticmethod
    def _prefix(run_id: str, round_id: int) -> str:
        return f"{run_id}/round_{round_id}/secagg"

    def _params_key(self, run_id: str, round_id: int) -> str:
        return f"{self._prefix(run_id, round_id)}/params.json"

    def _client_key_path(self, run_id: str, round_id: int, client_id: str) -> str:
        return f"{self._prefix(run_id, round_id)}/key_{client_id}.json"

    def _unmask_state_key(self, run_id: str, round_id: int) -> str:
        return f"{self._prefix(run_id, round_id)}/unmask_state.json"

    def _unmask_shares_path(self, run_id: str, round_id: int, client_id: str) -> str:
        return f"{self._prefix(run_id, round_id)}/unmask_{client_id}.json"

    # -- round privacy parameters --------------------------------------------------------------

    def save_params(self, run_id: str, round_id: int, params: RoundPrivacyParams) -> None:
        payload = {
            "clip_norm": params.clip_norm,
            "sigma_t": params.sigma_t,
            "trust_weights": params.trust_weights,
            "secagg_enabled": params.secagg_enabled,
            "secagg_threshold": params.secagg_threshold,
            "selected": params.selected,
        }
        self.models.put(self._params_key(run_id, round_id), json.dumps(payload).encode())

    def load_params(self, run_id: str, round_id: int) -> RoundPrivacyParams | None:
        try:
            data = json.loads(self.models.get(self._params_key(run_id, round_id)))
        except KeyError:
            return None
        return RoundPrivacyParams(
            clip_norm=float(data["clip_norm"]),
            sigma_t=float(data["sigma_t"]),
            trust_weights={k: float(v) for k, v in data["trust_weights"].items()},
            secagg_enabled=bool(data["secagg_enabled"]),
            secagg_threshold=(
                int(data["secagg_threshold"]) if data["secagg_threshold"] is not None else None
            ),
            selected=list(data["selected"]),
        )

    # -- key exchange --------------------------------------------------------------------------

    def save_client_key(self, submit: RoundKeySubmit) -> None:
        path = self._client_key_path(submit.run_id, submit.round_id, submit.client_id)
        self.models.put(path, submit.model_dump_json().encode())

    def load_client_keys(
        self, run_id: str, round_id: int, client_ids: list[str]
    ) -> dict[str, RoundKeySubmit]:
        found: dict[str, RoundKeySubmit] = {}
        for cid in sorted(client_ids):
            try:
                raw = self.models.get(self._client_key_path(run_id, round_id, cid))
            except KeyError:
                continue
            found[cid] = RoundKeySubmit.model_validate_json(raw)
        return found

    def keys_response_for(
        self,
        run_id: str,
        round_id: int,
        requester_id: str,
        selected: list[str],
        threshold: int,
    ) -> RoundKeysResponse:
        submitted = self.load_client_keys(run_id, round_id, selected)
        peers = [
            PeerKeyEntry(client_id=cid, public_key_hex=submitted[cid].public_key_hex)
            for cid in sorted(submitted)
        ]
        incoming = {
            cid: item.encrypted_shares[requester_id]
            for cid, item in submitted.items()
            if requester_id in item.encrypted_shares
        }
        return RoundKeysResponse(
            run_id=run_id,
            round_id=round_id,
            ready=len(submitted) == len(selected),
            secagg_threshold=threshold,
            peers=peers,
            incoming_shares=incoming,
        )

    # -- unmasking & dropout recovery ----------------------------------------------------------

    def save_unmask_state(self, run_id: str, round_id: int, state: UnmaskState) -> None:
        payload = {
            "surviving": state.surviving,
            "dropped": state.dropped,
            "unmask_deadline": state.unmask_deadline.isoformat(),
        }
        self.models.put(self._unmask_state_key(run_id, round_id), json.dumps(payload).encode())

    def load_unmask_state(self, run_id: str, round_id: int) -> UnmaskState | None:
        try:
            data = json.loads(self.models.get(self._unmask_state_key(run_id, round_id)))
        except KeyError:
            return None
        return UnmaskState(
            surviving=list(data["surviving"]),
            dropped=list(data["dropped"]),
            unmask_deadline=datetime.fromisoformat(data["unmask_deadline"]),
        )

    def save_unmask_shares(self, req: UnmaskSharesRequest) -> None:
        path = self._unmask_shares_path(req.run_id, req.round_id, req.client_id)
        self.models.put(path, req.model_dump_json().encode())

    def load_unmask_shares(
        self, run_id: str, round_id: int, client_ids: list[str]
    ) -> dict[str, UnmaskSharesRequest]:
        found: dict[str, UnmaskSharesRequest] = {}
        for cid in sorted(client_ids):
            try:
                raw = self.models.get(self._unmask_shares_path(run_id, round_id, cid))
            except KeyError:
                continue
            found[cid] = UnmaskSharesRequest.model_validate_json(raw)
        return found

    def collect_recovery_secrets(
        self,
        run_id: str,
        round_id: int,
        surviving: list[str],
        dropped: list[str],
        threshold: int,
    ) -> tuple[dict[str, bytes], dict[str, X25519PrivateKey]] | None:
        """Reconstruct self-mask seeds for survivors and mask keys for dropouts once >= k_t
        distinct shares have been submitted for every target. Returns None while still waiting."""
        submissions = self.load_unmask_shares(run_id, round_id, surviving)
        by_target: dict[tuple[str, str], dict[int, ShamirShare]] = {}
        for req in submissions.values():
            for item in req.shares:
                share = item.share.to_share()
                by_target.setdefault((item.target_client_id, item.kind), {})[share.x] = share

        self_seeds: dict[str, bytes] = {}
        for cid in surviving:
            shares = list(by_target.get((cid, ShareKind.SELF_MASK.value), {}).values())
            if len(shares) < threshold:
                return None
            self_seeds[cid] = reconstruct_secret(shares[:threshold])

        dropped_keys: dict[str, X25519PrivateKey] = {}
        for cid in dropped:
            shares = list(by_target.get((cid, ShareKind.MASK_KEY.value), {}).values())
            if len(shares) < threshold:
                return None
            dropped_keys[cid] = private_key_from_bytes(reconstruct_secret(shares[:threshold]))

        return self_seeds, dropped_keys

    def cleanup_round(self, run_id: str, round_id: int, client_ids: list[str]) -> None:
        """Delete ephemeral keys and Shamir shares once a round reaches a terminal state."""
        for cid in client_ids:
            self.models.delete(self._client_key_path(run_id, round_id, cid))
            self.models.delete(self._unmask_shares_path(run_id, round_id, cid))
        self.models.delete(self._unmask_state_key(run_id, round_id))
