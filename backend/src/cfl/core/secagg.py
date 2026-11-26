"""Secure Aggregation (SecAgg+) primitives: double masking and dropout recovery.

Follows Bonawitz et al., "Practical Secure Aggregation for Privacy-Preserving Machine
Learning" (CCS 2017):

1. **Ephemeral key exchange (ECDH / X25519):** each client generates a fresh key pair per
   round; the gateway only relays public keys. Two clients derive the same shared secret.

2. **Double masking:** client i uploads

       ŷ_i = x_i + PRG(b_i) + Σ_{j>i} PRG(s_ij) − Σ_{j<i} PRG(s_ij)   (mod M)

   - pairwise masks PRG(s_ij) cancel when the server sums all uploads;
   - the self mask PRG(b_i) (b_i is a random per-round seed) is removed only after the
     round, from Shamir shares held by the other clients.

3. **Dropout recovery (Shamir k-of-N):** every client secret-shares both its self-mask
   seed b_i and its mask private key s_i. After the upload deadline:
   - for each *surviving* client the server collects shares of b_i (remove self mask);
   - for each *dropped* client it collects shares of s_i (cancel the pairwise masks the
     survivors still carry).
   An honest client never reveals both kinds of share for the same peer
   (``share_to_reveal``). So even a compromised server that falsely claims a client
   dropped gets s_i but not b_i, and the client's update stays hidden behind PRG(b_i).

Guarantees: the server never sees an individual x_i; fewer than k colluding parties learn
nothing about any seed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum

import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.ciphers.modes import CTR
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from cfl.core.privacy import DEFAULT_MODULUS

QuantizedDict = dict[str, torch.Tensor]  # int64 tensors with values in [0, M)

# ---------------------------------------------------------------------------
# 1. Keys and seeds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeyPair:
    """An ephemeral X25519 key pair for one round."""

    private_key: X25519PrivateKey
    public_bytes: bytes  # 32 bytes, raw public key

    @classmethod
    def generate(cls) -> KeyPair:
        sk = X25519PrivateKey.generate()
        return cls(private_key=sk, public_bytes=sk.public_key().public_bytes_raw())


def public_key_from_bytes(raw: bytes) -> X25519PublicKey:
    return X25519PublicKey.from_public_bytes(raw)


def private_key_to_bytes(key: X25519PrivateKey) -> bytes:
    return key.private_bytes_raw()


def private_key_from_bytes(raw: bytes) -> X25519PrivateKey:
    return X25519PrivateKey.from_private_bytes(raw)


def generate_self_mask_seed() -> bytes:
    """Fresh 32-byte seed b_i for the client's self mask (new every round)."""
    return os.urandom(32)


def round_context(run_id: str, round_id: int) -> bytes:
    """Binds every derived mask to one (run, round): masks are never reused."""
    return f"{run_id}|{round_id}".encode()


# ---------------------------------------------------------------------------
# 2. PRG and flattening
# ---------------------------------------------------------------------------


def _derive_pair_seed(sk: X25519PrivateKey, peer_pk: X25519PublicKey, ctx: bytes) -> bytes:
    """ECDH shared secret -> 32-byte PRG seed (HKDF-SHA256), identical on both sides."""
    return HKDF(algorithm=SHA256(), length=32, salt=None, info=b"cfl-secagg-pair-v1|" + ctx).derive(
        sk.exchange(peer_pk)
    )


def _derive_self_seed(seed: bytes, ctx: bytes) -> bytes:
    return HKDF(algorithm=SHA256(), length=32, salt=None, info=b"cfl-secagg-self-v1|" + ctx).derive(
        seed
    )


def expand_mask(seed: bytes, num_elements: int, modulus: int = DEFAULT_MODULUS) -> torch.Tensor:
    """Expand a 32-byte seed into `num_elements` uniform values in [0, modulus).

    AES-256-CTR keystream as a cryptographic PRG; a fixed nonce is safe because each seed
    is unique per (pair or client, round). `modulus` must be a power of two so that
    reducing 64 random bits modulo it stays uniform.
    """
    if modulus & (modulus - 1):
        raise ValueError("modulus must be a power of two")
    encryptor = Cipher(algorithms.AES256(seed), CTR(b"\x00" * 16)).encryptor()
    stream = encryptor.update(b"\x00" * (num_elements * 8)) + encryptor.finalize()
    values = torch.frombuffer(bytearray(stream), dtype=torch.int64).clone()
    return torch.remainder(values, modulus)  # floor-mod: always non-negative


def flatten_state_dict(qd: QuantizedDict) -> tuple[torch.Tensor, list[str], list[torch.Size]]:
    """One 1-D tensor in sorted-name order, so every party lays out masks identically."""
    names = sorted(qd)
    shapes = [qd[n].shape for n in names]
    return torch.cat([qd[n].reshape(-1) for n in names]), names, shapes


def unflatten_state_dict(
    flat: torch.Tensor, names: list[str], shapes: list[torch.Size]
) -> QuantizedDict:
    result: QuantizedDict = {}
    offset = 0
    for name, shape in zip(names, shapes, strict=True):
        numel = shape.numel()
        result[name] = flat[offset : offset + numel].reshape(shape)
        offset += numel
    return result


# ---------------------------------------------------------------------------
# 3. Client side: masking
# ---------------------------------------------------------------------------


def apply_masks(
    quantized: QuantizedDict,
    my_id: str,
    my_keypair: KeyPair,
    peer_ids: list[str],
    peer_public_keys: dict[str, bytes],
    ctx: bytes,
    self_mask_seed: bytes | None = None,
    modulus: int = DEFAULT_MODULUS,
) -> QuantizedDict:
    """ŷ_i = x_i + PRG(b_i) + Σ_{j>i} PRG(s_ij) − Σ_{j<i} PRG(s_ij)  (mod M).

    `self_mask_seed` (b_i) should always be given in production; without it a server that
    recovers this client's private key could unmask its update.
    """
    flat, names, shapes = flatten_state_dict(quantized)
    flat = torch.remainder(flat, modulus)
    n = flat.numel()

    if self_mask_seed is not None:
        flat = (flat + expand_mask(_derive_self_seed(self_mask_seed, ctx), n, modulus)) % modulus

    for peer_id in peer_ids:
        if peer_id == my_id:
            continue
        seed = _derive_pair_seed(
            my_keypair.private_key, public_key_from_bytes(peer_public_keys[peer_id]), ctx
        )
        mask = expand_mask(seed, n, modulus)
        flat = (flat + mask) % modulus if my_id < peer_id else (flat - mask) % modulus

    return unflatten_state_dict(flat, names, shapes)


class ShareKind(StrEnum):
    SELF_MASK = "self_mask"  # share of b_j: peer j survived
    MASK_KEY = "mask_key"  # share of s_j: peer j dropped


def share_to_reveal(peer_id: str, surviving: set[str], dropped: set[str]) -> ShareKind:
    """Honest-client rule in the unmasking step: reveal b_j for survivors and s_j for
    dropouts, never both for the same peer. Refuses an inconsistent server request."""
    if peer_id in surviving and peer_id in dropped:
        raise ValueError(f"{peer_id} cannot be both surviving and dropped")
    if peer_id in surviving:
        return ShareKind.SELF_MASK
    if peer_id in dropped:
        return ShareKind.MASK_KEY
    raise ValueError(f"{peer_id} is not a participant of this round")


# ---------------------------------------------------------------------------
# 4. Server side: unmasking the aggregate
# ---------------------------------------------------------------------------


def sum_masked(
    masked_updates: list[QuantizedDict],
    modulus: int = DEFAULT_MODULUS,
) -> QuantizedDict:
    """Σ ŷ_i (mod M). The server only ever handles this sum, never an unmasked x_i."""
    if not masked_updates:
        raise ValueError("cannot sum an empty list of masked updates")
    return {
        name: torch.remainder(sum(u[name] for u in masked_updates), modulus)
        for name in masked_updates[0]
    }


def unmask_aggregate(
    aggregated: QuantizedDict,
    ctx: bytes,
    surviving_self_seeds: dict[str, bytes],
    dropped_private_keys: dict[str, X25519PrivateKey] | None = None,
    surviving_public_keys: dict[str, bytes] | None = None,
    modulus: int = DEFAULT_MODULUS,
) -> QuantizedDict:
    """Remove the remaining masks from Σ ŷ_i over the surviving clients.

    - subtract PRG(b_i) for every survivor (seeds reconstructed from Shamir shares);
    - for every dropped client d and survivor j, cancel the pairwise mask j still carries:
      j added PRG(s_jd) if j < d (subtract it), subtracted it if j > d (add it back).
    Result: Σ x_i over survivors (mod M).
    """
    flat, names, shapes = flatten_state_dict(aggregated)
    n = flat.numel()

    for seed in surviving_self_seeds.values():
        flat = (flat - expand_mask(_derive_self_seed(seed, ctx), n, modulus)) % modulus

    for dropped_id, dropped_sk in (dropped_private_keys or {}).items():
        if surviving_public_keys is None:
            raise ValueError("surviving_public_keys are needed to cancel dropout masks")
        for sid, pk_bytes in surviving_public_keys.items():
            mask = expand_mask(
                _derive_pair_seed(dropped_sk, public_key_from_bytes(pk_bytes), ctx), n, modulus
            )
            flat = (flat - mask) % modulus if sid < dropped_id else (flat + mask) % modulus

    return unflatten_state_dict(flat, names, shapes)


# ---------------------------------------------------------------------------
# 5. Shamir Secret Sharing over GF(p)
# ---------------------------------------------------------------------------

_SHAMIR_PRIME = (1 << 256) - 189  # largest prime below 2^256


@dataclass(frozen=True)
class ShamirShare:
    x: int  # evaluation point, 1-based
    y: int


def split_secret(secret: bytes, n: int, k: int) -> list[ShamirShare]:
    """Split a 32-byte secret into `n` shares; any `k` reconstruct it, k−1 reveal nothing."""
    if not (2 <= k <= n):
        raise ValueError(f"need 2 <= k <= n, got k={k}, n={n}")
    if len(secret) != 32:
        raise ValueError(f"secret must be 32 bytes, got {len(secret)}")
    s = int.from_bytes(secret, "big")
    p = _SHAMIR_PRIME
    if s >= p:
        raise ValueError("secret does not fit the field (regenerate it)")

    coeffs = [s] + [int.from_bytes(os.urandom(32), "big") % p for _ in range(k - 1)]
    shares = []
    for x in range(1, n + 1):
        y = 0
        for c in reversed(coeffs):  # Horner
            y = (y * x + c) % p
        shares.append(ShamirShare(x=x, y=y))
    return shares


def reconstruct_secret(shares: list[ShamirShare]) -> bytes:
    """Lagrange interpolation at 0. Needs at least k distinct shares."""
    if len(shares) < 2:
        raise ValueError("need at least 2 shares")
    if len({s.x for s in shares}) != len(shares):
        raise ValueError("duplicate share indices")
    p = _SHAMIR_PRIME
    secret = 0
    for i, si in enumerate(shares):
        num, den = 1, 1
        for j, sj in enumerate(shares):
            if i != j:
                num = (num * -sj.x) % p
                den = (den * (si.x - sj.x)) % p
        secret = (secret + si.y * num * pow(den, p - 2, p)) % p
    return secret.to_bytes(32, "big")
