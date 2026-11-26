"""Pydantic message models for the gateway API and client protocol.

Follows planning/api-spec.md and the report contract in planning/conformance-manifest.md.
Additions to the spec are marked "addition"; they are listed in planning/interfaces.md.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from cfl.core.secagg import ShamirShare
from cfl.protocol import PROTOCOL_VERSION

# -- 1. Registration ----------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    client_id: str
    client_version: str
    protocol_version: str = PROTOCOL_VERSION


class RegisterResponse(BaseModel):
    accepted: bool
    client_id: str | None = None
    status: str | None = None  # "REGISTERED"
    manifest_version: str | None = None  # manifest of the running run, if exactly one
    protocol_version: str = PROTOCOL_VERSION
    reason: str | None = None
    message: str | None = None


# -- 2. Conformance -----------------------------------------------------------------------------


class ManifestSummary(BaseModel):  # addition: GET /conformance lists the active manifests
    manifest_version: str  # "<dataset>@<version>", e.g. "bank_marketing@1.0.0"
    dataset: str
    manifest_sha256: str


class ModelParameterSpec(BaseModel):
    name: str
    shape: list[int]
    dtype: str = "float32"


class DeclaredModel(BaseModel):
    architecture_id: str
    architecture_version: str
    fingerprint: str | None = None  # addition: checked when given
    parameters: list[ModelParameterSpec] = Field(default_factory=list)


class DeclaredInput(BaseModel):
    feature_names: list[str]  # raw input features, in order
    feature_types: list[str]  # numeric | categorical | text
    # Per feature: the numeric unit, "category" for categorical, "text" for text features.
    feature_units: list[str]
    preprocessing_version: str
    categories: dict[str, list[str]] = Field(default_factory=dict)  # addition: categorical vocab
    tokenizer: dict[str, Any] | None = None  # addition: text tasks, = preprocessing.tokenizer


class DeclaredLabel(BaseModel):  # addition: label semantics and range
    column: str
    positive_value: str
    negative_value: str
    encoded_values: list[int]


class ReferenceTestOutput(BaseModel):
    test_id: str
    output_features: list[float]


class ForwardTestOutput(BaseModel):  # addition: manifest forward_reference_tests
    test_id: str
    logits: list[float]


class ConformanceCheck(BaseModel):
    code: str
    passed: bool


class ConformanceReportRequest(BaseModel):
    manifest_version: str
    manifest_sha256: str
    client_id: str
    client_version: str
    protocol_version: str = PROTOCOL_VERSION
    declared_model: DeclaredModel
    declared_input: DeclaredInput
    declared_label: DeclaredLabel | None = None
    client_passed: bool  # informational only; the gateway decides
    checks: list[ConformanceCheck] = Field(default_factory=list)
    reference_outputs: list[ReferenceTestOutput] = Field(default_factory=list)
    forward_outputs: list[ForwardTestOutput] = Field(default_factory=list)
    client_failure_codes: list[str] = Field(default_factory=list)


class ConformanceReportResponse(BaseModel):
    status: str  # "PASSED" or "FAILED"
    client_id: str
    manifest_version: str
    report_id: str | None = None
    reason_code: str | None = None
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    checked_at: datetime | None = None


# -- 3. Rounds and updates ----------------------------------------------------------------------


class TaskAssignment(BaseModel):
    run_id: str
    round_id: int
    model_version: int
    deadline: datetime
    config: dict[str, Any] = Field(default_factory=dict)
    # Per-round privacy / SecAgg+ parameters (None when privacy is disabled).
    clip_norm: float | None = Field(default=None, gt=0)
    sigma_t: float | None = Field(default=None, ge=0)
    trust_weight: float | None = Field(default=None, gt=0, le=1)
    secagg_threshold: int | None = Field(default=None, ge=2)


def _validate_x25519_hex(value: str) -> str:
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("public_key_hex must be valid hexadecimal") from exc
    if len(raw) != 32:
        raise ValueError(f"X25519 public key must be 32 bytes (64 hex chars), got {len(raw)} bytes")
    return raw.hex()


class ShamirSharePayload(BaseModel):
    """JSON-safe encoding of a Shamir share (256-bit y as hex to avoid float precision loss)."""

    x: int = Field(ge=1)
    y_hex: str

    @field_validator("y_hex")
    @classmethod
    def _valid_hex(cls, value: str) -> str:
        if not value:
            raise ValueError("y_hex must not be empty")
        try:
            n = int(value, 16)
        except ValueError as exc:
            raise ValueError("y_hex must be valid hexadecimal") from exc
        if n < 0:
            raise ValueError("y_hex must be non-negative")
        return value.lower()

    @classmethod
    def from_share(cls, share: ShamirShare) -> ShamirSharePayload:
        return cls(x=share.x, y_hex=format(share.y, "x"))

    def to_share(self) -> ShamirShare:
        return ShamirShare(x=self.x, y=int(self.y_hex, 16))


class PeerKeyEntry(BaseModel):
    client_id: str = Field(min_length=1)
    public_key_hex: str

    @field_validator("public_key_hex")
    @classmethod
    def _valid_pubkey(cls, value: str) -> str:
        return _validate_x25519_hex(value)

    def public_key_bytes(self) -> bytes:
        return bytes.fromhex(self.public_key_hex)


class RoundKeySubmit(BaseModel):
    """Client submits its ephemeral X25519 public key (and optional peer-encrypted shares)."""

    run_id: str
    round_id: int = Field(ge=1)
    client_id: str = Field(min_length=1)
    public_key_hex: str
    # Optional map: recipient_client_id -> opaque hex ciphertext of Shamir shares for that peer.
    encrypted_shares: dict[str, str] = Field(default_factory=dict)

    @field_validator("public_key_hex")
    @classmethod
    def _valid_pubkey(cls, value: str) -> str:
        return _validate_x25519_hex(value)


class RoundKeysResponse(BaseModel):
    """Gateway returns peer public keys for a SecAgg+ round once key exchange is ready."""

    run_id: str
    round_id: int = Field(ge=1)
    ready: bool
    secagg_threshold: int = Field(ge=2)
    peers: list[PeerKeyEntry] = Field(default_factory=list)
    # Optional map: sender_client_id -> opaque hex ciphertext addressed to the requesting client.
    incoming_shares: dict[str, str] = Field(default_factory=dict)


class RevealedShare(BaseModel):
    """One Shamir share revealed during unmasking: self_mask (survivor) or mask_key (dropout)."""

    target_client_id: str = Field(min_length=1)
    kind: Literal["self_mask", "mask_key"]
    share: ShamirSharePayload


class UnmaskSharesRequest(BaseModel):
    """Surviving client submits Shamir shares to unmask the round aggregate."""

    run_id: str
    round_id: int = Field(ge=1)
    client_id: str = Field(min_length=1)
    shares: list[RevealedShare] = Field(default_factory=list)

    @model_validator(mode="after")
    def _never_reveal_both_kinds_for_same_peer(self) -> UnmaskSharesRequest:
        seen_kind: dict[str, str] = {}
        for item in self.shares:
            prev = seen_kind.get(item.target_client_id)
            if prev is not None:
                if prev != item.kind:
                    raise ValueError(
                        f"cannot reveal both self_mask and mask_key for {item.target_client_id}"
                    )
                raise ValueError(f"duplicate {item.kind} share for {item.target_client_id}")
            seen_kind[item.target_client_id] = item.kind
        return self


class UnmaskSharesResponse(BaseModel):
    accepted: bool
    round_id: int = Field(ge=1)
    reason: str | None = None


class UpdateMetadata(BaseModel):
    run_id: str
    round_id: int
    client_id: str
    update_id: str  # UUIDv4 idempotency key generated by the client; retries reuse it
    model_version: int
    num_samples: int
    checksum_sha256: str
    metrics: dict[str, float] = Field(default_factory=dict)


class UpdateAck(BaseModel):
    accepted: bool
    update_id: str | None = None
    client_id: str | None = None
    round_id: int | None = None
    reason: str | None = None  # None, "duplicate_idempotent", "duplicate_update", ...


class TransferReport(BaseModel):  # addition: POST /transfer
    """How long a transfer took as the client saw it. Only the client can tell, because the
    gateway never sees the bytes cross the network."""

    run_id: str
    round_id: int | None = None
    message_type: Literal[
        "register",
        "conformance",
        "task",
        "model_download",
        "update_upload",
        "final_model_download",
        "secagg_keys",
        "secagg_unmask",
    ]
    duration_ms: float = Field(ge=0)
    message_bytes: int | None = Field(default=None, ge=0)  # request + response body


# -- 4. Errors ----------------------------------------------------------------------------------


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
