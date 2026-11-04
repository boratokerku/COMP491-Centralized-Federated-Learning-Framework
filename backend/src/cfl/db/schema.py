"""SQLAlchemy table definitions for planning/db-schema.md.

Alembic migrations (backend/migrations) create these tables; this module is the single
definition the coordinator, gateway and report generator query. Columns that are not in
db-schema.md yet are marked "addition".
"""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    Double,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

metadata = MetaData(
    naming_convention={
        "pk": "pk_%(table_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "ix": "ix_%(table_name)s_%(column_0_label)s",
    }
)

SHA256 = String(64)
TS = DateTime(timezone=True)
UUID_STR = UUID(as_uuid=False)

clients = Table(
    "clients",
    metadata,
    Column("client_id", Text, primary_key=True),
    Column("token_hash", Text, nullable=False, unique=True),
    Column("status", Text, nullable=False),
    Column("client_version", Text),
    Column("protocol_version", Text),
    Column("created_at", TS, nullable=False, server_default=func.now()),
    Column("registered_at", TS),
    Column("last_seen_at", TS),
    CheckConstraint("status IN ('APPROVED','REGISTERED','DISABLED')", name="status"),
)

runs = Table(
    "runs",
    metadata,
    Column("run_id", UUID_STR, primary_key=True),
    Column("name", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("strategy", Text, nullable=False),
    Column("task_name", Text, nullable=False),
    Column("seed", BigInteger, nullable=False),
    Column("config", JSONB, nullable=False),
    Column("config_sha256", SHA256, nullable=False),  # addition
    Column("protocol_version", Text, nullable=False),
    Column("model_version", Integer, nullable=False, server_default="0"),
    Column("model_path", Text),
    Column("model_sha256", SHA256),
    Column("manifest_version", Text, nullable=False),
    Column("manifest_sha256", SHA256, nullable=False),  # addition
    Column("preprocessing_version", Text, nullable=False),
    Column("model_architecture", Text, nullable=False),
    Column("model_architecture_version", Text, nullable=False),
    Column("model_fingerprint", SHA256, nullable=False),  # addition
    Column("started_at", TS),
    Column("completed_at", TS),
    Column("failure_reason", Text),
    Column("final_model_path", Text),
    Column("final_model_sha256", SHA256),
    Column("final_model_format", Text),
    Column("final_model_bytes", BigInteger),
    Column("created_at", TS, nullable=False, server_default=func.now()),
    CheckConstraint(
        "status IN ('PENDING','RUNNING','COMPLETED','FAILED','ABORTED')", name="status"
    ),
    CheckConstraint("model_version >= 0", name="model_version"),
    Index("ix_runs_status", "status"),
)

rounds = Table(
    "rounds",
    metadata,
    Column("run_id", UUID_STR, ForeignKey("runs.run_id"), primary_key=True),
    Column("round_id", Integer, primary_key=True),
    Column("state", Text, nullable=False),
    Column("input_model_version", Integer, nullable=False),
    Column("input_model_path", Text, nullable=False),
    Column("input_model_sha256", SHA256, nullable=False),
    Column("output_model_version", Integer),
    Column("output_model_path", Text),
    Column("output_model_sha256", SHA256),
    Column("min_participants", Integer, nullable=False),
    Column("deadline_at", TS, nullable=False),
    Column("opened_at", TS, nullable=False),
    Column("closed_at", TS),
    Column("failure_reason", Text),
    Column("state_version", BigInteger, nullable=False, server_default="0"),
    Column("created_at", TS, nullable=False, server_default=func.now()),
    CheckConstraint("round_id > 0", name="round_id"),
    CheckConstraint(
        "state IN ('OPEN','COLLECTING','AGGREGATING','DONE','FAILED')", name="state"
    ),
    CheckConstraint("min_participants > 0", name="min_participants"),
    CheckConstraint("state <> 'FAILED' OR failure_reason IS NOT NULL", name="failed_reason"),
    Index("ix_rounds_run_latest", "run_id", "round_id"),
    Index("ix_rounds_state_deadline", "state", "deadline_at"),
)

round_participants = Table(
    "round_participants",
    metadata,
    Column("run_id", UUID_STR, primary_key=True),
    Column("round_id", Integer, primary_key=True),
    Column("client_id", Text, ForeignKey("clients.client_id"), primary_key=True),
    Column("status", Text, nullable=False),
    Column("selected_at", TS, nullable=False, server_default=func.now()),
    Column("finished_at", TS),
    Column("reason_code", Text),
    ForeignKeyConstraint(["run_id", "round_id"], ["rounds.run_id", "rounds.round_id"]),
    CheckConstraint(
        "status IN ('SELECTED','UPDATE_ACCEPTED','DROPOUT','REJECTED')", name="status"
    ),
)

updates = Table(
    "updates",
    metadata,
    Column("update_id", UUID_STR, primary_key=True),
    Column("run_id", UUID_STR, nullable=False),
    Column("round_id", Integer, nullable=False),
    Column("client_id", Text, nullable=False),
    Column("model_version", Integer, nullable=False),
    Column("num_samples", Integer, nullable=False),
    Column("weights_path", Text, nullable=False),
    Column("checksum_sha256", SHA256, nullable=False),
    Column("payload_bytes", BigInteger, nullable=False),
    Column("metrics", JSONB, nullable=False, server_default="{}"),
    Column("received_at", TS, nullable=False, server_default=func.now()),
    Column("upload_duration_ms", Double),
    UniqueConstraint("run_id", "round_id", "client_id", name="uq_updates_run_round_client"),
    ForeignKeyConstraint(
        ["run_id", "round_id", "client_id"],
        [
            "round_participants.run_id",
            "round_participants.round_id",
            "round_participants.client_id",
        ],
        name="fk_updates_participant",
    ),
    CheckConstraint("num_samples > 0", name="num_samples"),
    CheckConstraint("payload_bytes >= 0", name="payload_bytes"),
    CheckConstraint("checksum_sha256 ~ '^[0-9a-f]{64}$'", name="checksum"),
    Index("ix_updates_client_received", "client_id", "received_at"),
)

conformance_reports = Table(
    "conformance_reports",
    metadata,
    Column("report_id", UUID_STR, primary_key=True),
    Column("client_id", Text, ForeignKey("clients.client_id"), nullable=False),
    Column("manifest_version", Text, nullable=False),
    Column("client_version", Text, nullable=False),
    Column("protocol_version", Text, nullable=False),
    Column("status", Text, nullable=False),
    Column("reason_code", Text),
    Column("report_payload", JSONB, nullable=False),
    Column("checked_at", TS, nullable=False, server_default=func.now()),
    CheckConstraint("status IN ('PASSED','FAILED')", name="status"),
    Index("ix_conformance_latest", "client_id", "manifest_version", "checked_at"),
)

events = Table(
    "events",
    metadata,
    Column("event_id", BigInteger, primary_key=True, autoincrement=True),
    Column("run_id", UUID_STR, ForeignKey("runs.run_id")),
    Column("round_id", Integer),
    Column("client_id", Text, ForeignKey("clients.client_id")),
    Column("event_type", Text, nullable=False),
    Column("message_type", Text),
    Column("status", Text),
    Column("reason_code", Text),
    Column("occurred_at", TS, nullable=False, server_default=func.now()),
    Column("duration_ms", Double),
    Column("message_bytes", BigInteger),
    Column("details", JSONB, nullable=False, server_default="{}"),
    ForeignKeyConstraint(["run_id", "round_id"], ["rounds.run_id", "rounds.round_id"]),
    CheckConstraint("message_bytes IS NULL OR message_bytes >= 0", name="message_bytes"),
    Index("ix_events_run_time", "run_id", "occurred_at"),
    Index("ix_events_run_round_time", "run_id", "round_id", "occurred_at"),
    Index("ix_events_client_time", "client_id", "occurred_at"),
)

metrics = Table(
    "metrics",
    metadata,
    Column("metric_id", BigInteger, primary_key=True, autoincrement=True),
    Column("run_id", UUID_STR, ForeignKey("runs.run_id"), nullable=False),
    Column("round_id", Integer),
    Column("client_id", Text, ForeignKey("clients.client_id")),
    Column("metric_name", Text, nullable=False),
    Column("value", Double, nullable=False),
    Column("unit", Text, nullable=False),
    Column("message_type", Text),
    Column("dimensions", JSONB, nullable=False, server_default="{}"),
    Column("recorded_at", TS, nullable=False, server_default=func.now()),
    ForeignKeyConstraint(["run_id", "round_id"], ["rounds.run_id", "rounds.round_id"]),
    Index("ix_metrics_run_name_time", "run_id", "metric_name", "recorded_at"),
    Index("ix_metrics_run_round_name", "run_id", "round_id", "metric_name"),
)
