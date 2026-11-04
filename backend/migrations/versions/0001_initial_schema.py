"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-11-04 17:15:00.832611
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:

    op.create_table(
        "clients",
        sa.Column("client_id", sa.Text(), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("client_version", sa.Text(), nullable=True),
        sa.Column("protocol_version", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("registered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('APPROVED','REGISTERED','DISABLED')", name=op.f("ck_clients_status")
        ),
        sa.PrimaryKeyConstraint("client_id", name=op.f("pk_clients")),
        sa.UniqueConstraint("token_hash", name=op.f("uq_clients_token_hash")),
    )
    op.create_table(
        "runs",
        sa.Column("run_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("strategy", sa.Text(), nullable=False),
        sa.Column("task_name", sa.Text(), nullable=False),
        sa.Column("seed", sa.BigInteger(), nullable=False),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("config_sha256", sa.String(length=64), nullable=False),
        sa.Column("protocol_version", sa.Text(), nullable=False),
        sa.Column("model_version", sa.Integer(), server_default="0", nullable=False),
        sa.Column("model_path", sa.Text(), nullable=True),
        sa.Column("model_sha256", sa.String(length=64), nullable=True),
        sa.Column("manifest_version", sa.Text(), nullable=False),
        sa.Column("manifest_sha256", sa.String(length=64), nullable=False),
        sa.Column("preprocessing_version", sa.Text(), nullable=False),
        sa.Column("model_architecture", sa.Text(), nullable=False),
        sa.Column("model_architecture_version", sa.Text(), nullable=False),
        sa.Column("model_fingerprint", sa.String(length=64), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("final_model_path", sa.Text(), nullable=True),
        sa.Column("final_model_sha256", sa.String(length=64), nullable=True),
        sa.Column("final_model_format", sa.Text(), nullable=True),
        sa.Column("final_model_bytes", sa.BigInteger(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('PENDING','RUNNING','COMPLETED','FAILED','ABORTED')",
            name=op.f("ck_runs_status"),
        ),
        sa.CheckConstraint("model_version >= 0", name=op.f("ck_runs_model_version")),
        sa.PrimaryKeyConstraint("run_id", name=op.f("pk_runs")),
    )
    op.create_index("ix_runs_status", "runs", ["status"], unique=False)
    op.create_table(
        "conformance_reports",
        sa.Column("report_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("client_id", sa.Text(), nullable=False),
        sa.Column("manifest_version", sa.Text(), nullable=False),
        sa.Column("client_version", sa.Text(), nullable=False),
        sa.Column("protocol_version", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("reason_code", sa.Text(), nullable=True),
        sa.Column("report_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "checked_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('PASSED','FAILED')", name=op.f("ck_conformance_reports_status")
        ),
        sa.ForeignKeyConstraint(
            ["client_id"],
            ["clients.client_id"],
            name=op.f("fk_conformance_reports_client_id_clients"),
        ),
        sa.PrimaryKeyConstraint("report_id", name=op.f("pk_conformance_reports")),
    )
    op.create_index(
        "ix_conformance_latest",
        "conformance_reports",
        ["client_id", "manifest_version", "checked_at"],
        unique=False,
    )
    op.create_table(
        "rounds",
        sa.Column("run_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("round_id", sa.Integer(), nullable=False),
        sa.Column("state", sa.Text(), nullable=False),
        sa.Column("input_model_version", sa.Integer(), nullable=False),
        sa.Column("input_model_path", sa.Text(), nullable=False),
        sa.Column("input_model_sha256", sa.String(length=64), nullable=False),
        sa.Column("output_model_version", sa.Integer(), nullable=True),
        sa.Column("output_model_path", sa.Text(), nullable=True),
        sa.Column("output_model_sha256", sa.String(length=64), nullable=True),
        sa.Column("min_participants", sa.Integer(), nullable=False),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("state_version", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "state <> 'FAILED' OR failure_reason IS NOT NULL", name=op.f("ck_rounds_failed_reason")
        ),
        sa.CheckConstraint(
            "state IN ('OPEN','COLLECTING','AGGREGATING','DONE','FAILED')",
            name=op.f("ck_rounds_state"),
        ),
        sa.CheckConstraint("min_participants > 0", name=op.f("ck_rounds_min_participants")),
        sa.CheckConstraint("round_id > 0", name=op.f("ck_rounds_round_id")),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], name=op.f("fk_rounds_run_id_runs")),
        sa.PrimaryKeyConstraint("run_id", "round_id", name=op.f("pk_rounds")),
    )
    op.create_index("ix_rounds_run_latest", "rounds", ["run_id", "round_id"], unique=False)
    op.create_index("ix_rounds_state_deadline", "rounds", ["state", "deadline_at"], unique=False)
    op.create_table(
        "events",
        sa.Column("event_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.UUID(as_uuid=False), nullable=True),
        sa.Column("round_id", sa.Integer(), nullable=True),
        sa.Column("client_id", sa.Text(), nullable=True),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("message_type", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=True),
        sa.Column("reason_code", sa.Text(), nullable=True),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("duration_ms", sa.Double(), nullable=True),
        sa.Column("message_bytes", sa.BigInteger(), nullable=True),
        sa.Column(
            "details", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False
        ),
        sa.CheckConstraint(
            "message_bytes IS NULL OR message_bytes >= 0", name=op.f("ck_events_message_bytes")
        ),
        sa.ForeignKeyConstraint(
            ["client_id"], ["clients.client_id"], name=op.f("fk_events_client_id_clients")
        ),
        sa.ForeignKeyConstraint(
            ["run_id", "round_id"],
            ["rounds.run_id", "rounds.round_id"],
            name=op.f("fk_events_run_id_rounds"),
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], name=op.f("fk_events_run_id_runs")),
        sa.PrimaryKeyConstraint("event_id", name=op.f("pk_events")),
    )
    op.create_index("ix_events_client_time", "events", ["client_id", "occurred_at"], unique=False)
    op.create_index(
        "ix_events_run_round_time", "events", ["run_id", "round_id", "occurred_at"], unique=False
    )
    op.create_index("ix_events_run_time", "events", ["run_id", "occurred_at"], unique=False)
    op.create_table(
        "metrics",
        sa.Column("metric_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("round_id", sa.Integer(), nullable=True),
        sa.Column("client_id", sa.Text(), nullable=True),
        sa.Column("metric_name", sa.Text(), nullable=False),
        sa.Column("value", sa.Double(), nullable=False),
        sa.Column("unit", sa.Text(), nullable=False),
        sa.Column("message_type", sa.Text(), nullable=True),
        sa.Column(
            "dimensions",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default="{}",
            nullable=False,
        ),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["client_id"], ["clients.client_id"], name=op.f("fk_metrics_client_id_clients")
        ),
        sa.ForeignKeyConstraint(
            ["run_id", "round_id"],
            ["rounds.run_id", "rounds.round_id"],
            name=op.f("fk_metrics_run_id_rounds"),
        ),
        sa.ForeignKeyConstraint(["run_id"], ["runs.run_id"], name=op.f("fk_metrics_run_id_runs")),
        sa.PrimaryKeyConstraint("metric_id", name=op.f("pk_metrics")),
    )
    op.create_index(
        "ix_metrics_run_name_time",
        "metrics",
        ["run_id", "metric_name", "recorded_at"],
        unique=False,
    )
    op.create_index(
        "ix_metrics_run_round_name", "metrics", ["run_id", "round_id", "metric_name"], unique=False
    )
    op.create_table(
        "round_participants",
        sa.Column("run_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("round_id", sa.Integer(), nullable=False),
        sa.Column("client_id", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "selected_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reason_code", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('SELECTED','UPDATE_ACCEPTED','DROPOUT','REJECTED')",
            name=op.f("ck_round_participants_status"),
        ),
        sa.ForeignKeyConstraint(
            ["client_id"],
            ["clients.client_id"],
            name=op.f("fk_round_participants_client_id_clients"),
        ),
        sa.ForeignKeyConstraint(
            ["run_id", "round_id"],
            ["rounds.run_id", "rounds.round_id"],
            name=op.f("fk_round_participants_run_id_rounds"),
        ),
        sa.PrimaryKeyConstraint(
            "run_id", "round_id", "client_id", name=op.f("pk_round_participants")
        ),
    )
    op.create_table(
        "updates",
        sa.Column("update_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("run_id", sa.UUID(as_uuid=False), nullable=False),
        sa.Column("round_id", sa.Integer(), nullable=False),
        sa.Column("client_id", sa.Text(), nullable=False),
        sa.Column("model_version", sa.Integer(), nullable=False),
        sa.Column("num_samples", sa.Integer(), nullable=False),
        sa.Column("weights_path", sa.Text(), nullable=False),
        sa.Column("checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("payload_bytes", sa.BigInteger(), nullable=False),
        sa.Column(
            "metrics", postgresql.JSONB(astext_type=sa.Text()), server_default="{}", nullable=False
        ),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("upload_duration_ms", sa.Double(), nullable=True),
        sa.CheckConstraint("checksum_sha256 ~ '^[0-9a-f]{64}$'", name=op.f("ck_updates_checksum")),
        sa.CheckConstraint("num_samples > 0", name=op.f("ck_updates_num_samples")),
        sa.CheckConstraint("payload_bytes >= 0", name=op.f("ck_updates_payload_bytes")),
        sa.ForeignKeyConstraint(
            ["run_id", "round_id", "client_id"],
            [
                "round_participants.run_id",
                "round_participants.round_id",
                "round_participants.client_id",
            ],
            name="fk_updates_participant",
        ),
        sa.PrimaryKeyConstraint("update_id", name=op.f("pk_updates")),
        sa.UniqueConstraint("run_id", "round_id", "client_id", name="uq_updates_run_round_client"),
    )
    op.create_index(
        "ix_updates_client_received", "updates", ["client_id", "received_at"], unique=False
    )


def downgrade() -> None:

    op.drop_index("ix_updates_client_received", table_name="updates")
    op.drop_table("updates")
    op.drop_table("round_participants")
    op.drop_index("ix_metrics_run_round_name", table_name="metrics")
    op.drop_index("ix_metrics_run_name_time", table_name="metrics")
    op.drop_table("metrics")
    op.drop_index("ix_events_run_time", table_name="events")
    op.drop_index("ix_events_run_round_time", table_name="events")
    op.drop_index("ix_events_client_time", table_name="events")
    op.drop_table("events")
    op.drop_index("ix_rounds_state_deadline", table_name="rounds")
    op.drop_index("ix_rounds_run_latest", table_name="rounds")
    op.drop_table("rounds")
    op.drop_index("ix_conformance_latest", table_name="conformance_reports")
    op.drop_table("conformance_reports")
    op.drop_index("ix_runs_status", table_name="runs")
    op.drop_table("runs")
    op.drop_table("clients")
