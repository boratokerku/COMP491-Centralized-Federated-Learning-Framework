"""web read-only role grants

The website and Grafana connect as `cfl_web`, a role created by deploy/postgres/init with
SELECT-only default privileges. This revision hides the clients' token hashes from that role.
Skipped when the role does not exist (e.g. a developer database without the deploy init).

Revision ID: 0002
Revises: 0001
"""

from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

CLIENT_COLUMNS = (
    "client_id, status, client_version, protocol_version, created_at, registered_at, last_seen_at"
)


def upgrade() -> None:
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'cfl_web') THEN
                GRANT SELECT ON ALL TABLES IN SCHEMA public TO cfl_web;
                REVOKE SELECT ON clients FROM cfl_web;
                GRANT SELECT ({CLIENT_COLUMNS}) ON clients TO cfl_web;
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'cfl_web') THEN
                REVOKE SELECT ON ALL TABLES IN SCHEMA public FROM cfl_web;
            END IF;
        END $$;
        """
    )
