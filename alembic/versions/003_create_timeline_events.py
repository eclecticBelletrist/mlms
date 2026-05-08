"""create timeline_events hypertable

Revision ID: 003
Revises: 002
Create Date: 2026-05-08
"""
from collections.abc import Sequence

from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE timeline_events (
            id          UUID        NOT NULL DEFAULT gen_random_uuid(),
            time        TIMESTAMPTZ NOT NULL,
            project_id  UUID        REFERENCES projects(id) ON DELETE CASCADE,
            event_type  TEXT        CHECK (event_type IN ('phase', 'action', 'decision')),
            title       TEXT,
            content     TEXT,
            metadata    JSONB
        )
    """)

    op.execute(
        "SELECT create_hypertable('timeline_events', 'time',"
        " if_not_exists => TRUE)"
    )
    # TimescaleDB requires partition column in any unique constraint
    op.execute("ALTER TABLE timeline_events ADD PRIMARY KEY (id, time)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS timeline_events")
