"""create projects + enable extensions

Revision ID: 001
Revises:
Create Date: 2026-05-08
"""

from collections.abc import Sequence

from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

    op.execute("""
        CREATE TABLE projects (
            id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            name        TEXT        NOT NULL,
            description TEXT,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        )
    """)


def downgrade() -> None:
    # Plain sequential downgrade (007 -> ... -> 001) already drops facts/timeline_events/
    # session_log — the FK-holding tables — in their own migrations first, so projects has
    # no dependents left by the time this runs. No CASCADE: a footgun here would silently
    # drop whatever still references projects instead of failing loudly on a bad downgrade order.
    op.execute("DROP TABLE IF EXISTS projects")
    op.execute("DROP EXTENSION IF EXISTS vector")
    op.execute("DROP EXTENSION IF EXISTS timescaledb")
