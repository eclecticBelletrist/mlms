"""create skills table

Revision ID: 005
Revises: 004
Create Date: 2026-05-08
"""
from collections.abc import Sequence

from alembic import op

revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE skills (
            id                 UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            name               TEXT        NOT NULL,
            domain             TEXT,
            content            TEXT        NOT NULL,
            trigger_conditions JSONB,
            version            INT         DEFAULT 1,
            created_at         TIMESTAMPTZ DEFAULT NOW()
        )
    """)
    op.execute("CREATE INDEX ON skills (name)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS skills")
