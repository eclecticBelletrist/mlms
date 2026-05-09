"""create facts table

Revision ID: 002
Revises: 001
Create Date: 2026-05-08
"""
from collections.abc import Sequence

from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("""
        CREATE TABLE facts (
            id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            content     TEXT        NOT NULL,
            embedding   VECTOR(1536),
            tags        TEXT[],
            category    TEXT,
            project_id  UUID        REFERENCES projects(id) ON DELETE CASCADE,
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            updated_at  TIMESTAMPTZ,
            version     INT         DEFAULT 1,
            is_current  BOOL        DEFAULT TRUE
        )
    """)

    op.execute("CREATE INDEX ON facts (project_id, is_current)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS facts")
