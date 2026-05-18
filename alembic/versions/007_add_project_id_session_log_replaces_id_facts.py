"""add project_id to session_log, replaces_id to facts

Revision ID: 007
Revises: 006
Create Date: 2026-05-18
"""
from collections.abc import Sequence

from alembic import op

revision: str = "007"
down_revision: str | None = "006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE session_log ADD COLUMN project_id UUID REFERENCES projects(id)")
    op.execute("CREATE INDEX ON session_log (project_id)")
    op.execute("ALTER TABLE facts ADD COLUMN replaces_id UUID REFERENCES facts(id)")


def downgrade() -> None:
    op.execute("ALTER TABLE session_log DROP COLUMN project_id")
    op.execute("ALTER TABLE facts DROP COLUMN replaces_id")
