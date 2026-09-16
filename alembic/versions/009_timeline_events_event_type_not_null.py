"""require event_type on timeline_events

Revision ID: 009
Revises: 008
Create Date: 2026-09-14

003's CHECK (event_type IN (...)) doesn't reject NULL — standard SQL CHECK
semantics pass NULL through. Adding NOT NULL explicitly.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "009"
down_revision: str | None = "008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE timeline_events ALTER COLUMN event_type SET NOT NULL")


def downgrade() -> None:
    op.execute("ALTER TABLE timeline_events ALTER COLUMN event_type DROP NOT NULL")
