"""add id column + composite PK to session_log

Revision ID: 008
Revises: 007
Create Date: 2026-09-14

session_log has no PK/UNIQUE at all (004-007). `time` comes from
datetime.now(UTC) in storage/session_log.py by default, but callers can also
pass an explicit `time` — neither path guarantees uniqueness per chat_id
(concurrent inserts, or two explicit-time callers, can collide). A composite
PK(chat_id, time) would be unsafe. Falling back to a surrogate id, same
pattern as timeline_events (003): TimescaleDB requires the partition column
in any PK, hence PK(id, time) rather than PK(id) alone.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "008"
down_revision: str | None = "007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE session_log ADD COLUMN id UUID NOT NULL DEFAULT gen_random_uuid()")
    op.execute("ALTER TABLE session_log ADD PRIMARY KEY (id, time)")


def downgrade() -> None:
    op.execute("ALTER TABLE session_log DROP CONSTRAINT session_log_pkey")
    op.execute("ALTER TABLE session_log DROP COLUMN id")
