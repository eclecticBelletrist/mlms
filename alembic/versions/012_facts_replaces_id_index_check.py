"""index + self-reference guard on facts.replaces_id

Revision ID: 012
Revises: 011
Create Date: 2026-09-14

007 added facts.replaces_id with no index (any lookup by replaces_id does a
seq scan) and no guard against a row pointing at itself. No ON DELETE clause
change here — no hard-delete path exists on facts, so that's out of scope.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "012"
down_revision: str | None = "011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE INDEX ON facts (replaces_id)")
    op.execute(
        "ALTER TABLE facts ADD CONSTRAINT facts_replaces_id_not_self "
        "CHECK (replaces_id IS NULL OR replaces_id <> id)"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE facts DROP CONSTRAINT facts_replaces_id_not_self")
    op.execute("DROP INDEX IF EXISTS facts_replaces_id_idx")
