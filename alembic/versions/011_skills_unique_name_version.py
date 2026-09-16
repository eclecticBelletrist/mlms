"""unique (name, version) on skills

Revision ID: 011
Revises: 010
Create Date: 2026-09-14

storage/skills.py computes the next version as
SELECT COALESCE(MAX(version), 0) + 1, then INSERTs — a read-then-write with
no constraint backing it, same TOCTOU race class as F002 on facts. Adding
the constraint here; fresh-deploy project, no legacy data expected to violate it.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "011"
down_revision: str | None = "010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE skills ADD CONSTRAINT skills_name_version_key UNIQUE (name, version)")


def downgrade() -> None:
    op.execute("ALTER TABLE skills DROP CONSTRAINT skills_name_version_key")
