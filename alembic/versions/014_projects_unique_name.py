"""unique constraint on projects.name

Revision ID: 014
Revises: 013
Create Date: 2026-09-18

create_project does get-or-create via INSERT ... ON CONFLICT (name) DO UPDATE
RETURNING id — needs a unique constraint on name to have a conflict target.
Fresh-deploy project, no legacy data expected to violate it.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "014"
down_revision: str | None = "013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE projects ADD CONSTRAINT projects_name_key UNIQUE (name)")


def downgrade() -> None:
    op.execute("ALTER TABLE projects DROP CONSTRAINT projects_name_key")
