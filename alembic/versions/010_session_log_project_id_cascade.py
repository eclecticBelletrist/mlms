"""cascade-delete session_log rows when their project is deleted

Revision ID: 010
Revises: 009
Create Date: 2026-09-14

007 added session_log.project_id via a bare `REFERENCES projects(id)`
(no CONSTRAINT name given inline), so Postgres assigned the default
`<table>_<column>_fkey` name: session_log_project_id_fkey. facts.project_id
(002) and timeline_events.project_id (003) both use ON DELETE CASCADE;
session_log.project_id was the odd one out at NO ACTION.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "010"
down_revision: str | None = "009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE session_log DROP CONSTRAINT session_log_project_id_fkey")
    op.execute(
        "ALTER TABLE session_log ADD CONSTRAINT session_log_project_id_fkey "
        "FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE session_log DROP CONSTRAINT session_log_project_id_fkey")
    op.execute(
        "ALTER TABLE session_log ADD CONSTRAINT session_log_project_id_fkey "
        "FOREIGN KEY (project_id) REFERENCES projects(id)"
    )
