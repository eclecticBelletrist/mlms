"""create session_log hypertable

Revision ID: 004
Revises: 003
Create Date: 2026-05-08
"""
from collections.abc import Sequence

from alembic import op

revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_VALID_TYPES = (
    "topic", "decision", "problem", "solution",
    "insight", "action", "source", "context", "result"
)


def upgrade() -> None:
    valid_list = ", ".join(f"'{t}'" for t in _VALID_TYPES)
    op.execute(f"""
        CREATE TABLE session_log (
            time        TIMESTAMPTZ NOT NULL,
            chat_id     TEXT        NOT NULL,
            type        TEXT        NOT NULL
                CONSTRAINT session_log_type_check CHECK (type IN ({valid_list})),
            label       TEXT        NOT NULL,
            content     TEXT,
            embedding   VECTOR(1536),
            tags        TEXT[],
            meta        JSONB
        )
    """)

    op.execute(
        "SELECT create_hypertable('session_log', 'time',"
        " if_not_exists => TRUE)"
    )

    op.execute("CREATE INDEX ON session_log (chat_id, time DESC)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS session_log")
