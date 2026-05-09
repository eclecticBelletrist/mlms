"""change embedding dim to 1536 (Matryoshka truncation)

Revision ID: 006
Revises: 005
Create Date: 2026-05-09
"""
from collections.abc import Sequence

from alembic import op

revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Drop ANN indexes if they somehow exist, then retype both embedding columns.
    # session_log is a TimescaleDB hypertable; ALTER propagates to all chunks.
    op.execute("DROP INDEX IF EXISTS facts_embedding_idx")
    op.execute("ALTER TABLE facts ALTER COLUMN embedding TYPE vector(1536)")
    op.execute(
        "CREATE INDEX ON facts USING hnsw (embedding vector_cosine_ops)"
        " WITH (m = 16, ef_construction = 64)"
    )

    op.execute("DROP INDEX IF EXISTS session_log_embedding_idx")
    op.execute("ALTER TABLE session_log ALTER COLUMN embedding TYPE vector(1536)")
    op.execute(
        "CREATE INDEX ON session_log USING hnsw (embedding vector_cosine_ops)"
        " WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    # Safe only on empty tables: vector(4096) data cannot be downcasted to vector(1536).
    op.execute("DROP INDEX IF EXISTS facts_embedding_idx")
    op.execute("ALTER TABLE facts ALTER COLUMN embedding TYPE vector(1536)")

    op.execute("DROP INDEX IF EXISTS session_log_embedding_idx")
    op.execute("ALTER TABLE session_log ALTER COLUMN embedding TYPE vector(1536)")
