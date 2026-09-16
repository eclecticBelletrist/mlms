"""create HNSW ANN indexes on facts/session_log embeddings

Revision ID: 006
Revises: 005
Create Date: 2026-05-09

002/004 already declare embedding as vector(1536) directly, so the ALTER COLUMN TYPE
and DROP INDEX IF EXISTS statements below are no-ops on a fresh deploy — kept only for
idempotency/compatibility with any hypothetical pre-1536 database state. The real, live
effect of this migration is creating the two HNSW indexes.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ALTER COLUMN TYPE + DROP INDEX IF EXISTS here are defensive no-ops on a fresh deploy
    # (002/004 already create embedding as vector(1536)); kept for compatibility with any
    # hypothetical pre-1536 database. session_log is a TimescaleDB hypertable; ALTER propagates
    # to all chunks. The indexes created below are this migration's real effect.
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
    raise NotImplementedError(
        "006 is irreversible: vector(4096) -> vector(1536) is a lossy Matryoshka "
        "truncation, original dimensions cannot be reconstructed. Restore from a "
        "pre-006 backup instead of downgrading."
    )
