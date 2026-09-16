"""rebuild HNSW ANN indexes with CONCURRENTLY

Revision ID: 013
Revises: 012
Create Date: 2026-09-14

006 built facts_embedding_idx / session_log_embedding_idx with plain
CREATE INDEX, which takes a lock that blocks writes for the build duration —
fine against an empty fresh-deploy table, not fine against a populated one.
CREATE/DROP INDEX CONCURRENTLY cannot run inside a transaction, hence
autocommit_block(). A CONCURRENTLY build that gets interrupted leaves an
INVALID index behind, so DROP INDEX CONCURRENTLY IF EXISTS runs first,
defensively, before each CREATE.

session_log is a TimescaleDB hypertable (003/004); TimescaleDB rejects
CREATE INDEX CONCURRENTLY on a hypertable outright (psycopg.errors.
FeatureNotSupported — confirmed against a live pg16 + timescaledb-ha
container, not just inferred), so this migration only rebuilds
facts_embedding_idx. session_log_embedding_idx keeps the blocking index
006 already built; there's no online-build path for it in the community
extension.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "013"
down_revision: str | None = "012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS facts_embedding_idx")
        op.execute(
            "CREATE INDEX CONCURRENTLY facts_embedding_idx ON facts "
            "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        )


def downgrade() -> None:
    # Restores the pre-013 state: same index, built non-concurrently (blocking) as 006 did.
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS facts_embedding_idx")
        op.execute(
            "CREATE INDEX facts_embedding_idx ON facts "
            "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
        )
