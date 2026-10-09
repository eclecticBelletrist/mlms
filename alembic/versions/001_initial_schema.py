"""initial schema

Revision ID: 001
Revises:
Create Date: 2026-10-09

Replaces the incremental migrations 001-014 from before the first release; databases
created by them are not upgradable and have to be recreated (see KEY_DECISIONS.md Decision 28).
"""

from collections.abc import Sequence

from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SESSION_LOG_TYPES = (
    "topic",
    "decision",
    "problem",
    "solution",
    "insight",
    "action",
    "source",
    "context",
    "result",
)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")

    op.execute("""
        CREATE TABLE projects (
            id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            name        TEXT        NOT NULL,
            description TEXT,
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            CONSTRAINT projects_name_key UNIQUE (name)
        )
    """)

    op.execute("""
        CREATE TABLE facts (
            id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            content     TEXT        NOT NULL,
            embedding   VECTOR(1536),
            tags        TEXT[],
            category    TEXT,
            project_id  UUID        REFERENCES projects(id) ON DELETE CASCADE,
            created_at  TIMESTAMPTZ DEFAULT NOW(),
            updated_at  TIMESTAMPTZ,
            version     INT         DEFAULT 1,
            is_current  BOOL        DEFAULT TRUE,
            replaces_id UUID        REFERENCES facts(id),
            CONSTRAINT facts_replaces_id_not_self
                CHECK (replaces_id IS NULL OR replaces_id <> id)
        )
    """)
    op.execute("CREATE INDEX ON facts (project_id, is_current)")
    op.execute("CREATE INDEX ON facts (replaces_id)")
    op.execute(
        "CREATE INDEX facts_embedding_idx ON facts USING hnsw (embedding vector_cosine_ops)"
        " WITH (m = 16, ef_construction = 64)"
    )

    op.execute("""
        CREATE TABLE timeline_events (
            id          UUID        NOT NULL DEFAULT gen_random_uuid(),
            time        TIMESTAMPTZ NOT NULL,
            project_id  UUID        REFERENCES projects(id) ON DELETE CASCADE,
            event_type  TEXT        NOT NULL CHECK (event_type IN ('phase', 'action', 'decision')),
            title       TEXT,
            content     TEXT,
            metadata    JSONB
        )
    """)
    op.execute("SELECT create_hypertable('timeline_events', 'time')")
    # TimescaleDB requires the partition column in any unique constraint
    op.execute("ALTER TABLE timeline_events ADD PRIMARY KEY (id, time)")

    valid_types = ", ".join(f"'{t}'" for t in _SESSION_LOG_TYPES)
    op.execute(f"""
        CREATE TABLE session_log (
            id          UUID        NOT NULL DEFAULT gen_random_uuid(),
            time        TIMESTAMPTZ NOT NULL,
            chat_id     TEXT        NOT NULL,
            project_id  UUID        REFERENCES projects(id) ON DELETE CASCADE,
            type        TEXT        NOT NULL
                CONSTRAINT session_log_type_check CHECK (type IN ({valid_types})),
            label       TEXT        NOT NULL,
            content     TEXT,
            embedding   VECTOR(1536),
            tags        TEXT[],
            meta        JSONB
        )
    """)
    op.execute("SELECT create_hypertable('session_log', 'time')")
    op.execute("ALTER TABLE session_log ADD PRIMARY KEY (id, time)")
    op.execute("CREATE INDEX ON session_log (chat_id, time DESC)")
    op.execute("CREATE INDEX ON session_log (project_id)")
    op.execute(
        "CREATE INDEX session_log_embedding_idx ON session_log USING hnsw"
        " (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    op.execute("""
        CREATE TABLE skills (
            id                 UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
            name               TEXT        NOT NULL,
            domain             TEXT,
            content            TEXT        NOT NULL,
            trigger_conditions JSONB,
            version            INT         DEFAULT 1,
            created_at         TIMESTAMPTZ DEFAULT NOW(),
            CONSTRAINT skills_name_version_key UNIQUE (name, version)
        )
    """)
    op.execute("CREATE INDEX ON skills (name)")


def downgrade() -> None:
    for table in ("skills", "session_log", "timeline_events", "facts", "projects"):
        op.execute(f"DROP TABLE IF EXISTS {table}")
    op.execute("DROP EXTENSION IF EXISTS vector")
    op.execute("DROP EXTENSION IF EXISTS timescaledb")
