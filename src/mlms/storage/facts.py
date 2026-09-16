"""Fact storage: upsert with cosine-based conflict detection, soft deletes."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import psycopg
import psycopg.rows

from mlms.config import COSINE_CONFLICT_THRESHOLD
from mlms.storage._vec import vec_literal


@dataclass(frozen=True)
class FactResult:
    id: UUID
    version: int
    is_conflict: bool
    replaced_id: UUID | None


# pgvector <=> is cosine DISTANCE (not similarity), so conflict ⟺ dist <= (1 - threshold)
_SELECT_NEAREST = """
    SELECT id, version, (embedding <=> %s::vector) AS dist
    FROM facts
    WHERE project_id = %s AND is_current = TRUE
    ORDER BY embedding <=> %s::vector
    LIMIT 1
"""

_UPDATE_RETIRE = "UPDATE facts SET is_current = FALSE, updated_at = %s WHERE id = %s"

_INSERT_FACT = """
    INSERT INTO facts (id, content, embedding, tags, category, project_id, version, is_current, updated_at, replaces_id)
    VALUES (%s, %s, %s::vector, %s, %s, %s, %s, TRUE, %s, %s)
"""

# Bounds the retry loop below — SERIALIZABLE conflicts are expected to be rare (only two
# concurrent upserts landing on near-duplicate embeddings for the same project), not a
# steady-state condition needing backoff.
_MAX_SERIALIZATION_RETRIES = 3


async def upsert_fact(
    *,
    content: str,
    project_id: UUID,
    embedding_vec: list[float],
    tags: list[str] | None = None,
    category: str | None = None,
    conn: psycopg.AsyncConnection[Any],
    threshold: float = COSINE_CONFLICT_THRESHOLD,
) -> FactResult:
    """Upsert fact; on cosine conflict soft-deletes the old row and bumps version.

    Runs at SERIALIZABLE isolation with a bounded retry: READ COMMITTED (the default)
    lets two concurrent upserts for near-duplicate embeddings both miss each other's
    uncommitted insert and both land is_current=TRUE, violating the single-current-row
    invariant get_facts()/search_memory() rely on.
    """
    vec_lit = vec_literal(embedding_vec)

    for attempt in range(_MAX_SERIALIZATION_RETRIES):
        new_id = uuid.uuid4()
        now = datetime.now(UTC)
        replaced_id: UUID | None = None
        version = 1

        try:
            async with conn.transaction():
                cur = conn.cursor(row_factory=psycopg.rows.dict_row)
                await cur.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")

                await cur.execute(_SELECT_NEAREST, (vec_lit, project_id, vec_lit))
                row = await cur.fetchone()

                if row is not None and row["dist"] is not None and row["dist"] <= 1.0 - threshold:
                    replaced_id = row["id"]
                    version = row["version"] + 1
                    await cur.execute(_UPDATE_RETIRE, (now, replaced_id))

                await cur.execute(
                    _INSERT_FACT,
                    (new_id, content, vec_lit, tags, category, project_id, version, now, replaced_id),
                )
            break
        except psycopg.errors.SerializationFailure:
            if attempt == _MAX_SERIALIZATION_RETRIES - 1:
                raise

    return FactResult(
        id=new_id,
        version=version,
        is_conflict=replaced_id is not None,
        replaced_id=replaced_id,
    )
