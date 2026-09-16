"""Session log: in-session by chat_id or cross-session by cosine search."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import psycopg
import psycopg.rows
from psycopg.types.json import Jsonb

from mlms.storage._vec import vec_literal


@dataclass(frozen=True)
class SessionInsertResult:
    time: datetime
    chat_id: str


@dataclass(frozen=True)
class SessionEntry:
    time: datetime
    chat_id: str
    entry_type: str
    label: str
    content: str | None
    tags: list[str] | None
    meta: dict[str, Any] | None


_INSERT = """
    INSERT INTO session_log (time, chat_id, project_id, type, label, content, embedding, tags, meta)
    VALUES (%s, %s, %s, %s, %s, %s, %s::vector, %s, %s)
"""

_SELECT_COLS = "SELECT time, chat_id, type, label, content, tags, meta FROM session_log"

# pgvector HNSW is approximate: it pulls its ef_search candidate window (default 40)
# before any WHERE filter runs, so a selective type/project filter can leave fewer than
# LIMIT matches even when plenty exist in the table. hnsw.iterative_scan makes the index
# keep expanding the search until LIMIT is filled (or hnsw.max_scan_tuples is hit).
_ITERATIVE_SCAN = "SET LOCAL hnsw.iterative_scan = 'strict_order'"


async def insert_session_entry(
    *,
    chat_id: str,
    entry_type: str,
    label: str,
    content: str | None = None,
    embedding_vec: list[float] | None = None,
    tags: list[str] | None = None,
    meta: dict[str, Any] | None = None,
    time: datetime | None = None,
    project_id: UUID | None = None,
    conn: psycopg.AsyncConnection[Any],
) -> SessionInsertResult:
    ts = time if time is not None else datetime.now(UTC)
    vec_lit = vec_literal(embedding_vec) if embedding_vec is not None else None
    meta_val: Jsonb | None = Jsonb(meta) if meta is not None else None

    await conn.execute(
        _INSERT,
        (ts, chat_id, project_id, entry_type, label, content, vec_lit, tags, meta_val),
    )
    return SessionInsertResult(time=ts, chat_id=chat_id)


async def get_session_log(
    *,
    chat_id: str | None = None,
    semantic_query: list[float] | None = None,
    project_id: UUID | None = None,
    entry_type: str | None = None,
    limit: int = 20,
    conn: psycopg.AsyncConnection[Any],
) -> list[SessionEntry]:
    if chat_id is not None and semantic_query is not None:
        raise ValueError("chat_id and semantic_query are mutually exclusive — pick one mode")
    if chat_id is None and semantic_query is None:
        raise ValueError("provide chat_id for in-session mode or semantic_query for cross-session")

    cur = conn.cursor(row_factory=psycopg.rows.dict_row)

    if chat_id is not None:
        where = ["chat_id = %s"]
        params: list[Any] = [chat_id]
        if entry_type is not None:
            where.append("type = %s")
            params.append(entry_type)
        params.append(limit)
        query = f"{_SELECT_COLS} WHERE {' AND '.join(where)} ORDER BY time DESC LIMIT %s"
        await cur.execute(query, params)
    else:
        assert semantic_query is not None
        where = ["embedding IS NOT NULL"]
        params = []
        if project_id is not None:
            where.append("project_id = %s::uuid")
            params.append(project_id)
        if entry_type is not None:
            where.append("type = %s")
            params.append(entry_type)
        params.append(vec_literal(semantic_query))
        params.append(limit)
        query = (
            f"{_SELECT_COLS} WHERE {' AND '.join(where)} ORDER BY embedding <=> %s::vector LIMIT %s"
        )
        # SET LOCAL only holds for the current transaction, and this connection runs in
        # autocommit — each bare execute() is its own transaction — so the GUC and the
        # SELECT must share one explicit transaction, or the setting is gone before the
        # query runs.
        async with conn.transaction():
            await cur.execute(_ITERATIVE_SCAN)
            await cur.execute(query, params)

    rows: list[dict[str, Any]] = await cur.fetchall()
    return [
        SessionEntry(
            time=r["time"],
            chat_id=r["chat_id"],
            entry_type=r["type"],
            label=r["label"],
            content=r["content"],
            tags=r["tags"],
            meta=r["meta"],
        )
        for r in rows
    ]
