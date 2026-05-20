"""Session log: in-session by chat_id or cross-session by cosine search."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import psycopg
import psycopg.rows
from psycopg.types.json import Jsonb


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(str(x) for x in vec) + "]"


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

_IN_SESSION = _SELECT_COLS + " WHERE chat_id = %s ORDER BY time DESC LIMIT %s"

_CROSS_SESSION = (
    _SELECT_COLS + " WHERE embedding IS NOT NULL" + " ORDER BY embedding <=> %s::vector LIMIT %s"
)

_CROSS_SESSION_FILTERED = (
    _SELECT_COLS
    + " WHERE embedding IS NOT NULL AND project_id = %s::uuid"
    + " ORDER BY embedding <=> %s::vector LIMIT %s"
)


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
    vec_lit = _vec_literal(embedding_vec) if embedding_vec is not None else None
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
    limit: int = 20,
    conn: psycopg.AsyncConnection[Any],
) -> list[SessionEntry]:
    if chat_id is not None and semantic_query is not None:
        raise ValueError("chat_id and semantic_query are mutually exclusive — pick one mode")
    if chat_id is None and semantic_query is None:
        raise ValueError("provide chat_id for in-session mode or semantic_query for cross-session")

    cur = conn.cursor(row_factory=psycopg.rows.dict_row)

    if chat_id is not None:
        await cur.execute(_IN_SESSION, (chat_id, limit))
    else:
        assert semantic_query is not None
        if project_id is not None:
            await cur.execute(_CROSS_SESSION_FILTERED, (project_id, _vec_literal(semantic_query), limit))
        else:
            await cur.execute(_CROSS_SESSION, (_vec_literal(semantic_query), limit))

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
