from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import psycopg
from pydantic import BaseModel

from mlms.embedding import embed
from mlms.storage.session_log import get_session_log as _storage_get_session_log


class SessionLogEntry(BaseModel):
    time: datetime
    chat_id: str
    entry_type: str
    label: str
    content: str | None = None
    tags: list[str] | None = None
    meta: dict[str, Any] | None = None


async def get_session_log(
    *,
    chat_id: str | None = None,
    limit: int = 50,
    entry_type: str | None = None,
    semantic_query: str | None = None,
    project_id: str | None = None,
    conn: psycopg.AsyncConnection[Any],
    redis: Any,
) -> list[SessionLogEntry]:
    if chat_id is not None and semantic_query is not None:
        raise ValueError("chat_id and semantic_query are mutually exclusive")
    if chat_id is None and semantic_query is None:
        return []

    vec: list[float] | None = None
    if semantic_query is not None:
        vec = await embed(semantic_query, redis)

    entries = await _storage_get_session_log(
        chat_id=chat_id,
        semantic_query=vec,
        project_id=UUID(project_id) if project_id is not None else None,
        limit=limit,
        conn=conn,
    )

    results = [
        SessionLogEntry(
            time=e.time,
            chat_id=e.chat_id,
            entry_type=e.entry_type,
            label=e.label,
            content=e.content,
            tags=e.tags,
            meta=e.meta,
        )
        for e in entries
    ]

    if entry_type is not None:
        results = [r for r in results if r.entry_type == entry_type]

    return results
