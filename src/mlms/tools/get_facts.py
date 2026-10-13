from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from uuid import UUID

import psycopg
import psycopg.rows
from pydantic import BaseModel

from mlms.embedding import embed
from mlms.storage._vec import vec_literal


class Fact(BaseModel):
    id: UUID
    content: str
    tags: list[str] | None = None
    category: str | None = None
    project_id: UUID | None = None
    version: int
    is_current: bool
    created_at: datetime
    updated_at: datetime | None = None


_EXACT_ID_RE = re.compile(r"\.\w{1,6}\b|\bv\d+[\._]\d|\b\d+\.\d+\.\d+")

_COLS = "id, content, tags, category, project_id, version, is_current, created_at, updated_at"


def _where(project_id: str | None, tags: list[str] | None) -> tuple[str, list[Any]]:
    clauses = ["is_current = TRUE"]
    params: list[Any] = []
    if project_id is not None:
        clauses.append("project_id = %s")
        params.append(UUID(project_id))
    if tags is not None:
        clauses.append("tags @> %s")
        params.append(tags)
    return " AND ".join(clauses), params


async def get_facts(
    *,
    about: str | None = None,
    project_id: str | None = None,
    tags: list[str] | None = None,
    limit: int = 10,
    conn: psycopg.AsyncConnection[Any],
    redis: Any,
) -> list[Fact]:
    cur = conn.cursor(row_factory=psycopg.rows.dict_row)
    cond, cond_params = _where(project_id, tags)

    if about is None:
        sql = f"SELECT {_COLS} FROM facts WHERE {cond} ORDER BY created_at DESC LIMIT %s"
        await cur.execute(sql, (*cond_params, limit))
        rows: list[dict[str, Any]] = await cur.fetchall()
        return [Fact.model_validate(r) for r in rows]

    vec = await embed(about, redis)
    vec_lit = vec_literal(vec)

    await cur.execute(
        f"SELECT {_COLS} FROM facts WHERE {cond} ORDER BY embedding <=> %s::vector LIMIT %s",
        (*cond_params, vec_lit, limit),
    )
    vec_rows: list[dict[str, Any]] = await cur.fetchall()

    if not _EXACT_ID_RE.search(about):
        return [Fact.model_validate(r) for r in vec_rows]

    await cur.execute(
        f"SELECT {_COLS} FROM facts WHERE {cond}"
        " AND to_tsvector('english', content) @@ plainto_tsquery('english', %s)"
        " LIMIT %s",
        (*cond_params, about, limit),
    )
    exact_rows: list[dict[str, Any]] = await cur.fetchall()
    exact_ids = {r["id"] for r in exact_rows}
    vec_ids = {r["id"] for r in vec_rows}
    extra = [r for r in exact_rows if r["id"] not in vec_ids]

    # An exact hit must make the page: it displaces the lowest-ranked vector rows
    # that are not exact hits themselves.
    overflow = len(vec_rows) + len(extra) - limit
    if overflow > 0:
        dropped = [r["id"] for r in reversed(vec_rows) if r["id"] not in exact_ids][:overflow]
        vec_rows = [r for r in vec_rows if r["id"] not in dropped]

    return [Fact.model_validate(r) for r in [*vec_rows, *extra]]
