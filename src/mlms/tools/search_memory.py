from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

import psycopg
import psycopg.rows
from pydantic import BaseModel

from mlms.embedding import embed
from mlms.storage._vec import vec_literal
from mlms.storage.skills import get_skills


class MemoryResult(BaseModel):
    layer: Literal["fact", "event", "skill"]
    id: str
    content: str
    score: float
    metadata: dict[str, Any] = {}


async def _search_facts(
    vec: list[float],
    project_id: str | None,
    conn: psycopg.AsyncConnection[Any],
    limit: int,
) -> list[MemoryResult]:
    vec_lit = vec_literal(vec)
    clauses = ["is_current = TRUE", "embedding IS NOT NULL"]
    params: list[Any] = [vec_lit]

    if project_id is not None:
        clauses.append("project_id = %s")
        params.append(UUID(project_id))

    where = " AND ".join(clauses)
    params.extend([vec_lit, limit])

    cur = conn.cursor(row_factory=psycopg.rows.dict_row)
    await cur.execute(
        "SELECT id, content, project_id, (embedding <=> %s::vector) AS dist"
        f" FROM facts WHERE {where}"
        " ORDER BY embedding <=> %s::vector LIMIT %s",
        params,
    )
    rows: list[dict[str, Any]] = await cur.fetchall()
    return [
        MemoryResult(
            layer="fact",
            id=str(r["id"]),
            content=r["content"],
            score=max(0.0, 1.0 - float(r["dist"])),
            metadata={"project_id": str(r["project_id"])} if r["project_id"] else {},
        )
        for r in rows
    ]


async def _search_events(
    query: str,
    project_id: str | None,
    time_range: dict[str, str] | None,
    conn: psycopg.AsyncConnection[Any],
    limit: int,
) -> list[MemoryResult]:
    clauses: list[str] = []
    params: list[Any] = []

    if project_id is not None:
        clauses.append("project_id = %s")
        params.append(UUID(project_id))

    if time_range is not None:
        if "start" in time_range:
            clauses.append("time >= %s")
            params.append(datetime.fromisoformat(time_range["start"]))
        if "end" in time_range:
            clauses.append("time <= %s")
            params.append(datetime.fromisoformat(time_range["end"]))

    clauses.append(
        "to_tsvector('english', coalesce(title,'') || ' ' || coalesce(content,''))"
        " @@ plainto_tsquery('english', %s)"
    )
    params.extend([query, limit])

    where = "WHERE " + " AND ".join(clauses)
    cur = conn.cursor(row_factory=psycopg.rows.dict_row)
    await cur.execute(
        f"SELECT id, content, project_id, time FROM timeline_events {where}"
        " ORDER BY time DESC LIMIT %s",
        params,
    )
    rows = await cur.fetchall()
    return [
        MemoryResult(
            layer="event",
            id=str(r["id"]),
            content=r["content"] or "",
            score=0.5,
            metadata={"time": r["time"].isoformat()} if r.get("time") else {},
        )
        for r in rows
    ]


async def _search_skills(
    query: str,
    conn: psycopg.AsyncConnection[Any],
    limit: int,
) -> list[MemoryResult]:
    skills = await get_skills(search=query, limit=limit, conn=conn)
    return [
        MemoryResult(
            layer="skill",
            id=str(s.id),
            content=s.content,
            score=0.5,
            metadata={"name": s.name},
        )
        for s in skills
    ]


async def search_memory(
    *,
    query: str,
    filters: dict[str, Any] | None = None,
    conn: psycopg.AsyncConnection[Any],
    redis: Any,
) -> list[MemoryResult]:
    resolved: dict[str, Any] = filters if filters is not None else {}
    memory_type: str | None = resolved.get("memory_type")
    project_id: str | None = resolved.get("project_id")
    time_range: dict[str, str] | None = resolved.get("time_range")

    fact_results: list[MemoryResult] = []
    event_results: list[MemoryResult] = []
    skill_results: list[MemoryResult] = []

    if memory_type is None:
        vec = await embed(query, redis)
        fact_results, event_results, skill_results = await asyncio.gather(
            _search_facts(vec, project_id, conn, 20),
            _search_events(query, project_id, time_range, conn, 20),
            _search_skills(query, conn, 10),
        )
    elif memory_type == "fact":
        vec = await embed(query, redis)
        fact_results = await _search_facts(vec, project_id, conn, 20)
    elif memory_type == "event":
        event_results = await _search_events(query, project_id, time_range, conn, 20)
    elif memory_type == "skill":
        skill_results = await _search_skills(query, conn, 10)

    # only facts carry a real cosine-derived score — events/skills are both a
    # hardcoded 0.5, so sorting them together against facts by score is meaningless.
    # facts come first as a real-ranked block, events/skills follow as a separate block.
    fact_results.sort(key=lambda r: r.score, reverse=True)
    return fact_results + event_results + skill_results
