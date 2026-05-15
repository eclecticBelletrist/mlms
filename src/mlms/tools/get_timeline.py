from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

import psycopg
import psycopg.rows
from pydantic import BaseModel


class TimelineEvent(BaseModel):
    id: UUID
    time: datetime
    project_id: UUID | None = None
    event_type: str | None = None
    title: str | None = None
    content: str | None = None
    metadata: dict[str, Any] | None = None


_COLS = "id, time, project_id, event_type, title, content, metadata"


async def get_timeline(
    *,
    project_id: str | None = None,
    time_range: dict[str, str] | None = None,
    event_type: Literal["phase", "action", "decision"] | None = None,
    limit: int = 20,
    conn: psycopg.AsyncConnection[Any],
) -> list[TimelineEvent]:
    clauses: list[str] = []
    params: list[Any] = []

    if project_id is not None:
        clauses.append("project_id = %s")
        params.append(UUID(project_id))

    if time_range is not None:
        clauses.append("time >= %s")
        params.append(datetime.fromisoformat(time_range["start"]))
        clauses.append("time <= %s")
        params.append(datetime.fromisoformat(time_range["end"]))

    if event_type is not None:
        clauses.append("event_type = %s")
        params.append(event_type)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(limit)

    cur = conn.cursor(row_factory=psycopg.rows.dict_row)
    await cur.execute(
        f"SELECT {_COLS} FROM timeline_events {where} ORDER BY time DESC LIMIT %s",
        params,
    )
    rows: list[dict[str, Any]] = await cur.fetchall()
    return [TimelineEvent.model_validate(r) for r in rows]
