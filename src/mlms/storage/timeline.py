"""Timeline event storage for episodic memory at project scale."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import psycopg
import psycopg.rows
from psycopg.types.json import Jsonb


@dataclass(frozen=True)
class EventResult:
    id: UUID
    time: datetime


@dataclass(frozen=True)
class TimelineEvent:
    id: UUID
    time: datetime
    project_id: UUID
    event_type: str | None
    title: str | None
    content: str | None
    metadata: dict[str, Any] | None


@dataclass(frozen=True)
class EventPage:
    events: list[TimelineEvent]
    next_cursor: tuple[datetime, UUID] | None


_INSERT_EVENT = """
    INSERT INTO timeline_events (id, time, project_id, event_type, title, content, metadata)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
"""


async def insert_event(
    *,
    project_id: UUID,
    event_type: str,
    title: str,
    content: str | None = None,
    metadata: dict[str, Any] | None = None,
    time: datetime | None = None,
    conn: psycopg.AsyncConnection[Any],
) -> EventResult:
    event_id = uuid.uuid4()
    ts = time if time is not None else datetime.now(UTC)
    metadata_val: Jsonb | None = Jsonb(metadata) if metadata is not None else None

    await conn.execute(
        _INSERT_EVENT,
        (event_id, ts, project_id, event_type, title, content, metadata_val),
    )
    return EventResult(id=event_id, time=ts)


async def get_events(
    *,
    project_id: UUID,
    start: datetime,
    end: datetime,
    event_type: str | None = None,
    limit: int = 50,
    cursor: tuple[datetime, UUID] | None = None,
    conn: psycopg.AsyncConnection[Any],
) -> EventPage:
    where = "WHERE project_id = %s AND time >= %s AND time <= %s"
    params: list[Any] = [project_id, start, end]

    if event_type is not None:
        where += " AND event_type = %s"
        params.append(event_type)

    if cursor is not None:
        cursor_time, cursor_id = cursor
        # keyset: advance past (cursor_time, cursor_id) in (time ASC, id ASC) order
        where += " AND (time > %s OR (time = %s AND id > %s))"
        params.extend([cursor_time, cursor_time, cursor_id])

    sql = (
        "SELECT id, time, project_id, event_type, title, content, metadata"
        " FROM timeline_events"
        f" {where}"
        " ORDER BY time ASC, id ASC"
        " LIMIT %s"
    )
    params.append(limit + 1)

    cur = conn.cursor(row_factory=psycopg.rows.dict_row)
    await cur.execute(sql, params)
    rows: list[dict[str, Any]] = await cur.fetchall()

    has_next = len(rows) > limit
    page_rows = rows[:limit]

    events = [
        TimelineEvent(
            id=r["id"],
            time=r["time"],
            project_id=r["project_id"],
            event_type=r["event_type"],
            title=r["title"],
            content=r["content"],
            metadata=r["metadata"],
        )
        for r in page_rows
    ]

    next_cursor: tuple[datetime, UUID] | None = None
    if has_next and page_rows:
        last = page_rows[-1]
        next_cursor = (last["time"], last["id"])

    return EventPage(events=events, next_cursor=next_cursor)
