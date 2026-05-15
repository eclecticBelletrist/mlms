from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mlms.tools.get_timeline import TimelineEvent, get_timeline

_NOW = datetime(2024, 3, 10, 9, 0, 0, tzinfo=UTC)
_PID = uuid.uuid4()


def _make_row(**kw: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "id": uuid.uuid4(),
        "time": _NOW,
        "project_id": _PID,
        "event_type": "action",
        "title": "some event",
        "content": "details",
        "metadata": None,
    }
    return {**defaults, **kw}


def _make_conn(rows: list[dict[str, Any]]) -> tuple[Any, MagicMock]:
    cur: MagicMock = MagicMock()
    cur.execute = AsyncMock()
    cur.fetchall = AsyncMock(return_value=rows)
    conn: Any = MagicMock()
    conn.cursor.return_value = cur
    return conn, cur


class TestGetTimeline:
    async def test_returns_list_of_timeline_event(self) -> None:
        conn, _ = _make_conn([_make_row()])
        result = await get_timeline(conn=conn)
        assert len(result) == 1
        assert isinstance(result[0], TimelineEvent)

    async def test_empty_results(self) -> None:
        conn, _ = _make_conn([])
        assert await get_timeline(conn=conn) == []

    async def test_no_filters_generates_no_where(self) -> None:
        conn, cur = _make_conn([])
        await get_timeline(conn=conn)
        sql: str = cur.execute.call_args[0][0]
        assert "WHERE" not in sql

    async def test_project_id_filter_in_sql(self) -> None:
        conn, cur = _make_conn([])
        await get_timeline(project_id=str(_PID), conn=conn)
        sql: str = cur.execute.call_args[0][0]
        assert "project_id" in sql

    async def test_event_type_filter_in_sql(self) -> None:
        conn, cur = _make_conn([])
        await get_timeline(event_type="decision", conn=conn)
        sql: str = cur.execute.call_args[0][0]
        assert "event_type" in sql

    async def test_time_range_adds_two_clauses(self) -> None:
        conn, cur = _make_conn([])
        tr = {"start": "2024-01-01T00:00:00+00:00", "end": "2024-12-31T23:59:59+00:00"}
        await get_timeline(time_range=tr, conn=conn)
        params: list[Any] = list(cur.execute.call_args[0][1])
        # should contain two datetime objects for start + end
        datetimes = [p for p in params if isinstance(p, datetime)]
        assert len(datetimes) == 2

    async def test_limit_in_params(self) -> None:
        conn, cur = _make_conn([])
        await get_timeline(limit=5, conn=conn)
        params: list[Any] = list(cur.execute.call_args[0][1])
        assert 5 in params

    async def test_fields_mapped_correctly(self) -> None:
        row = _make_row(event_type="phase", title="Phase 1", content="started")
        conn, _ = _make_conn([row])
        result = await get_timeline(conn=conn)
        e = result[0]
        assert e.event_type == "phase"
        assert e.title == "Phase 1"
        assert e.content == "started"
