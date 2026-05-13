"""Unit tests for storage/timeline.py — no real DB required."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from psycopg.types.json import Jsonb

from mlms.storage.timeline import EventPage, EventResult, TimelineEvent, get_events, insert_event

_PROJECT = uuid.uuid4()
_NOW = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)


def _make_conn(fetchall_result: list[dict[str, Any]] | None = None) -> tuple[Any, MagicMock]:
    """Return (mock_conn, mock_cursor) with psycopg3 async contract."""
    cur = MagicMock()
    cur.execute = AsyncMock(return_value=None)
    cur.fetchall = AsyncMock(return_value=fetchall_result or [])

    conn: Any = MagicMock()
    conn.execute = AsyncMock(return_value=None)
    conn.cursor.return_value = cur

    return conn, cur


def _make_row(
    *,
    event_id: uuid.UUID | None = None,
    ts: datetime | None = None,
    event_type: str = "action",
    title: str = "Test Event",
) -> dict[str, Any]:
    return {
        "id": event_id or uuid.uuid4(),
        "time": ts or _NOW,
        "project_id": _PROJECT,
        "event_type": event_type,
        "title": title,
        "content": None,
        "metadata": None,
    }


# ── insert_event ──────────────────────────────────────────────────────────────


class TestInsertEvent:
    async def test_returns_event_result(self) -> None:
        conn, _ = _make_conn()
        result = await insert_event(
            project_id=_PROJECT, event_type="action", title="Test", conn=conn
        )
        assert isinstance(result, EventResult)
        assert isinstance(result.id, uuid.UUID)
        assert isinstance(result.time, datetime)

    async def test_uses_current_time_when_not_provided(self) -> None:
        conn, _ = _make_conn()
        before = datetime.now(UTC)
        result = await insert_event(
            project_id=_PROJECT, event_type="phase", title="Phase Start", conn=conn
        )
        after = datetime.now(UTC)
        assert before <= result.time <= after

    async def test_uses_provided_time(self) -> None:
        conn, _ = _make_conn()
        custom_time = datetime(2024, 6, 1, tzinfo=UTC)
        result = await insert_event(
            project_id=_PROJECT,
            event_type="decision",
            title="Decision",
            time=custom_time,
            conn=conn,
        )
        assert result.time == custom_time

    async def test_execute_called_once(self) -> None:
        conn, _ = _make_conn()
        await insert_event(project_id=_PROJECT, event_type="action", title="Test", conn=conn)
        conn.execute.assert_called_once()

    async def test_metadata_none_passed_as_none(self) -> None:
        conn, _ = _make_conn()
        await insert_event(project_id=_PROJECT, event_type="action", title="Test", conn=conn)
        call_params = conn.execute.call_args[0][1]
        assert call_params[-1] is None

    async def test_metadata_dict_wrapped_as_jsonb(self) -> None:
        conn, _ = _make_conn()
        await insert_event(
            project_id=_PROJECT,
            event_type="action",
            title="Test",
            metadata={"key": "value"},
            conn=conn,
        )
        call_params = conn.execute.call_args[0][1]
        assert isinstance(call_params[-1], Jsonb)

    async def test_different_event_types_accepted(self) -> None:
        for etype in ("phase", "action", "decision"):
            conn, _ = _make_conn()
            result = await insert_event(project_id=_PROJECT, event_type=etype, title="T", conn=conn)
            assert isinstance(result.id, uuid.UUID)


# ── get_events ────────────────────────────────────────────────────────────────


class TestGetEvents:
    async def test_empty_result_returns_empty_page(self) -> None:
        conn, _ = _make_conn([])
        page = await get_events(
            project_id=_PROJECT,
            start=_NOW,
            end=_NOW + timedelta(hours=1),
            conn=conn,
        )
        assert page.events == []
        assert page.next_cursor is None

    async def test_fewer_than_limit_no_cursor(self) -> None:
        rows = [_make_row() for _ in range(3)]
        conn, _ = _make_conn(rows)
        page = await get_events(
            project_id=_PROJECT,
            start=_NOW,
            end=_NOW + timedelta(hours=1),
            limit=10,
            conn=conn,
        )
        assert len(page.events) == 3
        assert page.next_cursor is None

    async def test_limit_plus_one_rows_sets_cursor(self) -> None:
        limit = 3
        cursor_id = uuid.uuid4()
        cursor_time = _NOW + timedelta(minutes=2)
        rows = [
            _make_row(ts=_NOW),
            _make_row(ts=_NOW + timedelta(minutes=1)),
            _make_row(event_id=cursor_id, ts=cursor_time),  # last returned
            _make_row(ts=_NOW + timedelta(minutes=3)),  # extra → triggers has_next
        ]
        conn, _ = _make_conn(rows)
        page = await get_events(
            project_id=_PROJECT,
            start=_NOW,
            end=_NOW + timedelta(hours=1),
            limit=limit,
            conn=conn,
        )
        assert len(page.events) == limit
        assert page.next_cursor == (cursor_time, cursor_id)

    async def test_event_type_filter_in_sql_and_params(self) -> None:
        conn, cur = _make_conn([])
        await get_events(
            project_id=_PROJECT,
            start=_NOW,
            end=_NOW + timedelta(hours=1),
            event_type="decision",
            conn=conn,
        )
        sql: str = cur.execute.call_args[0][0]
        params: list[Any] = cur.execute.call_args[0][1]
        assert "event_type" in sql
        assert "decision" in params

    async def test_no_event_type_filter_absent_from_sql(self) -> None:
        conn, cur = _make_conn([])
        await get_events(
            project_id=_PROJECT,
            start=_NOW,
            end=_NOW + timedelta(hours=1),
            conn=conn,
        )
        sql: str = cur.execute.call_args[0][0]
        assert "AND event_type" not in sql

    async def test_cursor_adds_keyset_condition(self) -> None:
        cursor_time = _NOW + timedelta(minutes=5)
        cursor_id = uuid.uuid4()
        conn, cur = _make_conn([])
        await get_events(
            project_id=_PROJECT,
            start=_NOW,
            end=_NOW + timedelta(hours=1),
            cursor=(cursor_time, cursor_id),
            conn=conn,
        )
        sql: str = cur.execute.call_args[0][0]
        params: list[Any] = cur.execute.call_args[0][1]
        assert "time > %s" in sql
        assert cursor_time in params
        assert cursor_id in params
        # cursor_time appears twice (time > ? and time = ?)
        assert params.count(cursor_time) == 2

    async def test_no_cursor_condition_absent_from_sql(self) -> None:
        conn, cur = _make_conn([])
        await get_events(
            project_id=_PROJECT,
            start=_NOW,
            end=_NOW + timedelta(hours=1),
            conn=conn,
        )
        sql: str = cur.execute.call_args[0][0]
        assert "time > %s" not in sql

    async def test_returns_timeline_event_dataclasses(self) -> None:
        conn, _ = _make_conn([_make_row()])
        page = await get_events(
            project_id=_PROJECT,
            start=_NOW,
            end=_NOW + timedelta(hours=1),
            conn=conn,
        )
        assert isinstance(page, EventPage)
        assert isinstance(page.events[0], TimelineEvent)

    async def test_limit_plus_one_fetched_from_db(self) -> None:
        """DB receives limit+1 so next-page detection doesn't require extra query."""
        conn, cur = _make_conn([])
        await get_events(
            project_id=_PROJECT,
            start=_NOW,
            end=_NOW + timedelta(hours=1),
            limit=5,
            conn=conn,
        )
        params: list[Any] = cur.execute.call_args[0][1]
        assert params[-1] == 6  # limit + 1

    async def test_event_fields_mapped_correctly(self) -> None:
        eid = uuid.uuid4()
        row = _make_row(event_id=eid, ts=_NOW, event_type="phase", title="Phase 1")
        conn, _ = _make_conn([row])
        page = await get_events(
            project_id=_PROJECT,
            start=_NOW,
            end=_NOW + timedelta(hours=1),
            conn=conn,
        )
        e = page.events[0]
        assert e.id == eid
        assert e.time == _NOW
        assert e.project_id == _PROJECT
        assert e.event_type == "phase"
        assert e.title == "Phase 1"
        assert e.content is None
        assert e.metadata is None

    async def test_exactly_limit_rows_no_cursor(self) -> None:
        """Exactly limit rows (not limit+1): last page, no cursor."""
        limit = 3
        rows = [_make_row() for _ in range(limit)]
        conn, _ = _make_conn(rows)
        page = await get_events(
            project_id=_PROJECT,
            start=_NOW,
            end=_NOW + timedelta(hours=1),
            limit=limit,
            conn=conn,
        )
        assert len(page.events) == limit
        assert page.next_cursor is None
