"""Unit tests for storage/session_log.py — no real DB required."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from psycopg.types.json import Jsonb

from mlms.config import EMBEDDING_DIM
from mlms.storage.session_log import (
    SessionEntry,
    SessionInsertResult,
    get_session_log,
    insert_session_entry,
)

_CHAT = "chat_abc"
_EMBEDDING = [0.1] * EMBEDDING_DIM
_NOW = datetime(2024, 3, 10, 9, 0, 0, tzinfo=UTC)


def _make_conn(fetchall_result: list[dict[str, Any]] | None = None) -> tuple[Any, MagicMock]:
    cur = MagicMock()
    cur.execute = AsyncMock(return_value=None)
    cur.fetchall = AsyncMock(return_value=fetchall_result or [])

    conn: Any = MagicMock()
    conn.execute = AsyncMock(return_value=None)
    conn.cursor.return_value = cur

    return conn, cur


def _make_row(
    *,
    ts: datetime | None = None,
    chat_id: str = _CHAT,
    entry_type: str = "insight",
    label: str = "Test label",
) -> dict[str, Any]:
    return {
        "time": ts or _NOW,
        "chat_id": chat_id,
        "type": entry_type,
        "label": label,
        "content": None,
        "tags": None,
        "meta": None,
    }


# ── insert_session_entry ──────────────────────────────────────────────────────


class TestInsertSessionEntry:
    async def test_returns_result_with_chat_id_and_time(self) -> None:
        conn, _ = _make_conn()
        result = await insert_session_entry(
            chat_id=_CHAT, entry_type="insight", label="Test", conn=conn
        )
        assert isinstance(result, SessionInsertResult)
        assert result.chat_id == _CHAT
        assert isinstance(result.time, datetime)

    async def test_uses_current_time_when_not_provided(self) -> None:
        conn, _ = _make_conn()
        before = datetime.now(UTC)
        result = await insert_session_entry(
            chat_id=_CHAT, entry_type="topic", label="Test", conn=conn
        )
        after = datetime.now(UTC)
        assert before <= result.time <= after

    async def test_uses_provided_time(self) -> None:
        conn, _ = _make_conn()
        result = await insert_session_entry(
            chat_id=_CHAT, entry_type="action", label="Test", time=_NOW, conn=conn
        )
        assert result.time == _NOW

    async def test_execute_called_once(self) -> None:
        conn, _ = _make_conn()
        await insert_session_entry(chat_id=_CHAT, entry_type="decision", label="T", conn=conn)
        conn.execute.assert_called_once()

    async def test_embedding_none_sends_none(self) -> None:
        conn, _ = _make_conn()
        await insert_session_entry(chat_id=_CHAT, entry_type="insight", label="T", conn=conn)
        # embedding is 7th param (index 6): (ts, chat_id, project_id, type, label, content, embedding, tags, meta)
        call_params = conn.execute.call_args[0][1]
        assert call_params[6] is None

    async def test_embedding_vec_converted_to_string_literal(self) -> None:
        conn, _ = _make_conn()
        await insert_session_entry(
            chat_id=_CHAT,
            entry_type="insight",
            label="T",
            embedding_vec=_EMBEDDING,
            conn=conn,
        )
        call_params = conn.execute.call_args[0][1]
        assert isinstance(call_params[6], str)
        assert call_params[6].startswith("[")

    async def test_meta_none_passed_as_none(self) -> None:
        conn, _ = _make_conn()
        await insert_session_entry(chat_id=_CHAT, entry_type="insight", label="T", conn=conn)
        call_params = conn.execute.call_args[0][1]
        assert call_params[-1] is None

    async def test_meta_dict_wrapped_as_jsonb(self) -> None:
        conn, _ = _make_conn()
        await insert_session_entry(
            chat_id=_CHAT,
            entry_type="source",
            label="T",
            meta={"url": "https://example.com"},
            conn=conn,
        )
        call_params = conn.execute.call_args[0][1]
        assert isinstance(call_params[-1], Jsonb)


# ── get_session_log — in-session mode ─────────────────────────────────────────


class TestGetSessionLogInSession:
    async def test_in_session_filters_by_chat_id(self) -> None:
        conn, cur = _make_conn([])
        await get_session_log(chat_id=_CHAT, conn=conn)
        params: list[Any] = cur.execute.call_args[0][1]
        assert _CHAT in params

    async def test_in_session_uses_time_desc_order(self) -> None:
        conn, cur = _make_conn([])
        await get_session_log(chat_id=_CHAT, conn=conn)
        sql: str = cur.execute.call_args[0][0]
        assert "time DESC" in sql

    async def test_in_session_no_cosine_operator(self) -> None:
        conn, cur = _make_conn([])
        await get_session_log(chat_id=_CHAT, conn=conn)
        sql: str = cur.execute.call_args[0][0]
        assert "<=>" not in sql

    async def test_in_session_returns_session_entries(self) -> None:
        conn, _ = _make_conn([_make_row()])
        result = await get_session_log(chat_id=_CHAT, conn=conn)
        assert len(result) == 1
        assert isinstance(result[0], SessionEntry)

    async def test_in_session_fields_mapped_correctly(self) -> None:
        row = _make_row(ts=_NOW, entry_type="decision", label="Ship it")
        conn, _ = _make_conn([row])
        result = await get_session_log(chat_id=_CHAT, conn=conn)
        e = result[0]
        assert e.time == _NOW
        assert e.chat_id == _CHAT
        assert e.entry_type == "decision"
        assert e.label == "Ship it"
        assert e.content is None

    async def test_in_session_empty_result(self) -> None:
        conn, _ = _make_conn([])
        result = await get_session_log(chat_id=_CHAT, conn=conn)
        assert result == []

    async def test_in_session_respects_limit(self) -> None:
        conn, cur = _make_conn([])
        await get_session_log(chat_id=_CHAT, limit=7, conn=conn)
        params: list[Any] = cur.execute.call_args[0][1]
        assert 7 in params

    async def test_in_session_multiple_entries_returned(self) -> None:
        rows = [_make_row(entry_type=t) for t in ("topic", "insight", "action")]
        conn, _ = _make_conn(rows)
        result = await get_session_log(chat_id=_CHAT, conn=conn)
        assert len(result) == 3


# ── get_session_log — cross-session mode ──────────────────────────────────────


class TestGetSessionLogCrossSession:
    async def test_cross_session_uses_cosine_operator(self) -> None:
        conn, cur = _make_conn([])
        await get_session_log(semantic_query=_EMBEDDING, conn=conn)
        sql: str = cur.execute.call_args[0][0]
        assert "<=>" in sql

    async def test_cross_session_no_chat_id_filter(self) -> None:
        conn, cur = _make_conn([])
        await get_session_log(semantic_query=_EMBEDDING, conn=conn)
        sql: str = cur.execute.call_args[0][0]
        assert "chat_id = " not in sql

    async def test_cross_session_filters_null_embeddings(self) -> None:
        conn, cur = _make_conn([])
        await get_session_log(semantic_query=_EMBEDDING, conn=conn)
        sql: str = cur.execute.call_args[0][0]
        assert "embedding IS NOT NULL" in sql

    async def test_cross_session_vec_literal_in_params(self) -> None:
        conn, cur = _make_conn([])
        await get_session_log(semantic_query=_EMBEDDING, conn=conn)
        params: list[Any] = cur.execute.call_args[0][1]
        assert isinstance(params[0], str)
        assert params[0].startswith("[")

    async def test_cross_session_returns_session_entries(self) -> None:
        conn, _ = _make_conn([_make_row()])
        result = await get_session_log(semantic_query=_EMBEDDING, conn=conn)
        assert len(result) == 1
        assert isinstance(result[0], SessionEntry)

    async def test_cross_session_respects_limit(self) -> None:
        conn, cur = _make_conn([])
        await get_session_log(semantic_query=_EMBEDDING, limit=10, conn=conn)
        params: list[Any] = cur.execute.call_args[0][1]
        assert 10 in params

    async def test_cross_session_empty_result(self) -> None:
        conn, _ = _make_conn([])
        result = await get_session_log(semantic_query=_EMBEDDING, conn=conn)
        assert result == []


# ── get_session_log — cross-session with project_id filter ────────────────────

import uuid as _uuid


class TestGetSessionLogProjectFilter:
    _PROJECT = _uuid.uuid4()

    async def test_project_id_filter_uses_project_scoped_query(self) -> None:
        conn, cur = _make_conn([])
        await get_session_log(semantic_query=_EMBEDDING, project_id=self._PROJECT, conn=conn)
        sql: str = cur.execute.call_args[0][0]
        assert "project_id = " in sql

    async def test_project_id_filter_passes_project_id_as_first_param(self) -> None:
        conn, cur = _make_conn([])
        await get_session_log(semantic_query=_EMBEDDING, project_id=self._PROJECT, conn=conn)
        params: list[Any] = cur.execute.call_args[0][1]
        assert params[0] == self._PROJECT

    async def test_no_project_id_omits_project_filter(self) -> None:
        conn, cur = _make_conn([])
        await get_session_log(semantic_query=_EMBEDDING, conn=conn)
        sql: str = cur.execute.call_args[0][0]
        assert "project_id" not in sql


# ── validation — mutually exclusive modes ─────────────────────────────────────


class TestGetSessionLogValidation:
    async def test_both_params_raises_value_error(self) -> None:
        conn, _ = _make_conn()
        with pytest.raises(ValueError, match="mutually exclusive"):
            await get_session_log(chat_id=_CHAT, semantic_query=_EMBEDDING, conn=conn)

    async def test_neither_param_raises_value_error(self) -> None:
        conn, _ = _make_conn()
        with pytest.raises(ValueError):
            await get_session_log(conn=conn)

    async def test_only_chat_id_does_not_raise(self) -> None:
        conn, _ = _make_conn([])
        result = await get_session_log(chat_id=_CHAT, conn=conn)
        assert isinstance(result, list)

    async def test_only_semantic_query_does_not_raise(self) -> None:
        conn, _ = _make_conn([])
        result = await get_session_log(semantic_query=_EMBEDDING, conn=conn)
        assert isinstance(result, list)
