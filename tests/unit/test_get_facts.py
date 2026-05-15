from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlms.tools.get_facts import Fact, get_facts

_VEC = [0.1] * 1536
_NOW = datetime(2024, 3, 10, 9, 0, 0, tzinfo=UTC)
_PID = uuid.uuid4()
_P_EMBED = "mlms.tools.get_facts.embed"


def _make_row(**kw: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "id": uuid.uuid4(),
        "content": "some fact",
        "tags": None,
        "category": None,
        "project_id": _PID,
        "version": 1,
        "is_current": True,
        "created_at": _NOW,
        "updated_at": None,
    }
    return {**defaults, **kw}


def _make_conn(rows: list[dict[str, Any]]) -> tuple[Any, MagicMock]:
    cur: MagicMock = MagicMock()
    cur.execute = AsyncMock()
    cur.fetchall = AsyncMock(return_value=rows)
    conn: Any = MagicMock()
    conn.cursor.return_value = cur
    return conn, cur


class TestGetFactsNoAbout:
    async def test_returns_list_of_fact(self) -> None:
        conn, _ = _make_conn([_make_row()])
        redis = MagicMock()
        result = await get_facts(conn=conn, redis=redis)
        assert len(result) == 1
        assert isinstance(result[0], Fact)

    async def test_empty_results(self) -> None:
        conn, _ = _make_conn([])
        result = await get_facts(conn=conn, redis=MagicMock())
        assert result == []

    async def test_embed_not_called_without_about(self) -> None:
        conn, _ = _make_conn([])
        with patch(_P_EMBED, new=AsyncMock()) as mock_embed:
            await get_facts(conn=conn, redis=MagicMock())
            mock_embed.assert_not_called()

    async def test_project_id_in_sql_params(self) -> None:
        conn, cur = _make_conn([])
        pid = str(_PID)
        await get_facts(project_id=pid, conn=conn, redis=MagicMock())
        sql: str = cur.execute.call_args[0][0]
        assert "project_id" in sql

    async def test_tags_in_sql_params(self) -> None:
        conn, cur = _make_conn([])
        await get_facts(tags=["x"], conn=conn, redis=MagicMock())
        sql: str = cur.execute.call_args[0][0]
        assert "tags" in sql

    async def test_limit_in_params(self) -> None:
        conn, cur = _make_conn([])
        await get_facts(limit=7, conn=conn, redis=MagicMock())
        params: tuple[Any, ...] = cur.execute.call_args[0][1]
        assert 7 in params


class TestGetFactsWithAbout:
    async def test_embed_called_with_about(self) -> None:
        conn, _ = _make_conn([])
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)) as mock_embed:
            await get_facts(about="some query", conn=conn, redis=MagicMock())
            mock_embed.assert_called_once_with("some query", mock_embed.call_args[0][1])

    async def test_vec_in_sql(self) -> None:
        conn, cur = _make_conn([])
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)):
            await get_facts(about="some query", conn=conn, redis=MagicMock())
        sql: str = cur.execute.call_args[0][0]
        assert "vector" in sql

    async def test_exact_identifier_triggers_fts(self) -> None:
        conn, cur = _make_conn([])
        cur.fetchall = AsyncMock(side_effect=[[], []])
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)):
            await get_facts(about="embedding.py version 1.2.3", conn=conn, redis=MagicMock())
        assert cur.execute.call_count == 2
        fts_sql: str = cur.execute.call_args_list[1][0][0]
        assert "tsvector" in fts_sql

    async def test_no_fts_for_plain_query(self) -> None:
        conn, cur = _make_conn([])
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)):
            await get_facts(about="what is the capital", conn=conn, redis=MagicMock())
        assert cur.execute.call_count == 1

    async def test_fts_deduplicates_results(self) -> None:
        shared_id = uuid.uuid4()
        vec_row = _make_row(id=shared_id)
        fts_row = _make_row(id=shared_id)
        duplicate_extra = _make_row()
        conn, cur = _make_conn([])
        cur.fetchall = AsyncMock(side_effect=[[vec_row], [fts_row, duplicate_extra]])
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)):
            result = await get_facts(about="file.py", conn=conn, redis=MagicMock())
        assert len(result) == 2  # shared_id counted once + duplicate_extra

    async def test_returns_fact_pydantic_model(self) -> None:
        conn, _ = _make_conn([_make_row()])
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)):
            result = await get_facts(about="q", conn=conn, redis=MagicMock())
        assert isinstance(result[0], Fact)
        assert isinstance(result[0].id, uuid.UUID)
