"""Unit tests for storage/skills.py — no real DB required."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from psycopg.types.json import Jsonb

from mlms.storage.skills import (
    Skill,
    SkillInsertResult,
    get_skills,
    insert_skill,
)

_NAME = "my_skill"
_NOW = datetime(2024, 3, 10, 9, 0, 0, tzinfo=UTC)


def _make_insert_conn(max_version: int = 0) -> tuple[Any, MagicMock]:
    cur: MagicMock = MagicMock()
    cur.execute = AsyncMock(return_value=None)
    cur.fetchone = AsyncMock(return_value=(max_version,))
    conn: Any = MagicMock()
    conn.cursor.return_value = cur
    return conn, cur


def _make_get_conn(rows: list[dict[str, Any]]) -> tuple[Any, MagicMock]:
    cur: MagicMock = MagicMock()
    cur.execute = AsyncMock(return_value=None)
    cur.fetchall = AsyncMock(return_value=rows)
    conn: Any = MagicMock()
    conn.cursor.return_value = cur
    return conn, cur


def _make_skill_row(
    *,
    name: str = _NAME,
    version: int = 1,
    domain: str | None = None,
    content: str = "skill content",
    trigger_conditions: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": uuid.uuid4(),
        "name": name,
        "domain": domain,
        "content": content,
        "trigger_conditions": trigger_conditions,
        "version": version,
        "created_at": _NOW,
    }


# ── insert_skill ──────────────────────────────────────────────────────────────


class TestInsertSkill:
    async def test_returns_skill_insert_result(self) -> None:
        conn, _ = _make_insert_conn()
        result = await insert_skill(name=_NAME, content="c", conn=conn)
        assert isinstance(result, SkillInsertResult)

    async def test_new_skill_gets_version_1(self) -> None:
        conn, _ = _make_insert_conn(max_version=0)
        result = await insert_skill(name=_NAME, content="c", conn=conn)
        assert result.version == 1

    async def test_existing_skill_version_incremented(self) -> None:
        conn, _ = _make_insert_conn(max_version=1)
        result = await insert_skill(name=_NAME, content="c", conn=conn)
        assert result.version == 2

    async def test_two_inserts_same_name_create_version_1_and_2(self) -> None:
        conn1, _ = _make_insert_conn(max_version=0)
        r1 = await insert_skill(name=_NAME, content="v1 content", conn=conn1)

        conn2, _ = _make_insert_conn(max_version=1)
        r2 = await insert_skill(name=_NAME, content="v2 content", conn=conn2)

        assert r1.version == 1
        assert r2.version == 2
        assert r1.name == r2.name == _NAME
        # both produced an INSERT (execute called twice per conn: SELECT MAX + INSERT)
        assert conn1.cursor.return_value.execute.call_count == 2
        assert conn2.cursor.return_value.execute.call_count == 2

    async def test_result_name_matches_input(self) -> None:
        conn, _ = _make_insert_conn()
        result = await insert_skill(name="special_skill", content="c", conn=conn)
        assert result.name == "special_skill"

    async def test_result_id_is_uuid(self) -> None:
        conn, _ = _make_insert_conn()
        result = await insert_skill(name=_NAME, content="c", conn=conn)
        assert isinstance(result.id, uuid.UUID)

    async def test_result_created_at_is_datetime(self) -> None:
        conn, _ = _make_insert_conn()
        result = await insert_skill(name=_NAME, content="c", conn=conn)
        assert isinstance(result.created_at, datetime)

    async def test_domain_none_does_not_raise(self) -> None:
        conn, _ = _make_insert_conn()
        result = await insert_skill(name=_NAME, content="c", domain=None, conn=conn)
        assert result is not None

    async def test_trigger_conditions_none_sends_none(self) -> None:
        conn, cur = _make_insert_conn()
        await insert_skill(name=_NAME, content="c", trigger_conditions=None, conn=conn)
        # INSERT is the second execute call; tc_val is index 4 in params tuple
        insert_params: tuple[Any, ...] = cur.execute.call_args_list[1][0][1]
        assert insert_params[4] is None

    async def test_trigger_conditions_dict_wrapped_as_jsonb(self) -> None:
        conn, cur = _make_insert_conn()
        await insert_skill(
            name=_NAME,
            content="c",
            trigger_conditions={"on": "new_session"},
            conn=conn,
        )
        insert_params: tuple[Any, ...] = cur.execute.call_args_list[1][0][1]
        assert isinstance(insert_params[4], Jsonb)

    async def test_execute_called_twice_select_then_insert(self) -> None:
        conn, cur = _make_insert_conn()
        await insert_skill(name=_NAME, content="c", conn=conn)
        assert cur.execute.call_count == 2


# ── get_skills ────────────────────────────────────────────────────────────────


class TestGetSkills:
    async def test_returns_list_of_skill(self) -> None:
        conn, _ = _make_get_conn([_make_skill_row()])
        result = await get_skills(conn=conn)
        assert len(result) == 1
        assert isinstance(result[0], Skill)

    async def test_returns_empty_list_when_no_rows(self) -> None:
        conn, _ = _make_get_conn([])
        result = await get_skills(conn=conn)
        assert result == []

    async def test_name_filter_passed_in_params(self) -> None:
        conn, cur = _make_get_conn([])
        await get_skills(name="foo", conn=conn)
        params: tuple[Any, ...] = cur.execute.call_args[0][1]
        assert "foo" in params

    async def test_no_name_passes_limit_in_params(self) -> None:
        conn, cur = _make_get_conn([])
        await get_skills(limit=7, conn=conn)
        params: tuple[Any, ...] = cur.execute.call_args[0][1]
        assert 7 in params

    async def test_fields_mapped_correctly(self) -> None:
        row = _make_skill_row(
            name="sql_skill",
            version=3,
            domain="database",
            content="use indexes",
            trigger_conditions={"on": "query"},
        )
        conn, _ = _make_get_conn([row])
        result = await get_skills(conn=conn)
        s = result[0]
        assert s.name == "sql_skill"
        assert s.version == 3
        assert s.domain == "database"
        assert s.content == "use indexes"
        assert s.trigger_conditions == {"on": "query"}
        assert s.created_at == _NOW

    async def test_multiple_skills_returned(self) -> None:
        rows = [_make_skill_row(name=n) for n in ("a", "b", "c")]
        conn, _ = _make_get_conn(rows)
        result = await get_skills(conn=conn)
        assert len(result) == 3

    async def test_no_name_uses_distinct_on_query(self) -> None:
        conn, cur = _make_get_conn([])
        await get_skills(conn=conn)
        sql: str = cur.execute.call_args[0][0]
        assert "DISTINCT ON" in sql

    async def test_name_filter_uses_where_name_clause(self) -> None:
        conn, cur = _make_get_conn([])
        await get_skills(name="foo", conn=conn)
        sql: str = cur.execute.call_args[0][0]
        assert "WHERE name" in sql
        assert "DISTINCT ON" not in sql

    async def test_skill_id_is_uuid(self) -> None:
        conn, _ = _make_get_conn([_make_skill_row()])
        result = await get_skills(conn=conn)
        assert isinstance(result[0].id, uuid.UUID)


# ── edge cases ────────────────────────────────────────────────────────────────


class TestSkillEdgeCases:
    async def test_high_version_increment(self) -> None:
        conn, _ = _make_insert_conn(max_version=99)
        result = await insert_skill(name=_NAME, content="c", conn=conn)
        assert result.version == 100

    async def test_different_names_get_independent_versions(self) -> None:
        conn_a, _ = _make_insert_conn(max_version=0)
        r_a = await insert_skill(name="skill_a", content="ca", conn=conn_a)

        conn_b, _ = _make_insert_conn(max_version=0)
        r_b = await insert_skill(name="skill_b", content="cb", conn=conn_b)

        assert r_a.version == 1
        assert r_b.version == 1

    async def test_insert_never_deletes_uses_insert_sql(self) -> None:
        conn, cur = _make_insert_conn(max_version=2)
        await insert_skill(name=_NAME, content="c", conn=conn)
        for call in cur.execute.call_args_list:
            sql: str = call[0][0]
            assert "DELETE" not in sql.upper()
            assert "UPDATE" not in sql.upper()
