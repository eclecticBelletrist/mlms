"""Unit tests for tools/revert_fact.py — no real DB required."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mlms.tools.revert_fact import revert_fact

_PROJECT_A = uuid.uuid4()
_PROJECT_B = uuid.uuid4()
_FACT_ID = uuid.uuid4()
_PREV_ID = uuid.uuid4()


def _make_conn(
    fact_row: dict[str, Any] | None,
    restored_row: dict[str, Any] | None,
) -> Any:
    cur = MagicMock()
    cur.execute = AsyncMock(return_value=None)
    cur.fetchone = AsyncMock(side_effect=[fact_row, restored_row])

    tx = AsyncMock()
    tx.__aexit__ = AsyncMock(return_value=False)

    conn: Any = MagicMock()
    conn.cursor.return_value = cur
    conn.execute = AsyncMock(return_value=None)
    conn.transaction.return_value = tx

    return conn


class TestRevertFact:
    async def test_happy_path_returns_ok_and_reverted_to(self) -> None:
        fact_row = {"project_id": _PROJECT_A, "replaces_id": _PREV_ID}
        restored_row = {"project_id": _PROJECT_A, "replaces_id": None}
        conn = _make_conn(fact_row, restored_row)

        result = await revert_fact(fact_id=_FACT_ID, conn=conn)

        assert result["ok"] is True
        assert result["reverted_to"] == str(_PREV_ID)

    async def test_happy_path_executes_retire_then_restore(self) -> None:
        fact_row = {"project_id": _PROJECT_A, "replaces_id": _PREV_ID}
        restored_row = {"project_id": _PROJECT_A, "replaces_id": None}
        conn = _make_conn(fact_row, restored_row)

        await revert_fact(fact_id=_FACT_ID, conn=conn)

        calls = conn.execute.call_args_list
        assert len(calls) == 2
        retire_sql: str = calls[0][0][0]
        restore_sql: str = calls[1][0][0]
        assert "FALSE" in retire_sql
        assert "TRUE" in restore_sql

    async def test_happy_path_uses_transaction(self) -> None:
        fact_row = {"project_id": _PROJECT_A, "replaces_id": _PREV_ID}
        restored_row = {"project_id": _PROJECT_A, "replaces_id": None}
        conn = _make_conn(fact_row, restored_row)

        await revert_fact(fact_id=_FACT_ID, conn=conn)

        conn.transaction.assert_called_once()

    async def test_fact_not_found_raises(self) -> None:
        conn = _make_conn(None, None)
        with pytest.raises(ValueError, match=str(_FACT_ID)):
            await revert_fact(fact_id=_FACT_ID, conn=conn)

    async def test_no_replaces_id_raises(self) -> None:
        fact_row = {"project_id": _PROJECT_A, "replaces_id": None}
        conn = _make_conn(fact_row, None)
        with pytest.raises(ValueError, match="no prior version"):
            await revert_fact(fact_id=_FACT_ID, conn=conn)

    async def test_restored_fact_not_found_raises(self) -> None:
        fact_row = {"project_id": _PROJECT_A, "replaces_id": _PREV_ID}
        conn = _make_conn(fact_row, None)
        with pytest.raises(ValueError, match=str(_PREV_ID)):
            await revert_fact(fact_id=_FACT_ID, conn=conn)

    async def test_cross_project_revert_blocked(self) -> None:
        fact_row = {"project_id": _PROJECT_A, "replaces_id": _PREV_ID}
        restored_row = {"project_id": _PROJECT_B, "replaces_id": None}
        conn = _make_conn(fact_row, restored_row)
        with pytest.raises(ValueError, match="cross-project revert blocked"):
            await revert_fact(fact_id=_FACT_ID, conn=conn)
