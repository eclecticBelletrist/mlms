"""Unit tests for storage/facts.py — no real DB required."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from mlms.config import COSINE_CONFLICT_THRESHOLD
from mlms.storage.facts import FactResult, _vec_literal, upsert_fact

_PROJECT = uuid.uuid4()
_EMBEDDING = [0.1] * 1536


# ── _vec_literal ─────────────────────────────────────────────────────────────


class TestVecLiteral:
    def test_format(self) -> None:
        assert _vec_literal([0.1, 0.2, 0.3]) == "[0.1,0.2,0.3]"

    def test_round_trips(self) -> None:
        vec = [float(i) for i in range(10)]
        parts = _vec_literal(vec).strip("[]").split(",")
        assert [float(p) for p in parts] == vec


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_conn(fetchone_result: dict[str, Any] | None) -> tuple[Any, MagicMock]:
    """Return (mock_conn, mock_cursor) with psycopg3 async contract wired up."""
    cur = MagicMock()
    cur.execute = AsyncMock(return_value=None)
    cur.fetchone = AsyncMock(return_value=fetchone_result)

    # transaction() must be an async context manager that does NOT suppress exceptions
    tx = AsyncMock()
    tx.__aexit__ = AsyncMock(return_value=False)

    conn: Any = MagicMock()
    conn.cursor.return_value = cur
    conn.transaction.return_value = tx

    return conn, cur


# ── upsert_fact ───────────────────────────────────────────────────────────────


class TestUpsertFact:
    async def test_no_existing_facts_inserts_version_1(self) -> None:
        conn, _ = _make_conn(None)

        result = await upsert_fact(
            content="sky is blue",
            project_id=_PROJECT,
            embedding_vec=_EMBEDDING,
            conn=conn,
        )

        assert result.version == 1
        assert not result.is_conflict
        assert result.replaced_id is None

    async def test_below_threshold_no_conflict(self) -> None:
        """dist=0.10 → similarity=0.90 < 0.96 → no conflict."""
        conn, _ = _make_conn({"id": uuid.uuid4(), "version": 1, "dist": 0.10})

        result = await upsert_fact(
            content="sky is blue",
            project_id=_PROJECT,
            embedding_vec=_EMBEDDING,
            conn=conn,
        )

        assert result.version == 1
        assert not result.is_conflict
        assert result.replaced_id is None

    async def test_conflict_updates_old_inserts_new(self) -> None:
        """dist=0.02 → similarity=0.98 >= 0.96 → conflict; version incremented."""
        old_id = uuid.uuid4()
        conn, cur = _make_conn({"id": old_id, "version": 2, "dist": 0.02})

        result = await upsert_fact(
            content="sky is blue",
            project_id=_PROJECT,
            embedding_vec=_EMBEDDING,
            conn=conn,
        )

        assert result.version == 3
        assert result.is_conflict
        assert result.replaced_id == old_id

    async def test_conflict_at_exact_threshold_boundary(self) -> None:
        """dist == 1 - threshold is still a conflict (<=, not <)."""
        dist = round(1.0 - COSINE_CONFLICT_THRESHOLD, 10)  # 0.04
        old_id = uuid.uuid4()
        conn, _ = _make_conn({"id": old_id, "version": 1, "dist": dist})

        result = await upsert_fact(
            content="sky is blue",
            project_id=_PROJECT,
            embedding_vec=_EMBEDDING,
            conn=conn,
        )

        assert result.is_conflict
        assert result.replaced_id == old_id

    async def test_conflict_calls_update_then_insert(self) -> None:
        """On conflict: SELECT → UPDATE → INSERT (order matters)."""
        conn, cur = _make_conn({"id": uuid.uuid4(), "version": 1, "dist": 0.01})

        await upsert_fact(
            content="sky is blue",
            project_id=_PROJECT,
            embedding_vec=_EMBEDDING,
            conn=conn,
        )

        assert cur.execute.call_count == 3
        sql_calls = [c.args[0] for c in cur.execute.call_args_list]
        assert "SELECT" in sql_calls[0]
        assert "UPDATE" in sql_calls[1]
        assert "INSERT" in sql_calls[2]

    async def test_no_conflict_skips_update(self) -> None:
        conn, cur = _make_conn(None)

        await upsert_fact(
            content="sky is blue",
            project_id=_PROJECT,
            embedding_vec=_EMBEDDING,
            conn=conn,
        )

        assert cur.execute.call_count == 2
        sql_calls = [c.args[0] for c in cur.execute.call_args_list]
        assert not any("UPDATE" in s for s in sql_calls)

    async def test_returns_fact_result_with_uuid(self) -> None:
        conn, _ = _make_conn(None)

        result = await upsert_fact(
            content="fact",
            project_id=_PROJECT,
            embedding_vec=_EMBEDDING,
            conn=conn,
        )

        assert isinstance(result, FactResult)
        assert isinstance(result.id, uuid.UUID)

    async def test_transaction_entered_once(self) -> None:
        conn, _ = _make_conn(None)

        await upsert_fact(
            content="fact",
            project_id=_PROJECT,
            embedding_vec=_EMBEDDING,
            conn=conn,
        )

        conn.transaction.assert_called_once()

    async def test_custom_threshold_widens_conflict_zone(self) -> None:
        """dist=0.05 → sim=0.95. Default 0.96: no conflict. Custom 0.94: conflict."""
        old_id = uuid.uuid4()
        conn, _ = _make_conn({"id": old_id, "version": 1, "dist": 0.05})

        result = await upsert_fact(
            content="fact",
            project_id=_PROJECT,
            embedding_vec=_EMBEDDING,
            conn=conn,
            threshold=0.94,
        )

        assert result.is_conflict

    async def test_tags_and_category_passed_to_insert(self) -> None:
        conn, cur = _make_conn(None)

        await upsert_fact(
            content="fact",
            project_id=_PROJECT,
            embedding_vec=_EMBEDDING,
            tags=["science", "history"],
            category="knowledge",
            conn=conn,
        )

        # INSERT is the last execute call; params = (id, content, vec_lit, tags, category, ...)
        insert_params = cur.execute.call_args_list[-1].args[1]
        assert insert_params[3] == ["science", "history"]
        assert insert_params[4] == "knowledge"

    async def test_null_dist_treated_as_no_conflict(self) -> None:
        """Defensive: if embedding IS NULL in DB, dist will be None — not a conflict."""
        conn, _ = _make_conn({"id": uuid.uuid4(), "version": 1, "dist": None})

        result = await upsert_fact(
            content="fact",
            project_id=_PROJECT,
            embedding_vec=_EMBEDDING,
            conn=conn,
        )

        assert not result.is_conflict
        assert result.version == 1
