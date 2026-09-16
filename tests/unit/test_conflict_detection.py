"""Unit tests for conflict detection race condition scenarios.

test_facts.py covers individual-call logic; this file tests concurrency behaviour
and version chain invariants that require simulating multiple writers.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from mlms.config import COSINE_CONFLICT_THRESHOLD
from mlms.storage.facts import upsert_fact

_PROJECT = uuid.uuid4()
_VEC = [0.1] * 1536


def _make_conn(fetchone_result: dict[str, Any] | None) -> tuple[Any, MagicMock]:
    cur = MagicMock()
    cur.execute = AsyncMock(return_value=None)
    cur.fetchone = AsyncMock(return_value=fetchone_result)

    tx = AsyncMock()
    tx.__aexit__ = AsyncMock(return_value=False)

    conn: Any = MagicMock()
    conn.cursor.return_value = cur
    conn.transaction.return_value = tx
    return conn, cur


class TestConflictAtomicity:
    async def test_transaction_wraps_all_ops_on_conflict(self) -> None:
        conn, cur = _make_conn({"id": uuid.uuid4(), "version": 1, "dist": 0.01})
        await upsert_fact(content="x", project_id=_PROJECT, embedding_vec=_VEC, conn=conn)
        conn.transaction.assert_called_once()
        assert cur.execute.call_count == 4  # SET ISOLATION + SELECT + UPDATE + INSERT

    async def test_transaction_wraps_insert_only_path(self) -> None:
        conn, cur = _make_conn(None)
        await upsert_fact(content="x", project_id=_PROJECT, embedding_vec=_VEC, conn=conn)
        conn.transaction.assert_called_once()
        assert cur.execute.call_count == 3  # SET ISOLATION + SELECT + INSERT

    async def test_update_precedes_insert_in_conflict_path(self) -> None:
        conn, cur = _make_conn({"id": uuid.uuid4(), "version": 1, "dist": 0.01})
        await upsert_fact(content="x", project_id=_PROJECT, embedding_vec=_VEC, conn=conn)
        calls = [c.args[0] for c in cur.execute.call_args_list]
        assert "UPDATE" in calls[2]
        assert "INSERT" in calls[3]

    async def test_threshold_boundary_dist_exactly_1_minus_threshold(self) -> None:
        dist = round(1.0 - COSINE_CONFLICT_THRESHOLD, 10)  # 0.04 → still conflict (<=)
        old_id = uuid.uuid4()
        conn, _ = _make_conn({"id": old_id, "version": 1, "dist": dist})
        result = await upsert_fact(content="x", project_id=_PROJECT, embedding_vec=_VEC, conn=conn)
        assert result.is_conflict, f"dist={dist} must be conflict (<=, not <)"
        assert result.replaced_id == old_id


class TestRaceCondition:
    async def test_concurrent_writes_both_complete(self) -> None:
        """Two simultaneous calls on separate (mocked) connections both return valid results.

        Connections here are AsyncMock-backed with no real I/O, so this does not exercise
        deadlock-safety under genuine interleaving — only that gather() over two upsert_fact()
        coroutines resolves cleanly. DB-level serialization via transaction isolation is
        verified in integration tests.
        """
        conn_a, _ = _make_conn(None)
        conn_b, _ = _make_conn(None)

        results = await asyncio.gather(
            upsert_fact(content="fact A", project_id=_PROJECT, embedding_vec=_VEC, conn=conn_a),
            upsert_fact(content="fact B", project_id=_PROJECT, embedding_vec=_VEC, conn=conn_b),
        )

        assert len(results) == 2
        assert all(r.version >= 1 for r in results)
        assert all(r.id is not None for r in results)

    async def test_concurrent_writes_each_use_own_transaction(self) -> None:
        conn_a, _ = _make_conn(None)
        conn_b, _ = _make_conn(None)

        await asyncio.gather(
            upsert_fact(content="a", project_id=_PROJECT, embedding_vec=_VEC, conn=conn_a),
            upsert_fact(content="b", project_id=_PROJECT, embedding_vec=_VEC, conn=conn_b),
        )

        conn_a.transaction.assert_called_once()
        conn_b.transaction.assert_called_once()

    async def test_second_writer_sees_first_insert_as_conflict(self) -> None:
        """Simulate: writer A committed. Writer B's SELECT finds A's fact as near-duplicate.

        Expected: B soft-deletes A's row and inserts version 2.
        """
        first_id = uuid.uuid4()
        conn_b, cur_b = _make_conn({"id": first_id, "version": 1, "dist": 0.01})

        result = await upsert_fact(
            content="near-duplicate of A",
            project_id=_PROJECT,
            embedding_vec=_VEC,
            conn=conn_b,
        )

        assert result.version == 2
        assert result.is_conflict
        assert result.replaced_id == first_id
        update_params = cur_b.execute.call_args_list[2].args[1]
        assert first_id in update_params

    async def test_version_chain_increments_correctly(self) -> None:
        """Three sequential 'same-content' writes produce version 1 → 2 → 3."""
        conn1, _ = _make_conn(None)
        r1 = await upsert_fact(content="c", project_id=_PROJECT, embedding_vec=_VEC, conn=conn1)
        assert r1.version == 1

        conn2, _ = _make_conn({"id": r1.id, "version": 1, "dist": 0.01})
        r2 = await upsert_fact(content="c", project_id=_PROJECT, embedding_vec=_VEC, conn=conn2)
        assert r2.version == 2

        conn3, _ = _make_conn({"id": r2.id, "version": 2, "dist": 0.01})
        r3 = await upsert_fact(content="c", project_id=_PROJECT, embedding_vec=_VEC, conn=conn3)
        assert r3.version == 3

    async def test_no_physical_delete_on_conflict(self) -> None:
        """Conflict path must use UPDATE (soft-delete) not DELETE."""
        conn, cur = _make_conn({"id": uuid.uuid4(), "version": 1, "dist": 0.01})
        await upsert_fact(content="x", project_id=_PROJECT, embedding_vec=_VEC, conn=conn)
        sqls = [c.args[0] for c in cur.execute.call_args_list]
        assert not any("DELETE" in s for s in sqls)

    async def test_replaced_ids_are_different_new_ids_are_unique(self) -> None:
        """Each conflict produces a new UUID for the inserted fact."""
        old_id = uuid.uuid4()
        conn, _ = _make_conn({"id": old_id, "version": 1, "dist": 0.01})
        r = await upsert_fact(content="x", project_id=_PROJECT, embedding_vec=_VEC, conn=conn)
        assert r.id != old_id
        assert r.replaced_id == old_id
