"""Integration: conflict detection, version chains, and revert_fact atomicity.

MLMS_INTEGRATION=1 pytest tests/integration/test_amend.py -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import psycopg
import psycopg.rows
import pytest

from mlms.storage.facts import upsert_fact
from mlms.tools.revert_fact import revert_fact

pytestmark = pytest.mark.skipif(
    os.getenv("MLMS_INTEGRATION") != "1",
    reason="Set MLMS_INTEGRATION=1 to run (requires docker compose up)",
)

_DIM = 1536
# Orthogonal unit vectors — cosine similarity = 0 → NO conflict (0 < 0.96)
_VEC_ALPHA: list[float] = [1.0] + [0.0] * (_DIM - 1)
_VEC_BETA: list[float]  = [0.0, 1.0] + [0.0] * (_DIM - 2)

# Identical direction — cosine distance = 0 → CONFLICT (0 ≤ 1 - 0.96 = 0.04)
_VEC_SAME: list[float] = [1.0] + [0.0] * (_DIM - 1)


async def _fetch(conn: psycopg.AsyncConnection[Any], fact_id: uuid.UUID) -> dict[str, Any]:
    cur = conn.cursor(row_factory=psycopg.rows.dict_row)
    await cur.execute(
        "SELECT id, version, is_current, replaces_id FROM facts WHERE id = %s",
        (fact_id,),
    )
    row = await cur.fetchone()
    assert row is not None, f"fact {fact_id} not found"
    return row


class TestConflictDetection:
    async def test_orthogonal_both_current(
        self, pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID
    ) -> None:
        """cosine < 0.96 → both facts remain is_current=TRUE."""
        r1 = await upsert_fact(
            content="A", project_id=project_id, embedding_vec=_VEC_ALPHA, conn=pg
        )
        r2 = await upsert_fact(
            content="B", project_id=project_id, embedding_vec=_VEC_BETA, conn=pg
        )
        assert (await _fetch(pg, r1.id))["is_current"] is True
        assert (await _fetch(pg, r2.id))["is_current"] is True
        assert r1.is_conflict is False
        assert r2.is_conflict is False

    async def test_parallel_retires_predecessor(
        self, pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID
    ) -> None:
        """cosine ≥ 0.96 → old fact is_current=FALSE, new is_current=TRUE."""
        r1 = await upsert_fact(
            content="v1", project_id=project_id, embedding_vec=_VEC_ALPHA, conn=pg
        )
        r2 = await upsert_fact(
            content="v2", project_id=project_id, embedding_vec=_VEC_SAME, conn=pg
        )
        assert (await _fetch(pg, r1.id))["is_current"] is False
        assert (await _fetch(pg, r2.id))["is_current"] is True
        assert r2.is_conflict is True
        assert r2.replaced_id == r1.id

    async def test_three_version_replaces_chain(
        self, pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID
    ) -> None:
        """v1 ← v2 ← v3: replaces_id forms a backward chain through all three versions."""
        r1 = await upsert_fact(
            content="v1", project_id=project_id, embedding_vec=_VEC_ALPHA, conn=pg
        )
        r2 = await upsert_fact(
            content="v2", project_id=project_id, embedding_vec=_VEC_SAME, conn=pg
        )
        r3 = await upsert_fact(
            content="v3", project_id=project_id, embedding_vec=_VEC_SAME, conn=pg
        )

        row2 = await _fetch(pg, r2.id)
        row3 = await _fetch(pg, r3.id)

        assert row2["replaces_id"] == r1.id, "v2 must point back to v1"
        assert row3["replaces_id"] == r2.id, "v3 must point back to v2"
        assert row3["is_current"] is True

        # Only one current fact per semantic cluster
        cur = pg.cursor(row_factory=psycopg.rows.dict_row)
        await cur.execute(
            "SELECT COUNT(*) AS n FROM facts WHERE project_id = %s AND is_current = TRUE",
            (project_id,),
        )
        row = await cur.fetchone()
        assert row["n"] == 1


class TestRevert:
    async def test_revert_v3_restores_v2(
        self, pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID
    ) -> None:
        """revert_fact(v3) → v2 is_current=TRUE, v3 is_current=FALSE (atomic)."""
        r1 = await upsert_fact(
            content="v1", project_id=project_id, embedding_vec=_VEC_ALPHA, conn=pg
        )
        r2 = await upsert_fact(
            content="v2", project_id=project_id, embedding_vec=_VEC_SAME, conn=pg
        )
        r3 = await upsert_fact(
            content="v3", project_id=project_id, embedding_vec=_VEC_SAME, conn=pg
        )

        result = await revert_fact(fact_id=r3.id, conn=pg)

        assert result["ok"] is True
        assert result["reverted_to"] == str(r2.id)
        assert (await _fetch(pg, r3.id))["is_current"] is False
        assert (await _fetch(pg, r2.id))["is_current"] is True
        # v1 stays retired — revert only steps back one version
        assert (await _fetch(pg, r1.id))["is_current"] is False

    async def test_revert_no_predecessor_raises(
        self, pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID
    ) -> None:
        """revert_fact on version-1 fact (no replaces_id) → ValueError."""
        r1 = await upsert_fact(
            content="only-version", project_id=project_id, embedding_vec=_VEC_ALPHA, conn=pg
        )
        with pytest.raises(ValueError, match="no prior version to revert to"):
            await revert_fact(fact_id=r1.id, conn=pg)
