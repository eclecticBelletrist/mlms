"""Integration: memorize(type='fact') → conflict → is_current in real DB.

Run with: MLMS_INTEGRATION=1 pytest tests/integration/test_conflict_flow.py -v
"""

from __future__ import annotations

import math
import os
import random
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import psycopg
import psycopg.rows
import pytest

from mlms.config import EMBEDDING_DIM
from mlms.storage._vec import vec_literal
from mlms.storage.facts import upsert_fact
from mlms.tools.memorize import memorize

pytestmark = pytest.mark.skipif(
    os.getenv("MLMS_INTEGRATION") != "1",
    reason="Set MLMS_INTEGRATION=1 to run (requires docker compose up)",
)

_VEC_A = [0.9] + [0.0] * 1535  # unit vector along dim-0
_VEC_B = [0.9999] + [0.0] * 1535  # nearly identical → conflict (sim ≈ 1.0)
_VEC_C = [0.0, 0.9] + [0.0] * 1534  # orthogonal → no conflict


async def _fetch_facts(
    conn: psycopg.AsyncConnection[Any], project_id: uuid.UUID
) -> list[dict[str, Any]]:
    cur = conn.cursor(row_factory=psycopg.rows.dict_row)
    await cur.execute(
        "SELECT id, version, is_current FROM facts WHERE project_id = %s ORDER BY version",
        (project_id,),
    )
    return await cur.fetchall()


class TestConflictFlowDirect:
    async def test_first_insert_creates_version_1_is_current(
        self, pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID
    ) -> None:
        await upsert_fact(content="fact A", project_id=project_id, embedding_vec=_VEC_A, conn=pg)
        await pg.commit()

        rows = await _fetch_facts(pg, project_id)
        assert len(rows) == 1
        assert rows[0]["version"] == 1
        assert rows[0]["is_current"] is True

    async def test_similar_fact_triggers_conflict_soft_deletes_old(
        self, pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID
    ) -> None:
        r1 = await upsert_fact(
            content="fact A", project_id=project_id, embedding_vec=_VEC_A, conn=pg
        )
        await pg.commit()

        r2 = await upsert_fact(
            content="fact A v2", project_id=project_id, embedding_vec=_VEC_B, conn=pg
        )
        await pg.commit()

        assert r2.is_conflict
        assert r2.version == 2
        assert r2.replaced_id == r1.id

        rows = await _fetch_facts(pg, project_id)
        assert len(rows) == 2
        old = next(r for r in rows if r["id"] == r1.id)
        new = next(r for r in rows if r["id"] == r2.id)
        assert old["is_current"] is False
        assert new["is_current"] is True

    async def test_dissimilar_fact_no_conflict_both_current(
        self, pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID
    ) -> None:
        await upsert_fact(content="fact A", project_id=project_id, embedding_vec=_VEC_A, conn=pg)
        r2 = await upsert_fact(
            content="fact C", project_id=project_id, embedding_vec=_VEC_C, conn=pg
        )
        await pg.commit()

        assert not r2.is_conflict
        assert r2.version == 1

        rows = await _fetch_facts(pg, project_id)
        assert all(r["is_current"] for r in rows)
        assert len(rows) == 2

    async def test_version_chain_no_physical_delete(
        self, pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID
    ) -> None:
        """After 3 conflict writes, DB has 3 rows; only latest is_current=TRUE."""
        await upsert_fact(content="v1", project_id=project_id, embedding_vec=_VEC_A, conn=pg)
        await pg.commit()
        await upsert_fact(content="v2", project_id=project_id, embedding_vec=_VEC_B, conn=pg)
        await pg.commit()
        await upsert_fact(content="v3", project_id=project_id, embedding_vec=_VEC_A, conn=pg)
        await pg.commit()

        rows = await _fetch_facts(pg, project_id)
        assert len(rows) == 3
        current = [r for r in rows if r["is_current"]]
        assert len(current) == 1
        assert current[0]["version"] == 3


class TestConflictFlowViaMemoize:
    """Full path: memorize(type='fact') → embed → upsert_fact → DB."""

    async def test_memorize_fact_creates_db_entry(
        self, pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID, rd: Any
    ) -> None:
        with patch("mlms.tools.memorize.embed", new=AsyncMock(return_value=_VEC_A)):
            result = await memorize(
                content="Python uses reference counting for memory management",
                type="fact",
                metadata={"project_id": str(project_id)},
                conn=pg,
                redis=rd,
            )
        await pg.commit()

        assert result["ok"] is True
        rows = await _fetch_facts(pg, project_id)
        assert len(rows) == 1

    async def test_memorize_fact_conflict_increments_version(
        self, pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID, rd: Any
    ) -> None:
        with patch("mlms.tools.memorize.embed", new=AsyncMock(return_value=_VEC_A)):
            await memorize(
                content="fact A",
                type="fact",
                metadata={"project_id": str(project_id)},
                conn=pg,
                redis=rd,
            )
        await pg.commit()

        with patch("mlms.tools.memorize.embed", new=AsyncMock(return_value=_VEC_B)):
            await memorize(
                content="fact A revised",
                type="fact",
                metadata={"project_id": str(project_id)},
                conn=pg,
                redis=rd,
            )
        await pg.commit()

        rows = await _fetch_facts(pg, project_id)
        assert len(rows) == 2
        assert sum(1 for r in rows if r["is_current"]) == 1
        assert max(r["version"] for r in rows) == 2


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _random_unit(rnd: random.Random) -> list[float]:
    return _unit([rnd.gauss(0, 1) for _ in range(EMBEDDING_DIM)])


def _at_cosine(q: list[float], cos: float, rnd: random.Random) -> list[float]:
    r = _random_unit(rnd)
    d = sum(a * b for a, b in zip(r, q, strict=True))
    ortho = _unit([a - d * b for a, b in zip(r, q, strict=True)])
    s = math.sqrt(1 - cos * cos)
    return [cos * a + s * b for a, b in zip(q, ortho, strict=True)]


class TestConflictLookupInLargeProject:
    """The HNSW candidates of the nearest-fact lookup can all belong to another project."""

    async def test_conflict_found_when_other_project_holds_the_nearest_rows(
        self, pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID
    ) -> None:
        rnd = random.Random(7)
        query = _random_unit(rnd)
        other = uuid.uuid4()
        insert = (
            "INSERT INTO facts (id, content, embedding, project_id, version, is_current, updated_at)"
            " VALUES (%s, %s, %s::vector, %s, 1, TRUE, now())"
        )
        await pg.execute(
            "INSERT INTO projects (id, name) VALUES (%s, %s)", (other, f"other-{other}")
        )
        try:
            async with pg.cursor() as cur:
                await cur.executemany(
                    insert,
                    [
                        (uuid.uuid4(), "filler", vec_literal(_random_unit(rnd)), project_id)
                        for _ in range(800)
                    ],
                )
                await cur.executemany(
                    insert,
                    [
                        (uuid.uuid4(), "near", vec_literal(_at_cosine(query, 0.99, rnd)), other)
                        for _ in range(100)
                    ],
                )
            old = await upsert_fact(
                content="target",
                project_id=project_id,
                embedding_vec=_at_cosine(query, 0.97, rnd),
                conn=pg,
            )
            await pg.execute("ANALYZE facts")
            await pg.commit()

            plan = await (
                await pg.execute(
                    "EXPLAIN SELECT id FROM facts WHERE project_id = %s AND is_current = TRUE"
                    " ORDER BY embedding <=> %s::vector LIMIT 1",
                    (project_id, vec_literal(query)),
                )
            ).fetchall()
            await pg.commit()
            assert any("facts_embedding_idx" in row[0] for row in plan)

            new = await upsert_fact(
                content="target revised", project_id=project_id, embedding_vec=query, conn=pg
            )
            await pg.commit()

            assert new.is_conflict
            assert new.replaced_id == old.id
        finally:
            await pg.rollback()
            await pg.execute("DELETE FROM facts WHERE project_id = %s", (other,))
            await pg.execute("DELETE FROM projects WHERE id = %s", (other,))
            await pg.commit()
