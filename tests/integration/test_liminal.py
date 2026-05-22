"""Edge-case integration tests: data durability, project isolation, Redis degradation.

MLMS_INTEGRATION=1 pytest tests/integration/test_liminal.py -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import psycopg
import psycopg.rows
import pytest

from mlms.storage.facts import upsert_fact
from mlms.tools.get_facts import get_facts

pytestmark = pytest.mark.skipif(
    os.getenv("MLMS_INTEGRATION") != "1",
    reason="Set MLMS_INTEGRATION=1 to run (requires docker compose up)",
)

_DIM = 1536
_VEC_A: list[float] = [1.0] + [0.0] * (_DIM - 1)
_VEC_LIT = "[" + ",".join(str(x) for x in _VEC_A) + "]"


async def test_facts_survive_redis_expire(
    pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID, rd: Any
) -> None:
    """PostgreSQL is the durable store — Redis flush must not lose any facts."""
    await upsert_fact(
        content="durable fact", project_id=project_id, embedding_vec=_VEC_A, conn=pg
    )

    # Simulate all Redis keys expiring
    await rd.flushdb()

    # Non-semantic query: bypasses embed() entirely, reads straight from PostgreSQL
    facts = await get_facts(project_id=str(project_id), conn=pg, redis=rd)
    assert any(f.content == "durable fact" for f in facts)


async def test_project_isolation(
    pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID, rd: Any
) -> None:
    """Facts scoped to project A must not appear in project B queries."""
    pid_b = uuid.uuid4()
    try:
        await pg.execute(
            "INSERT INTO projects (id, name) VALUES (%s, %s)",
            (pid_b, f"isolation-b-{pid_b}"),
        )

        await upsert_fact(
            content="project A only", project_id=project_id, embedding_vec=_VEC_A, conn=pg
        )

        facts_b = await get_facts(project_id=str(pid_b), conn=pg, redis=rd)
        assert len(facts_b) == 0, "fact from project A leaked into project B"

        facts_a = await get_facts(project_id=str(project_id), conn=pg, redis=rd)
        assert any(f.content == "project A only" for f in facts_a)
    finally:
        # Clean up project B before project_id fixture teardown commits
        await pg.execute("DELETE FROM projects WHERE id = %s", (pid_b,))


async def test_redis_down_get_facts_postgres_fallback(
    pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID
) -> None:
    """get_facts(about=None) never calls embed() — works even when Redis is unreachable."""
    await pg.execute(
        "INSERT INTO facts (project_id, content, embedding, version, is_current)"
        " VALUES (%s, 'postgres-direct', %s::vector, 1, TRUE)",
        (project_id, _VEC_LIT),
    )

    broken_redis = MagicMock()
    broken_redis.get = AsyncMock(side_effect=ConnectionError("Redis is down"))
    broken_redis.setex = AsyncMock(side_effect=ConnectionError("Redis is down"))
    broken_redis.hgetall = AsyncMock(side_effect=ConnectionError("Redis is down"))

    # about=None → no embed() call → broken_redis is never touched
    facts = await get_facts(project_id=str(project_id), conn=pg, redis=broken_redis)
    assert any(f.content == "postgres-direct" for f in facts)
