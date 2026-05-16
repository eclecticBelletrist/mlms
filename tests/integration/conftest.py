"""Shared fixtures for integration tests — require running Docker Compose.

Set MLMS_INTEGRATION=1 to run: MLMS_INTEGRATION=1 pytest tests/integration/ -v
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import psycopg
import psycopg.rows
import pytest
import redis.asyncio as aioredis

from mlms.config import settings

_PG_DSN = settings.database_url.replace("postgresql+psycopg://", "postgresql://", 1)

pytestmark = pytest.mark.skipif(
    os.getenv("MLMS_INTEGRATION") != "1",
    reason="Set MLMS_INTEGRATION=1 to run (requires docker compose up)",
)


@pytest.fixture(scope="session")
def skip_without_infra() -> None:
    if os.getenv("MLMS_INTEGRATION") != "1":
        pytest.skip("integration tests disabled")


@pytest.fixture
async def pg() -> AsyncGenerator[psycopg.AsyncConnection[Any], None]:
    conn = await psycopg.AsyncConnection.connect(_PG_DSN, autocommit=False)
    yield conn
    await conn.rollback()
    await conn.close()


@pytest.fixture
async def rd() -> AsyncGenerator[Any, None]:
    r = aioredis.from_url(settings.redis_url, decode_responses=False)
    yield r
    await r.aclose()


@pytest.fixture
async def project_id(pg: psycopg.AsyncConnection[Any]) -> AsyncGenerator[uuid.UUID, None]:
    pid = uuid.uuid4()
    await pg.execute(
        "INSERT INTO projects (id, name) VALUES (%s, %s)",
        (pid, f"integration-test-{pid}"),
    )
    await pg.commit()
    yield pid
    await pg.execute("DELETE FROM facts WHERE project_id = %s", (pid,))
    await pg.execute("DELETE FROM projects WHERE id = %s", (pid,))
    await pg.commit()
