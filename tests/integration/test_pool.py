"""Pool-boundary integration tests for the 1.2 connection-pool migration.

MLMS_INTEGRATION=1 pytest tests/integration/test_pool.py -v
"""

from __future__ import annotations

import os
import time
import uuid

import psycopg
import pytest
from psycopg_pool import AsyncConnectionPool

from mlms.config import settings
from mlms.server import PoolUnavailable, _acquire, _AppCtx

pytestmark = pytest.mark.skipif(
    os.getenv("MLMS_INTEGRATION") != "1",
    reason="Set MLMS_INTEGRATION=1 to run (requires docker compose up)",
)

_PG_DSN = settings.database_url.replace("postgresql+psycopg://", "postgresql://", 1)


async def test_pool_unavailable_fails_clean_and_bounded() -> None:
    """DB unreachable → tool boundary raises PoolUnavailable within the pool timeout,
    not an unbounded hang (documented behavior: a misconfigured/unreachable DB blocks
    checkouts until PoolTimeout — see psycopg pool docs,
    https://www.psycopg.org/psycopg3/docs/advanced/pool.html)."""
    bad_dsn = "postgresql://nobody:nobody@10.255.255.1:5432/nothing"
    pool: AsyncConnectionPool[psycopg.AsyncConnection[None]] = AsyncConnectionPool(
        bad_dsn,
        min_size=0,
        max_size=1,
        open=False,
        timeout=2,
        kwargs={"autocommit": True, "connect_timeout": 1},
    )
    await pool.open(wait=False)
    app = _AppCtx(pool=pool, redis=None)
    try:
        start = time.monotonic()
        with pytest.raises(PoolUnavailable):
            async with _acquire(app):
                pass
        elapsed = time.monotonic() - start
        assert elapsed < 10, f"pool checkout hung instead of failing bounded: {elapsed}s"
    finally:
        await pool.close()


async def test_pool_rolls_back_on_exception_in_checkout() -> None:
    """Exception inside an explicit transaction during checkout leaves no leftover
    transaction state on the connection for the next checkout."""
    pool: AsyncConnectionPool[psycopg.AsyncConnection[None]] = AsyncConnectionPool(
        _PG_DSN,
        min_size=1,
        max_size=2,
        open=False,
        timeout=10,
        kwargs={"autocommit": True},
    )
    await pool.open()
    app = _AppCtx(pool=pool, redis=None)
    marker_id = uuid.uuid4()
    try:
        with pytest.raises(RuntimeError, match="boom"):
            async with _acquire(app) as conn:
                async with conn.transaction():
                    await conn.execute(
                        "INSERT INTO projects (id, name) VALUES (%s, %s)",
                        (marker_id, "pool-rollback-test"),
                    )
                    raise RuntimeError("boom")

        async with _acquire(app) as conn:
            row = await (
                await conn.execute("SELECT 1 FROM projects WHERE id = %s", (marker_id,))
            ).fetchone()
            assert row is None, "rollback did not happen — leftover insert survived"

            state = await (
                await conn.execute(
                    "SELECT state FROM pg_stat_activity WHERE pid = pg_backend_pid()"
                )
            ).fetchone()
            assert state is not None
            assert state[0] != "idle in transaction", (
                "connection handed back with an open transaction"
            )
    finally:
        await pool.close()
