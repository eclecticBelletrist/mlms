#!/usr/bin/env python
"""Latency benchmark: p50/p95/p99 for NFR-1 compliance.

Usage:
    MLMS_INTEGRATION=1 python scripts/measure_latency.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import psycopg.rows
import redis.asyncio as aioredis

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlms.config import LATENCY_P99_TARGET_MS, LATENCY_SESSION_CTX_MS, settings
from mlms.embedding import embed
from mlms.storage.facts import upsert_fact
from mlms.storage.session_ctx import SessionContext, set_session_context
from mlms.tools.get_facts import get_facts
from mlms.tools.get_session_ctx import get_session_context

_PG_DSN = settings.database_url.replace("postgresql+psycopg://", "postgresql://", 1)
_N = 100


def _pct(values: list[float], p: int) -> float:
    s = sorted(values)
    idx = max(0, min(int(len(s) * p / 100), len(s) - 1))
    return s[idx]


async def bench_session_context(rd: Any) -> list[float]:
    chat_id = f"bench-ctx-{uuid.uuid4()}"
    ctx = SessionContext(
        project_id=str(uuid.uuid4()),
        last_activity=datetime.now(UTC),
        open_todos=["fix bug"],
        active_entities=["auth.py"],
        recent_decisions=["use postgres"],
    )
    await set_session_context(chat_id, ctx, redis=rd)

    times: list[float] = []
    for _ in range(_N):
        t0 = time.perf_counter()
        await get_session_context(chat_id=chat_id, redis=rd)
        times.append((time.perf_counter() - t0) * 1000.0)

    await rd.delete(f"session:{chat_id}")
    return times


async def bench_get_facts(conn: psycopg.AsyncConnection[Any], rd: Any) -> list[float]:
    pid = uuid.uuid4()
    await conn.execute(
        "INSERT INTO projects (id, name) VALUES (%s, %s)", (pid, f"bench-gf-{pid}")
    )

    seeds = [
        "User prefers PostgreSQL for relational data storage",
        "Redis handles session state with 48-hour TTL",
        "pgvector enables cosine ANN search via HNSW index",
        "All embedding vectors use 1536 dimensions (Matryoshka)",
        "Docker Compose orchestrates all infrastructure locally",
        "Alembic manages schema migrations incrementally",
        "MCP protocol connects LLM agents to persistent memory tools",
        "pytest covers unit and integration test suites",
        "TimescaleDB partitions timeline events hypertable by time",
        "Conflict detection applies cosine threshold 0.96 for deduplication",
    ]
    for text in seeds:
        vec = await embed(text, rd)
        await upsert_fact(content=text, project_id=pid, embedding_vec=vec, conn=conn)
    await conn.commit()

    query = "database preferences and infrastructure configuration"
    await embed(query, rd)  # warm query embedding cache

    times: list[float] = []
    for _ in range(_N):
        t0 = time.perf_counter()
        await get_facts(about=query, project_id=str(pid), limit=5, conn=conn, redis=rd)
        times.append((time.perf_counter() - t0) * 1000.0)

    await conn.execute("DELETE FROM facts WHERE project_id = %s", (pid,))
    await conn.execute("DELETE FROM projects WHERE id = %s", (pid,))
    await conn.commit()
    return times


async def bench_memorize_cold(conn: psycopg.AsyncConnection[Any], rd: Any) -> list[float]:
    """Full memorize cold path: unique content per iteration forces API embed call."""
    pid = uuid.uuid4()
    await conn.execute(
        "INSERT INTO projects (id, name) VALUES (%s, %s)", (pid, f"bench-cold-{pid}")
    )
    await conn.commit()

    times: list[float] = []
    for i in range(_N):
        content = f"Cold-path latency benchmark fact iteration={i} uid={uuid.uuid4()}"
        t0 = time.perf_counter()
        vec = await embed(content, rd)
        await upsert_fact(content=content, project_id=pid, embedding_vec=vec, conn=conn)
        await conn.commit()
        times.append((time.perf_counter() - t0) * 1000.0)

    await conn.execute("DELETE FROM facts WHERE project_id = %s", (pid,))
    await conn.execute("DELETE FROM projects WHERE id = %s", (pid,))
    await conn.commit()
    return times


async def main() -> None:
    if os.getenv("MLMS_INTEGRATION") != "1":
        print("Set MLMS_INTEGRATION=1 to run (requires docker compose up)")
        sys.exit(1)

    conn = await psycopg.AsyncConnection.connect(_PG_DSN, autocommit=False)
    # decode_responses=True required for session_ctx.hgetall; works for embedding cache too
    rd = aioredis.from_url(settings.redis_url, decode_responses=True)

    print(f"Iterations per operation: {_N}")
    print("Note: cold path benchmark makes 100 live embedding API calls (~5-10 min)")
    print()

    results: dict[str, Any] = {}

    try:
        print("[1/3] get_session_context() — Redis cached path...")
        sc = await bench_session_context(rd)
        results["get_session_context"] = {
            "p50": round(_pct(sc, 50), 2),
            "p95": round(_pct(sc, 95), 2),
            "p99": round(_pct(sc, 99), 2),
            "target_ms": LATENCY_SESSION_CTX_MS,
        }
        r = results["get_session_context"]
        print(f"  p50={r['p50']:.1f}ms  p95={r['p95']:.1f}ms  p99={r['p99']:.1f}ms")

        print("[2/3] get_facts() — warm embedding cache...")
        gf = await bench_get_facts(conn, rd)
        results["get_facts_cached"] = {
            "p50": round(_pct(gf, 50), 2),
            "p95": round(_pct(gf, 95), 2),
            "p99": round(_pct(gf, 99), 2),
            "target_ms": LATENCY_P99_TARGET_MS,
        }
        r = results["get_facts_cached"]
        print(f"  p50={r['p50']:.1f}ms  p95={r['p95']:.1f}ms  p99={r['p99']:.1f}ms")

        print("[3/3] memorize(fact) — cold path (API embed + DB upsert + commit)...")
        mc = await bench_memorize_cold(conn, rd)
        results["memorize_cold"] = {
            "p50": round(_pct(mc, 50), 2),
            "p95": round(_pct(mc, 95), 2),
            "p99": round(_pct(mc, 99), 2),
            "target_ms": None,
        }
        r = results["memorize_cold"]
        print(f"  p50={r['p50']:.1f}ms  p95={r['p95']:.1f}ms  p99={r['p99']:.1f}ms")

    finally:
        await conn.close()
        await rd.aclose()

    print()
    print("=" * 68)
    print("LATENCY BENCHMARK RESULTS (ms)")
    print("=" * 68)
    print(f"  {'Operation':<28} {'p50':>7} {'p95':>7} {'p99':>7}  Result")
    print(f"  {'-'*28} {'-'*7} {'-'*7} {'-'*7}  ------")
    for op, r in results.items():
        p99 = r["p99"]
        t = r["target_ms"]
        status = ("PASS" if p99 <= t else "FAIL") if t is not None else "N/A "
        tstr = f"  [<={t}ms]" if t is not None else "  [no SLA]"
        print(f"  {op:<28} {r['p50']:>7.1f} {r['p95']:>7.1f} {p99:>7.1f}  {status}{tstr}")
    print("=" * 68)

    out = Path(__file__).parent / "latency_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved -> {out}")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
