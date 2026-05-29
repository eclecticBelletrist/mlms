# DEPRECATED: delta=0 на synthetic_100 (тривиальный результат, 5 фактов при k=5). Используй run_hard_ab.py.
# DEPRECATED: delta=0 на synthetic_100 — Qwen3-8B устраняет semantic drift наивно, SAR не нужен
# Используй: scripts/run_hard_ab.py (synthetic_hard.json, 400 фактов, delta=+0.303)

#!/usr/bin/env python
"""A/B benchmark: naive Recall@5 vs SAR Recall@5 for semantic_drift category.

Usage:
    MLMS_INTEGRATION=1 python scripts/run_ab_benchmark.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import psycopg
import psycopg.rows
import redis.asyncio as aioredis

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlms.config import settings
from mlms.embedding import embed
from mlms.storage.facts import upsert_fact
from mlms.storage.session_log import insert_session_entry
from mlms.tools.get_facts import get_facts
from mlms.tools.get_session_log import get_session_log

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_FIXTURE = Path(__file__).parent.parent / "tests" / "fixtures" / "synthetic_100.json"
_PG_DSN = settings.database_url.replace("postgresql+psycopg://", "postgresql://", 1)


async def _seed_facts(
    facts: list[dict[str, Any]],
    project_id: uuid.UUID,
    *,
    conn: psycopg.AsyncConnection[Any],
    rd: Any,
) -> dict[str, uuid.UUID]:
    id_map: dict[str, uuid.UUID] = {}
    for fact in facts:
        vec = await embed(fact["content"], rd)
        result = await upsert_fact(
            content=fact["content"],
            project_id=project_id,
            embedding_vec=vec,
            conn=conn,
        )
        id_map[fact["id"]] = result.id
    return id_map


async def _run_ab_scenario(
    scenario: dict[str, Any],
    *,
    conn: psycopg.AsyncConnection[Any],
    rd: Any,
) -> dict[str, Any]:
    pid = uuid.uuid4()
    sid = scenario["id"]
    query: str = scenario["query"]
    relevant_ids_fixture: list[str] = scenario["expected"]["relevant_ids"]
    session_log_labels: list[str] = scenario["setup"].get("session_log", [])
    chat_id = f"ab-{sid}-{pid}"

    await conn.execute(
        "INSERT INTO projects (id, name) VALUES (%s, %s)",
        (pid, f"ab-{sid}"),
    )

    try:
        id_map = await _seed_facts(
            scenario["setup"].get("facts", []), pid, conn=conn, rd=rd
        )
        for label in session_log_labels:
            await insert_session_entry(
                chat_id=chat_id,
                entry_type="context",
                label=label,
                conn=conn,
            )
        await conn.commit()

        relevant_db_ids = {str(id_map[fid]) for fid in relevant_ids_fixture if fid in id_map}

        # Naive: raw query
        naive_facts = await get_facts(
            about=query, project_id=str(pid), limit=5, conn=conn, redis=rd
        )
        naive_recall = len({str(f.id) for f in naive_facts} & relevant_db_ids) / max(
            len(relevant_db_ids), 1
        )

        # SAR: session log labels + query
        session_entries = await get_session_log(
            chat_id=chat_id, limit=50, conn=conn, redis=rd
        )
        enriched = " ".join(e.label for e in session_entries) + " " + query
        sar_facts = await get_facts(
            about=enriched.strip(), project_id=str(pid), limit=5, conn=conn, redis=rd
        )
        sar_recall = len({str(f.id) for f in sar_facts} & relevant_db_ids) / max(
            len(relevant_db_ids), 1
        )

        log.info("  %-28s naive=%.2f  sar=%.2f  delta=%+.2f", sid, naive_recall, sar_recall, sar_recall - naive_recall)
        return {"id": sid, "naive_recall": naive_recall, "sar_recall": sar_recall}

    finally:
        await conn.execute("DELETE FROM facts WHERE project_id = %s", (pid,))
        await conn.execute("DELETE FROM session_log WHERE chat_id = %s", (chat_id,))
        await conn.execute("DELETE FROM projects WHERE id = %s", (pid,))
        await conn.commit()


async def main() -> None:
    if os.getenv("MLMS_INTEGRATION") != "1":
        print("Set MLMS_INTEGRATION=1 to run (requires docker compose up)")
        sys.exit(1)

    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    scenarios = [
        s for s in data["scenarios"]
        if s["category"] == "semantic_drift" and s["meta"].get("ab_test")
    ]
    log.info("Running A/B benchmark on %d semantic_drift scenarios", len(scenarios))

    conn = await psycopg.AsyncConnection.connect(_PG_DSN, autocommit=False)
    rd = aioredis.from_url(settings.redis_url, decode_responses=False)

    results: list[dict[str, Any]] = []
    try:
        for i, scenario in enumerate(scenarios, 1):
            log.info("[%2d/%d] %s", i, len(scenarios), scenario["id"])
            try:
                r = await _run_ab_scenario(scenario, conn=conn, rd=rd)
                results.append(r)
            except Exception as exc:
                log.error("  ERROR in %s: %s", scenario["id"], exc, exc_info=True)
                results.append({"id": scenario["id"], "naive_recall": 0.0, "sar_recall": 0.0})
    finally:
        await conn.close()
        await rd.aclose()

    n = len(results)
    naive_avg = sum(r["naive_recall"] for r in results) / n
    sar_avg = sum(r["sar_recall"] for r in results) / n
    delta = sar_avg - naive_avg

    naive_wins = sum(1 for r in results if r["sar_recall"] > r["naive_recall"])
    ties = sum(1 for r in results if r["sar_recall"] == r["naive_recall"])
    naive_better = sum(1 for r in results if r["naive_recall"] > r["sar_recall"])

    print()
    print("=" * 55)
    print("SAR A/B BENCHMARK — semantic_drift (n={})".format(n))
    print("=" * 55)
    print(f"  {'Mode':<28} {'Recall@5':>8}")
    print(f"  {'-'*28} {'-'*8}")
    print(f"  {'Naive (raw query)':<28} {naive_avg:>8.3f}")
    print(f"  {'SAR (session-enriched)':<28} {sar_avg:>8.3f}")
    print(f"  {'Delta (SAR - naive)':<28} {delta:>+8.3f}")
    print()
    print(f"  SAR better: {naive_wins}/{n}  tie: {ties}/{n}  naive better: {naive_better}/{n}")
    sar_status = "PASS (SAR > naive)" if delta > 0 else "FAIL (no improvement)"
    print(f"  SAR proof: {sar_status}")
    print("=" * 55)

    out = Path(__file__).parent / "ab_results.json"
    out.write_text(
        json.dumps({
            "n": n,
            "naive_avg": round(naive_avg, 4),
            "sar_avg": round(sar_avg, 4),
            "delta": round(delta, 4),
            "sar_better": naive_wins,
            "tie": ties,
            "naive_better": naive_better,
            "scenarios": results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Results saved -> %s", out)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
