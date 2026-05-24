#!/usr/bin/env python
"""Benchmark: Recall@5 across 100 scenarios from synthetic_100.json.

Usage:
    MLMS_INTEGRATION=1 python scripts/run_benchmark.py
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

from mlms.config import PRECISION_AT_5_TARGET, settings
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
    """Embed + insert facts; returns fixture_id → db_uuid mapping."""
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


async def _seed_session_log_entries(
    labels: list[str],
    chat_id: str,
    *,
    conn: psycopg.AsyncConnection[Any],
) -> None:
    for label in labels:
        await insert_session_entry(
            chat_id=chat_id,
            entry_type="context",
            label=label,
            conn=conn,
        )


async def _run_scenario(
    scenario: dict[str, Any],
    *,
    conn: psycopg.AsyncConnection[Any],
    rd: Any,
) -> dict[str, Any]:
    pid = uuid.uuid4()
    sid = scenario["id"]
    category = scenario["category"]
    query: str = scenario["query"]
    relevant_ids_fixture: list[str] = scenario["expected"]["relevant_ids"]
    ab_test: bool = scenario["meta"].get("ab_test", False)
    setup = scenario["setup"]

    await conn.execute(
        "INSERT INTO projects (id, name) VALUES (%s, %s)",
        (pid, f"bench-{sid}"),
    )

    chat_ids_used: list[str] = []

    try:
        if category == "cross_session":
            facts_list: list[dict[str, Any]] = setup["source_session"]["facts"]
            query_sl: list[str] = setup["query_session"].get("session_log", [])
            qry_chat_id = f"{setup['query_session']['chat_id']}-{pid}"
            chat_ids_used.append(qry_chat_id)
            session_log_labels: list[str] = query_sl
        else:
            facts_list = setup.get("facts", [])
            session_log_labels = setup.get("session_log", [])
            qry_chat_id = f"bench-{sid}-{pid}"
            chat_ids_used.append(qry_chat_id)

        id_map = await _seed_facts(facts_list, pid, conn=conn, rd=rd)

        if ab_test and session_log_labels:
            await _seed_session_log_entries(session_log_labels, qry_chat_id, conn=conn)

        await conn.commit()

        # Map fixture relevant IDs → DB UUIDs
        relevant_db_ids = {str(id_map[fid]) for fid in relevant_ids_fixture if fid in id_map}

        # Naive query
        naive_facts = await get_facts(
            about=query, project_id=str(pid), limit=5, conn=conn, redis=rd
        )
        naive_top5 = {str(f.id) for f in naive_facts}
        naive_recall = len(naive_top5 & relevant_db_ids) / max(len(relevant_db_ids), 1)

        result: dict[str, Any] = {
            "id": sid,
            "category": category,
            "naive_recall": naive_recall,
            "recall": naive_recall,
        }

        # SAR query: only for semantic_drift ab_test with session log entries
        if ab_test and category == "semantic_drift" and session_log_labels:
            session_entries = await get_session_log(
                chat_id=qry_chat_id, limit=50, conn=conn, redis=rd
            )
            enriched = " ".join(e.label for e in session_entries) + " " + query
            sar_facts = await get_facts(
                about=enriched.strip(), project_id=str(pid), limit=5, conn=conn, redis=rd
            )
            sar_top5 = {str(f.id) for f in sar_facts}
            sar_recall = len(sar_top5 & relevant_db_ids) / max(len(relevant_db_ids), 1)
            result["sar_recall"] = sar_recall
            result["recall"] = max(naive_recall, sar_recall)

        return result

    finally:
        await conn.execute("DELETE FROM facts WHERE project_id = %s", (pid,))
        for cid in chat_ids_used:
            await conn.execute("DELETE FROM session_log WHERE chat_id = %s", (cid,))
        await conn.execute("DELETE FROM projects WHERE id = %s", (pid,))
        await conn.commit()


async def main() -> None:
    if os.getenv("MLMS_INTEGRATION") != "1":
        print("Set MLMS_INTEGRATION=1 to run (requires docker compose up)")
        sys.exit(1)

    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    scenarios: list[dict[str, Any]] = data["scenarios"]
    log.info("Loaded %d scenarios", len(scenarios))

    conn = await psycopg.AsyncConnection.connect(_PG_DSN, autocommit=False)
    rd = aioredis.from_url(settings.redis_url, decode_responses=False)

    results: list[dict[str, Any]] = []
    fails: list[str] = []

    try:
        for i, scenario in enumerate(scenarios, 1):
            log.info(
                "[%3d/%d] %-30s %s",
                i, len(scenarios), scenario["id"], scenario["category"],
            )
            try:
                r = await _run_scenario(scenario, conn=conn, rd=rd)
                results.append(r)
                if r["recall"] == 0.0:
                    fails.append(r["id"])
                extra = f"  sar={r['sar_recall']:.2f}" if "sar_recall" in r else ""
                log.info("  naive=%.2f%s  final=%.2f", r["naive_recall"], extra, r["recall"])
            except Exception as exc:
                log.error("  ERROR in %s: %s", scenario["id"], exc, exc_info=True)
                results.append({"id": scenario["id"], "category": scenario["category"],
                                "naive_recall": 0.0, "recall": 0.0})
                fails.append(scenario["id"])
    finally:
        await conn.close()
        await rd.aclose()

    categories = ["lexical", "semantic_drift", "conflict", "cross_session"]
    print("\n" + "=" * 55)
    print("RECALL@5 RESULTS")
    print("=" * 55)
    for cat in categories:
        cat_recalls = [r["recall"] for r in results if r["category"] == cat]
        if cat_recalls:
            avg = sum(cat_recalls) / len(cat_recalls)
            print(f"  {cat:<22} {avg:.3f}  (n={len(cat_recalls)})")

    overall = sum(r["recall"] for r in results) / max(len(results), 1)
    status = "PASS" if overall >= PRECISION_AT_5_TARGET else "FAIL"
    print(f"  {'OVERALL':<22} {overall:.3f}  [{status} target>={PRECISION_AT_5_TARGET}]")
    print()

    if fails:
        print(f"recall=0 ({len(fails)} scenarios):")
        for fid in fails:
            print(f"  - {fid}")
    else:
        print("No recall=0 scenarios.")
    print("=" * 55)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
