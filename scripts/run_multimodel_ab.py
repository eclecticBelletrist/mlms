# DEPRECATED: delta=0 на synthetic_100 (тривиальный результат, 5 фактов при k=5). Используй run_hard_ab.py.
# DEPRECATED: тривиальные результаты на synthetic_100 — 5 фактов/сценарий, BM25=1.0, Qwen3-0.6B N/A (OpenRouter 404)
# Используй: scripts/run_hard_ab.py (synthetic_hard.json, 400 фактов, 4 backend-а включая Ollama)

#!/usr/bin/env python
"""Multi-model A/B: SAR delta across Qwen3-8B, Qwen3-0.6B, text-embedding-3-small, BM25.

Retrieval is in-memory (no pgvector). Each model gets its own vector space.
BM25 uses rank_bm25; OpenAI model requires OPENAI_API_KEY.

Usage:
    MLMS_INTEGRATION=1 [OPENAI_API_KEY=sk-...] python scripts/run_multimodel_ab.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlms.config import EMBEDDING_DIM, EMBEDDING_FALLBACK, EMBEDDING_MODEL, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_FIXTURE = Path(__file__).parent.parent / "tests" / "fixtures" / "synthetic_100.json"
_TIMEOUT = httpx.Timeout(30.0, connect=5.0)


def _truncate_normalize(vec: list[float]) -> list[float]:
    v = vec[:EMBEDDING_DIM]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else v


async def _api_embed(
    text: str,
    model: str,
    *,
    api_base: str,
    api_key: str,
    cache: dict[tuple[str, str], list[float]],
) -> list[float]:
    key = (model, text)
    if key in cache:
        return cache[key]
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        r = await client.post(
            f"{api_base}/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "input": text,
                "dimensions": EMBEDDING_DIM,
                "encoding_format": "float",
            },
        )
        r.raise_for_status()
        vec = _truncate_normalize(r.json()["data"][0]["embedding"])
    cache[key] = vec
    return vec


def _bm25_scores(corpus: list[str], query: str) -> list[float]:
    from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
    tokenized = [doc.lower().split() for doc in corpus]
    return list(BM25Okapi(tokenized).get_scores(query.lower().split()))


def _top_k_idx(scores: list[float], k: int) -> set[int]:
    return set(sorted(range(len(scores)), key=lambda i: -scores[i])[:k])


@dataclass
class Backend:
    label: str
    model: str | None  # None = BM25
    api_base: str = ""
    api_key: str = ""
    _cache: dict[tuple[str, str], list[float]] = field(default_factory=dict, init=False, repr=False)

    async def embed(self, text: str) -> list[float]:
        assert self.model is not None
        return await _api_embed(text, self.model, api_base=self.api_base, api_key=self.api_key, cache=self._cache)


async def _ab_scenario(scenario: dict[str, Any], backend: Backend) -> dict[str, float]:
    facts: list[dict[str, Any]] = scenario["setup"].get("facts", [])
    session_labels: list[str] = scenario["setup"].get("session_log", [])
    query: str = scenario["query"]
    relevant: set[str] = set(scenario["expected"]["relevant_ids"])
    fact_texts = [f["content"] for f in facts]
    fact_ids = [f["id"] for f in facts]
    k = min(5, len(fact_ids))
    if not fact_ids:
        return {"naive_recall": 0.0, "sar_recall": 0.0}
    enriched = (" ".join(session_labels) + " " + query).strip()

    if backend.model is None:
        naive_top = {fact_ids[i] for i in _top_k_idx(_bm25_scores(fact_texts, query), k)}
        sar_top = {fact_ids[i] for i in _top_k_idx(_bm25_scores(fact_texts, enriched), k)}
    else:
        fact_vecs = [await backend.embed(t) for t in fact_texts]
        q_naive = await backend.embed(query)
        q_sar = await backend.embed(enriched)
        # vectors are L2-normalised → dot product == cosine similarity
        naive_scores = [sum(a * b for a, b in zip(q_naive, fv)) for fv in fact_vecs]
        sar_scores = [sum(a * b for a, b in zip(q_sar, fv)) for fv in fact_vecs]
        naive_top = {fact_ids[i] for i in _top_k_idx(naive_scores, k)}
        sar_top = {fact_ids[i] for i in _top_k_idx(sar_scores, k)}

    denom = max(len(relevant), 1)
    return {
        "naive_recall": len(naive_top & relevant) / denom,
        "sar_recall": len(sar_top & relevant) / denom,
    }


async def _run_backend(scenarios: list[dict[str, Any]], backend: Backend) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for i, s in enumerate(scenarios, 1):
        log.info("[%s] [%2d/%d] %s", backend.label, i, len(scenarios), s["id"])
        try:
            r = await _ab_scenario(s, backend)
        except Exception as exc:
            log.error("ERROR %s: %s", s["id"], exc, exc_info=True)
            r = {"naive_recall": 0.0, "sar_recall": 0.0}
        rows.append({"id": s["id"], **r})
    n = len(rows)
    naive_avg = sum(r["naive_recall"] for r in rows) / n
    sar_avg = sum(r["sar_recall"] for r in rows) / n
    return {
        "label": backend.label,
        "n": n,
        "naive_avg": round(naive_avg, 4),
        "sar_avg": round(sar_avg, 4),
        "delta": round(sar_avg - naive_avg, 4),
        "scenarios": rows,
    }


async def main() -> None:
    if os.getenv("MLMS_INTEGRATION") != "1":
        print("Set MLMS_INTEGRATION=1 to run")
        sys.exit(1)

    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    scenarios = [
        s for s in data["scenarios"]
        if s["category"] == "semantic_drift" and s["meta"].get("ab_test")
    ]
    log.info("Loaded %d semantic_drift scenarios", len(scenarios))

    backends: list[Backend] = [
        Backend("Qwen3-8B", EMBEDDING_MODEL, settings.embedding_api_base, settings.embedding_api_key),
        Backend("Qwen3-0.6B", EMBEDDING_FALLBACK, settings.embedding_api_base, settings.embedding_api_key),
    ]
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        backends.append(
            Backend("text-embedding-3-small", "text-embedding-3-small", "https://api.openai.com/v1", openai_key)
        )
    else:
        log.info("OPENAI_API_KEY not set -- skipping text-embedding-3-small")
    backends.append(Backend("BM25", None))

    all_results: list[dict[str, Any]] = []
    for backend in backends:
        log.info("=== %s ===", backend.label)
        res = await _run_backend(scenarios, backend)
        all_results.append(res)
        log.info("  naive=%.3f  sar=%.3f  delta=%+.3f", res["naive_avg"], res["sar_avg"], res["delta"])

    print()
    print("=" * 65)
    print(f"MULTI-MODEL SAR A/B -- semantic_drift (n={len(scenarios)})")
    print("=" * 65)
    print(f"  {'Model':<30} {'Naive':>7} {'SAR':>7} {'Delta':>8}")
    print(f"  {'-'*30} {'-'*7} {'-'*7} {'-'*8}")
    for r in all_results:
        print(f"  {r['label']:<30} {r['naive_avg']:>7.3f} {r['sar_avg']:>7.3f} {r['delta']:>+8.3f}")
    print("=" * 65)

    out = Path(__file__).parent / "multimodel_results.json"
    out.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Results -> %s", out)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
