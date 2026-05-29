#!/usr/bin/env python
"""Full A/B benchmark on synthetic_hard.json (400-fact shared corpus, 93 scenarios).

Backends:
  BM25              rank_bm25, pure lexical
  nomic-embed-text  Ollama http://localhost:11434
  mxbai-embed-large Ollama http://localhost:11434
  Qwen3-8B          OpenRouter (EMBEDDING_API_KEY from .env)

Naive: embed(query) -> cosine top-5 of 400 facts
SAR:   embed(session_log_content + query) -> cosine top-5 of 400 facts
Recall@5 = |relevant in top-5| / |relevant|

Usage:
    MLMS_INTEGRATION=1 python scripts/run_hard_ab.py
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

from mlms.config import EMBEDDING_DIM, EMBEDDING_MODEL, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_FIXTURE = Path(__file__).parent.parent / "tests" / "fixtures" / "synthetic_hard.json"
_RESULTS = Path(__file__).parent / "hard_ab_results.json"
_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_K = 5

_OLLAMA_BASE = "http://localhost:11434/v1"


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


def _truncate_normalize(vec: list[float], dim: int) -> list[float]:
    return _normalize(vec[:dim])


async def _api_embed(
    text: str,
    model: str,
    *,
    api_base: str,
    api_key: str,
    cache: dict[tuple[str, str], list[float]],
    request_dim: int | None = None,
    max_chars: int | None = None,
) -> list[float]:
    if max_chars:
        text = text[:max_chars]
    key = (model, text)
    if key in cache:
        return cache[key]
    payload: dict[str, Any] = {
        "model": model,
        "input": text,
        "encoding_format": "float",
    }
    if request_dim is not None:
        payload["dimensions"] = request_dim
    delay = 1.0
    for attempt in range(4):
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                r = await client.post(
                    f"{api_base}/embeddings",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=payload,
                )
                r.raise_for_status()
                raw = r.json()["data"][0]["embedding"]
            break
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.RemoteProtocolError) as exc:
            if attempt == 3:
                raise
            log.warning("embed retry %d/3 (%s): %s", attempt + 1, model, exc)
            await asyncio.sleep(delay)
            delay *= 2
    if request_dim is not None:
        vec = _truncate_normalize(raw, request_dim)
    else:
        vec = _normalize(raw)
    cache[key] = vec
    return vec


def _bm25_top(fact_texts: list[str], query: str, k: int) -> set[int]:
    from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
    tokenized = [t.lower().split() for t in fact_texts]
    scores = list(BM25Okapi(tokenized).get_scores(query.lower().split()))
    return set(sorted(range(len(scores)), key=lambda i: -scores[i])[:k])


def _cosine_top(fact_vecs: list[list[float]], q_vec: list[float], k: int) -> set[int]:
    scores = [sum(a * b for a, b in zip(q_vec, fv)) for fv in fact_vecs]
    return set(sorted(range(len(scores)), key=lambda i: -scores[i])[:k])


@dataclass
class Backend:
    label: str
    model: str | None      # None = BM25
    api_base: str = ""
    api_key: str = ""
    request_dim: int | None = None  # None = no truncation, e.g. for Ollama
    max_chars: int | None = None    # truncate input text before embedding (mxbai 512-token limit)
    _cache: dict[tuple[str, str], list[float]] = field(
        default_factory=dict, init=False, repr=False
    )

    async def embed(self, text: str) -> list[float]:
        assert self.model is not None
        return await _api_embed(
            text,
            self.model,
            api_base=self.api_base,
            api_key=self.api_key,
            cache=self._cache,
            request_dim=self.request_dim,
            max_chars=self.max_chars,
        )

    async def check_alive(self) -> bool:
        if self.model is None:
            return True
        try:
            await self.embed("test")
            return True
        except Exception as exc:
            log.warning("[%s] unavailable: %s", self.label, exc)
            return False


def _sar_query(scenario: dict[str, Any]) -> str:
    parts = [e["content"] for e in scenario["setup"].get("session_log", [])]
    parts.append(scenario["query"])
    return " ".join(parts)


async def _run_backend(
    backend: Backend,
    corpus: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    fact_ids = [f["id"] for f in corpus]
    fact_texts = [f["content"] for f in corpus]
    n_facts = len(fact_ids)

    # Build index once
    log.info("[%s] embedding %d facts...", backend.label, n_facts)
    if backend.model is None:
        fact_vecs: list[list[float]] = []  # BM25 doesn't pre-compute
    else:
        fact_vecs = []
        for i, text in enumerate(fact_texts):
            if i % 50 == 0:
                log.info("[%s] corpus %d/%d", backend.label, i, n_facts)
            fact_vecs.append(await backend.embed(text))
        log.info("[%s] corpus indexed", backend.label)

    rows: list[dict[str, Any]] = []
    for i, s in enumerate(scenarios, 1):
        if i % 20 == 0:
            log.info("[%s] scenario %d/%d", backend.label, i, len(scenarios))
        query = s["query"]
        enriched = _sar_query(s)
        relevant = set(s["expected"]["relevant_ids"])

        try:
            if backend.model is None:
                naive_top = {fact_ids[j] for j in _bm25_top(fact_texts, query, _K)}
                sar_top = {fact_ids[j] for j in _bm25_top(fact_texts, enriched, _K)}
            else:
                q_naive = await backend.embed(query)
                q_sar = await backend.embed(enriched)
                naive_top = {fact_ids[j] for j in _cosine_top(fact_vecs, q_naive, _K)}
                sar_top = {fact_ids[j] for j in _cosine_top(fact_vecs, q_sar, _K)}

            denom = max(len(relevant), 1)
            naive_r = len(naive_top & relevant) / denom
            sar_r = len(sar_top & relevant) / denom
        except Exception as exc:
            log.error("[%s] %s failed: %s", backend.label, s["id"], exc)
            naive_r = sar_r = 0.0

        rows.append({
            "id": s["id"],
            "type": s["type"],
            "naive_recall": naive_r,
            "sar_recall": sar_r,
        })

    return {"label": backend.label, "n": len(rows), "rows": rows}


def _aggregate(result: dict[str, Any]) -> dict[str, Any]:
    rows = result["rows"]

    def _avg(subset: list[dict[str, Any]]) -> tuple[float, float, float]:
        if not subset:
            return 0.0, 0.0, 0.0
        naive = sum(r["naive_recall"] for r in subset) / len(subset)
        sar = sum(r["sar_recall"] for r in subset) / len(subset)
        return round(naive, 4), round(sar, 4), round(sar - naive, 4)

    types = ["lexical_drift", "semantic_drift", "cross_domain", "multi_hop"]
    by_type = {
        t: _avg([r for r in rows if r["type"] == t]) for t in types
    }
    overall = _avg(rows)
    return {
        "label": result["label"],
        "n": result["n"],
        "overall": {"naive": overall[0], "sar": overall[1], "delta": overall[2]},
        "by_type": {
            t: {"naive": v[0], "sar": v[1], "delta": v[2], "n": sum(1 for r in rows if r["type"] == t)}
            for t, v in by_type.items()
        },
    }


def _print_results(agg: list[dict[str, Any]]) -> None:
    print()
    print("=" * 70)
    print("HARD A/B BENCHMARK --- Naive vs SAR (synthetic_hard.json, k=5)")
    print("=" * 70)

    print()
    print("OVERALL (all 93 scenarios, 400-fact corpus)")
    print(f"  {'Model':<25} {'N':>4}  {'Naive':>7}  {'SAR':>7}  {'Delta':>8}")
    print(f"  {'-'*25} {'-'*4}  {'-'*7}  {'-'*7}  {'-'*8}")
    for r in agg:
        o = r["overall"]
        print(f"  {r['label']:<25} {r['n']:>4}  {o['naive']:>7.3f}  {o['sar']:>7.3f}  {o['delta']:>+8.3f}")

    types = ["lexical_drift", "semantic_drift", "cross_domain", "multi_hop"]
    type_labels = {"lexical_drift": "lexical_drift", "semantic_drift": "sem_drift",
                   "cross_domain": "cross_domain", "multi_hop": "multi_hop"}

    print()
    print("PER-TYPE BREAKDOWN (Naive / SAR / Delta)")
    hdr = f"  {'Model':<25}"
    for t in types:
        hdr += f"  {type_labels[t]:^21}"
    print(hdr)
    hdr2 = f"  {'':<25}"
    for _ in types:
        hdr2 += f"  {'N':>4}{'Naive':>6}{'SAR':>6}{'D':>6}"
    print(hdr2)
    print(f"  {'-'*25}" + "  " + ("  " + "-"*21) * len(types))

    for r in agg:
        row = f"  {r['label']:<25}"
        for t in types:
            bt = r["by_type"][t]
            row += f"  {bt['n']:>4}{bt['naive']:>6.3f}{bt['sar']:>6.3f}{bt['delta']:>+6.3f}"
        print(row)

    print("=" * 70)


async def main() -> None:
    if os.getenv("MLMS_INTEGRATION") != "1":
        print("Set MLMS_INTEGRATION=1 to run")
        sys.exit(1)

    data = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    corpus: list[dict[str, Any]] = data["corpus"]
    scenarios: list[dict[str, Any]] = data["scenarios"]
    log.info("Loaded %d facts, %d scenarios", len(corpus), len(scenarios))

    candidates: list[Backend] = [
        Backend("BM25", None),
        Backend(
            "nomic-embed-text",
            "nomic-embed-text",
            api_base=_OLLAMA_BASE,
            api_key="ollama",
            request_dim=None,
        ),
        Backend(
            "mxbai-embed-large",
            "mxbai-embed-large",
            api_base=_OLLAMA_BASE,
            api_key="ollama",
            request_dim=None,
            max_chars=800,  # 512-token limit; ~800 chars of Russian text ≈ 400-500 tokens
        ),
        Backend(
            "Qwen3-8B",
            EMBEDDING_MODEL,
            api_base=settings.embedding_api_base,
            api_key=settings.embedding_api_key,
            request_dim=EMBEDDING_DIM,
        ),
    ]

    # Probe Ollama backends
    backends: list[Backend] = [candidates[0]]  # BM25 always runs
    for b in candidates[1:]:
        if b.model == EMBEDDING_MODEL or await b.check_alive():
            backends.append(b)
        else:
            log.warning("[%s] skipped (unavailable)", b.label)

    all_results: list[dict[str, Any]] = []
    for backend in backends:
        log.info("=== %s ===", backend.label)
        raw = await _run_backend(backend, corpus, scenarios)
        agg = _aggregate(raw)
        all_results.append(agg)
        o = agg["overall"]
        log.info("[%s] naive=%.3f  sar=%.3f  delta=%+.3f", backend.label, o["naive"], o["sar"], o["delta"])

    _RESULTS.write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Results -> %s", _RESULTS)

    _print_results(all_results)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
