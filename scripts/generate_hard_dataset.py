#!/usr/bin/env python
"""Generate tests/fixtures/synthetic_hard.json — harder retrieval benchmark.

Shared corpus: 400 facts (10 domains × 40)
100 scenarios: lexical_drift:30 | semantic_drift:30 | cross_domain:25 | multi_hop:15
3-step verification pipeline: generator + verifier1 (GPT) + verifier2 (Grok)
Resumable from checkpoints in tests/fixtures/checkpoint_*.json

Usage:
    python scripts/generate_hard_dataset.py

Requires in .env: OPENROUTER_API_KEY, MODEL_CLAUDE, MODEL_GPT4O, MODEL_GROK
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

# ── paths ──────────────────────────────────────────────────────────────────
REPO = Path(__file__).parent.parent
FIXTURES = REPO / "tests" / "fixtures"
FIXTURES.mkdir(parents=True, exist_ok=True)

OUT_FILE = FIXTURES / "synthetic_hard.json"
CKPT_CORPUS = FIXTURES / "checkpoint_corpus.json"
CKPT_SCENARIOS = FIXTURES / "checkpoint_scenarios.json"
COST_LOG_FILE = FIXTURES / "cost_log.json"

# ── config ─────────────────────────────────────────────────────────────────
DOMAINS = [
    "kubernetes", "postgresql", "redis", "kafka", "fastapi",
    "react", "docker", "elasticsearch", "nginx", "pytorch",
]
FACTS_PER_DOMAIN = 40

SCENARIO_PLAN: list[tuple[str, int]] = [
    ("lexical_drift", 30),
    ("semantic_drift", 30),
    ("cross_domain", 25),
    ("multi_hop", 15),
]
LEXICAL_GAP_MAX = {
    "lexical_drift": 2,
    "semantic_drift": 2,
    "cross_domain": 2,
    "multi_hop": 1,
}

# Russian + English stopwords for lexical gap scoring
_STOPS = {
    "а", "в", "и", "к", "на", "не", "с", "о", "от", "по", "за", "из",
    "то", "что", "как", "это", "для", "при", "до", "же", "но", "он", "она",
    "оно", "они", "их", "им", "его", "её", "ли", "уже", "ещё", "но", "так",
    "is", "the", "a", "an", "in", "of", "to", "and", "or", "for", "with",
    "this", "that", "are", "was", "were", "be", "been", "it", "its", "by",
    "at", "from", "on", "as", "not", "has", "have", "had", "will", "can",
    "do", "does", "did", "if", "then", "when", "how", "what", "which",
}

API_BASE = "https://openrouter.ai/api/v1"

# ── env loading ─────────────────────────────────────────────────────────────
def _load_dotenv() -> None:
    env_path = REPO / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()

try:
    API_KEY = os.environ["OPENROUTER_API_KEY"]
    GENERATOR = os.environ["MODEL_CLAUDE"]
    VERIFIER_1 = os.environ["MODEL_GPT4O"]
    VERIFIER_2 = os.environ["MODEL_GROK"]
except KeyError as e:
    sys.exit(f"Missing env var: {e}. Set it in .env or environment.")

# ── cost tracking ───────────────────────────────────────────────────────────
_cost_log: list[dict] = []


def _track(model: str, usage: dict, purpose: str) -> None:
    _cost_log.append({
        "model": model,
        "purpose": purpose,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "cost_usd": usage.get("cost", 0.0),
    })


def _total_cost() -> float:
    return sum(float(e.get("cost_usd", 0)) for e in _cost_log)


# ── API call with exponential backoff ───────────────────────────────────────
async def _chat(
    client: httpx.AsyncClient,
    model: str,
    messages: list[dict],
    purpose: str,
    temperature: float = 0.4,
) -> str:
    delay = 1.0
    for attempt in range(4):
        try:
            resp = await client.post(
                f"{API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": model, "messages": messages, "temperature": temperature},
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            if "usage" in data:
                _track(model, data["usage"], purpose)
            return str(data["choices"][0]["message"]["content"])
        except Exception as exc:
            if attempt == 3:
                raise
            print(f"    [retry {attempt + 1}/3] {model} — {exc!r} — sleep {delay:.0f}s")
            await asyncio.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


# ── JSON extraction ─────────────────────────────────────────────────────────
def _extract_json(text: str) -> dict | list:
    for pat in (
        r"```json\s*([\s\S]+?)\s*```",
        r"```\s*([\{|\[][\s\S]+?[\}|\]])\s*```",
    ):
        m = re.search(pat, text)
        if m:
            return json.loads(m.group(1))
    # Try raw JSON block (first { or [)
    for start in (text.find("{"), text.find("[")):
        if start == -1:
            continue
        # Find matching close
        for end in range(len(text), start, -1):
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON in response (first 300 chars): {text[:300]}")


# ── lexical gap score ───────────────────────────────────────────────────────
def _words(text: str) -> set[str]:
    return {w for w in re.split(r"[\W_]+", text.lower()) if len(w) > 2 and w not in _STOPS}


def lexical_gap_score(query: str, corpus: list[dict], relevant_ids: list[str]) -> int:
    relevant_text = " ".join(f["content"] for f in corpus if f["id"] in relevant_ids)
    return len(_words(query) & _words(relevant_text))


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — Corpus generation
# ═══════════════════════════════════════════════════════════════════════════

_CORPUS_PROMPT = """\
Сгенерируй ровно {n} технических фактов о {domain} для бенчмарка поиска по памяти.

Требования:
- Формальный технический язык (стиль системной документации или спецификации)
- Каждый факт: конкретная проблема, значение параметра, команда или ошибка — без абстрактных советов
- 1-2 предложения максимум
- Факты разнообразны: разные аспекты, без дублирования
- Технические термины, команды, параметры — на английском; описание — на русском

Верни ТОЛЬКО JSON-массив без пояснений:
[{{"id": "f{start_id:03d}", "domain": "{domain}", "content": "...", "tags": ["tag1", "tag2"]}}, ...]

ID от f{start_id:03d} до f{end_id:03d} включительно (нули слева до 3 знаков).
"""


async def generate_corpus(client: httpx.AsyncClient) -> list[dict]:
    if CKPT_CORPUS.exists():
        existing: list[dict] = json.loads(CKPT_CORPUS.read_text(encoding="utf-8"))
        done_domains = {f["domain"] for f in existing}
        print(f"  Resuming corpus: {len(existing)} facts, done={sorted(done_domains)}")
    else:
        existing = []
        done_domains: set[str] = set()

    for i, domain in enumerate(DOMAINS):
        if domain in done_domains:
            print(f"  {domain}: skip (done)")
            continue
        start_id = i * FACTS_PER_DOMAIN + 1
        end_id = start_id + FACTS_PER_DOMAIN - 1
        prompt = _CORPUS_PROMPT.format(
            n=FACTS_PER_DOMAIN, domain=domain,
            start_id=start_id, end_id=end_id,
        )
        print(f"  Generating {FACTS_PER_DOMAIN} facts: {domain} (f{start_id:03d}–f{end_id:03d})...")
        raw = await _chat(client, GENERATOR, [{"role": "user", "content": prompt}], f"corpus_{domain}")
        facts: list[dict] = _extract_json(raw)  # type: ignore[assignment]
        if len(facts) != FACTS_PER_DOMAIN:
            print(f"    WARNING: expected {FACTS_PER_DOMAIN}, got {len(facts)} — padding/truncating")
            facts = facts[:FACTS_PER_DOMAIN]
        for j, fact in enumerate(facts):
            fact["id"] = f"f{start_id + j:03d}"
            fact.setdefault("domain", domain)
            fact.setdefault("tags", [])
        existing.extend(facts)
        CKPT_CORPUS.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    OK — checkpoint ({len(existing)} total)")

    print(f"  Corpus done: {len(existing)} facts")
    return existing


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — Scenario generation
# ═══════════════════════════════════════════════════════════════════════════

_TYPE_GUIDANCE = {
    "lexical_drift": (
        "Пользователь описывает техническую проблему разговорным/пользовательским языком. "
        "Пример: пользователь говорит «сервис тормозит» о connection pool exhaustion, "
        "а факт содержит 'max_connections exceeded, TCP backlog overflow'. "
        "НОЛЬ технических терминов из фактов в query."
    ),
    "semantic_drift": (
        "Пользователь описывает симптом/эффект; факт описывает причину/механизм. "
        "Пример: пользователь спрашивает 'почему данные иногда пропадают' о WAL compaction, "
        "факт содержит 'log.retention.bytes triggers segment deletion'. "
        "Семантически связаны, но словарь полностью разный."
    ),
    "cross_domain": (
        "Проблема пользователя затрагивает два домена. Релевантные факты из РАЗНЫХ доменов. "
        "Query не называет технические термины ни одного домена — только симптом на пользовательском языке."
    ),
    "multi_hop": (
        "Ответ требует цепочки из 2+ фактов. Ни один факт не отвечает на вопрос сам по себе. "
        "Session_log устанавливает контекст цепочки. "
        "ЖЁСТКИЙ lexical_gap_score ≤ 1 — практически ноль общих слов."
    ),
}

_SCENARIO_PROMPT = """\
Ты генерируешь сценарий бенчмарка поиска по памяти.

ТИП СЦЕНАРИЯ: {stype}
ID: {sid}
Специфика типа: {guidance}

ДОСТУПНЫЙ КОРПУС (именно эти факты «сохранены в памяти системы»):
{corpus_text}

=== ШАГ A — Внутреннее обоснование (не попадёт в JSON) ===
Напиши:
Механизм дрейфа: [точный механизм vocab/semantic gap]
BM25 провалится потому что: [конкретно — ноль или почти ноль общих значимых слов]
Qwen3-8B naive провалится потому что: [семантическая причина без контекста сессии]
SAR выиграет потому что: [какие bridge terms из session_log решают задачу]

Если не можешь заполнить все три убедительно — напиши CANNOT_GENERATE и остановись.

=== ШАГ B — JSON сценария ===
После обоснования верни JSON:

```json
{{
  "relevant_ids": ["fXXX", "fYYY", "fZZZ"],
  "session_log": [
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ],
  "query": "...",
  "drift_mechanism": "...",
  "sar_bridge_terms": ["term1", "term2"]
}}
```

Правила:
- relevant_ids: ОБЯЗАТЕЛЬНО {min_relevant}-5 ID из корпуса выше (реальные ID!). Никогда не меньше {min_relevant}.
  Каждый релевантный факт должен содержать часть ответа — выбирай группу взаимосвязанных фактов, не один.
- session_log: 2-4 записи, содержащие BRIDGE TERMS — точные технические термины из текста фактов.
  ВАЖНО: session_log должен содержать ИМЕННО те технические слова, которые есть в relevant_ids фактах,
  иначе SAR не сможет найти их.
- query: ТОЛЬКО язык из session_log и пользовательские/разговорные термины — НОЛЬ технических терминов из текста фактов
- lexical_gap_score (вычислю сам) должен быть ≤ {gap_max}
- sar_bridge_terms: ключевые термины из session_log, делающие поиск возможным
"""

_VERIFIER_1_PROMPT = """\
Задача: оценить сложность поискового сценария.

ФАКТЫ В СИСТЕМЕ (только релевантные):
{facts_text}

ЗАПРОС: «{query}»

Вопрос: можно ли найти эти факты ({relevant_ids}) в топ-5 результатов по данному запросу БЕЗ контекста сессии?
Учитывай и BM25 (лексический поиск), и семантическое сходство векторов.

Ответь строго в формате:
VERDICT: yes
REASONING: [одно предложение]

или:

VERDICT: no
REASONING: [одно предложение]
"""

_VERIFIER_2_PROMPT = """\
Задача: оценить восстанавливаемость сценария через Session-Augmented RAG.

ФАКТЫ В СИСТЕМЕ (только релевантные):
{facts_text}

КОНТЕКСТ СЕССИИ:
{session_text}

ОБОГАЩЁННЫЙ ЗАПРОС: «{enriched}»

Вопрос: после добавления терминов из сессии к запросу, можно ли найти факты ({relevant_ids}) в топ-5?

Ответь строго в формате:
VERDICT: yes
REASONING: [одно предложение]

или:

VERDICT: no
REASONING: [одно предложение]
"""


def _parse_verdict(text: str) -> tuple[bool, str]:
    is_yes = bool(re.search(r"verdict\s*:\s*yes", text, re.IGNORECASE))
    m = re.search(r"reasoning\s*:\s*(.+)", text, re.IGNORECASE)
    reasoning = m.group(1).strip() if m else text[:200]
    return is_yes, reasoning


def _pick_domains(stype: str, used_recent: list[str]) -> list[str]:
    recent3 = used_recent[-3:]
    overused = {d for d in recent3 if recent3.count(d) >= 3}
    available = [d for d in DOMAINS if d not in overused] or DOMAINS
    if stype == "cross_domain":
        d1 = random.choice(available)
        remaining = [d for d in available if d != d1] or [d for d in DOMAINS if d != d1]
        d2 = random.choice(remaining)
        return [d1, d2]
    return [random.choice(available)]


def _make_excerpt(corpus: list[dict], domains: list[str], n: int = 12) -> list[dict]:
    per_domain = n // len(domains)
    result: list[dict] = []
    for d in domains:
        pool = [f for f in corpus if f["domain"] == d]
        result.extend(random.sample(pool, min(per_domain, len(pool))))
    return result


async def _gen_one(
    client: httpx.AsyncClient,
    corpus: list[dict],
    stype: str,
    sid: str,
    used_domains: list[str],
    last_mechanism: str,
) -> dict | None:
    domains = _pick_domains(stype, used_domains)
    excerpt = _make_excerpt(corpus, domains)
    corpus_text = "\n".join(f"[{f['id']}] ({f['domain']}) {f['content']}" for f in excerpt)
    gap_max = LEXICAL_GAP_MAX[stype]
    min_relevant = 2 if stype == "multi_hop" else 3

    prompt = _SCENARIO_PROMPT.format(
        stype=stype,
        sid=sid,
        guidance=_TYPE_GUIDANCE[stype],
        corpus_text=corpus_text,
        gap_max=gap_max,
        min_relevant=min_relevant,
    )

    for attempt in range(3):
        attempt_label = f"{sid} attempt {attempt + 1}/3"
        raw = await _chat(client, GENERATOR, [{"role": "user", "content": prompt}], f"scenario_{sid}")

        if "CANNOT_GENERATE" in raw:
            print(f"    [{attempt_label}] CANNOT_GENERATE — retry")
            continue

        try:
            data: dict = _extract_json(raw)  # type: ignore[assignment]
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"    [{attempt_label}] JSON parse error: {exc} — retry")
            continue

        relevant_ids: list[str] = data.get("relevant_ids", [])
        if not (min_relevant <= len(relevant_ids) <= 5):
            print(f"    [{attempt_label}] relevant_ids count={len(relevant_ids)}, need {min_relevant}-5 — retry")
            continue

        excerpt_ids = {f["id"] for f in excerpt}
        if not all(rid in excerpt_ids for rid in relevant_ids):
            print(f"    [{attempt_label}] unknown fact IDs — retry")
            continue

        query: str = data.get("query", "")
        session_log: list[dict] = data.get("session_log", [])
        drift_mechanism: str = data.get("drift_mechanism", "unknown")
        bridge_terms: list[str] = data.get("sar_bridge_terms", [])

        if drift_mechanism == last_mechanism:
            print(f"    [{attempt_label}] repeated mechanism '{drift_mechanism}' — retry")
            continue

        gap = lexical_gap_score(query, excerpt, relevant_ids)
        if gap > gap_max:
            print(f"    [{attempt_label}] lexical_gap_score={gap} > {gap_max} — retry")
            continue

        # Verifier 1: too_easy?
        rel_text = "\n".join(f"[{f['id']}] {f['content']}" for f in excerpt if f["id"] in relevant_ids)
        v1_raw = await _chat(
            client, VERIFIER_1,
            [{"role": "user", "content": _VERIFIER_1_PROMPT.format(
                facts_text=rel_text,
                query=query,
                relevant_ids=", ".join(relevant_ids),
            )}],
            f"v1_{sid}",
        )
        too_easy, v1_reason = _parse_verdict(v1_raw)
        if too_easy:
            print(f"    [{attempt_label}] Verifier1: too_easy — retry")
            continue

        # Verifier 2: sar_solvable?
        session_text = "\n".join(f"[{e.get('role','?')}] {e.get('content','')}" for e in session_log)
        enriched = (" ".join(e.get("content", "") for e in session_log) + " " + query).strip()
        v2_raw = await _chat(
            client, VERIFIER_2,
            [{"role": "user", "content": _VERIFIER_2_PROMPT.format(
                facts_text=rel_text,
                session_text=session_text,
                enriched=enriched,
                relevant_ids=", ".join(relevant_ids),
            )}],
            f"v2_{sid}",
        )
        sar_solvable, v2_reason = _parse_verdict(v2_raw)
        if not sar_solvable:
            print(f"    [{attempt_label}] Verifier2: NOT sar_solvable — retry")
            continue

        irrelevant_ids = [f["id"] for f in excerpt if f["id"] not in relevant_ids]
        domain_label = "/".join(domains)

        return {
            "id": sid,
            "type": stype,
            "domain": domain_label,
            "setup": {"session_log": session_log},
            "query": query,
            "expected": {
                "relevant_ids": relevant_ids,
                "irrelevant_ids": irrelevant_ids,
            },
            "meta": {
                "lexical_gap_score": gap,
                "difficulty": "hard",
                "drift_mechanism": drift_mechanism,
                "sar_bridge_terms": bridge_terms,
                "too_easy": False,
                "sar_solvable": True,
                "verifier_1_reasoning": v1_reason,
                "verifier_2_reasoning": v2_reason,
            },
        }

    print(f"    [SKIP] {sid} — exhausted 3 attempts")
    return None


async def generate_scenarios(client: httpx.AsyncClient, corpus: list[dict]) -> list[dict]:
    if CKPT_SCENARIOS.exists():
        existing: list[dict] = json.loads(CKPT_SCENARIOS.read_text(encoding="utf-8"))
        print(f"  Resuming scenarios: {len(existing)} done")
    else:
        existing = []

    done_ids = {s["id"] for s in existing}
    used_domains: list[str] = [s["domain"].split("/")[0] for s in existing[-10:]]
    last_mechanism = existing[-1]["meta"]["drift_mechanism"] if existing else ""

    # Build target list
    targets: list[tuple[str, str]] = []
    for stype, count in SCENARIO_PLAN:
        for i in range(1, count + 1):
            prefix = {"lexical_drift": "ld", "semantic_drift": "sd", "cross_domain": "cd", "multi_hop": "mh"}[stype]
            targets.append((f"{prefix}_{i:03d}", stype))

    for sid, stype in targets:
        if sid in done_ids:
            print(f"  {sid}: skip (done)")
            continue

        print(f"  Generating {sid} ({stype})...")
        scenario = await _gen_one(client, corpus, stype, sid, used_domains, last_mechanism)
        if scenario is None:
            continue

        existing.append(scenario)
        used_domains.append(scenario["domain"].split("/")[0])
        last_mechanism = scenario["meta"]["drift_mechanism"]

        if len(existing) % 10 == 0:
            CKPT_SCENARIOS.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"    checkpoint: {len(existing)} scenarios saved")

    CKPT_SCENARIOS.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Scenarios done: {len(existing)}")
    return existing


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3 — BM25 validation
# ═══════════════════════════════════════════════════════════════════════════

def bm25_validate(corpus: list[dict], scenarios: list[dict]) -> float:
    from rank_bm25 import BM25Okapi  # type: ignore[import]

    fact_ids = [f["id"] for f in corpus]
    bm25 = BM25Okapi([f["content"].lower().split() for f in corpus])

    hits = 0
    for s in scenarios:
        relevant = set(s["expected"]["relevant_ids"])
        scores = bm25.get_scores(s["query"].lower().split())
        top5 = {fact_ids[i] for i in sorted(range(len(scores)), key=lambda i: -scores[i])[:5]}
        if top5 & relevant:
            hits += 1

    return hits / len(scenarios) if scenarios else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

async def main() -> None:
    print(f"Using: GENERATOR={GENERATOR} VERIFIER_1={VERIFIER_1} VERIFIER_2={VERIFIER_2}")
    print(f"Output: {OUT_FILE}")
    print()

    async with httpx.AsyncClient(timeout=120.0) as client:
        print("=== STAGE 1 --- Corpus (400 facts) ===")
        corpus = await generate_corpus(client)
        print()

        print("=== STAGE 2 --- Scenarios (100 scenarios) ===")
        scenarios = await generate_scenarios(client, corpus)
        print()

    print("=== STAGE 3 --- BM25 Validation ===")
    recall = bm25_validate(corpus, scenarios)
    print(f"BM25 naive Recall@5 (full 400-fact corpus) = {recall:.3f}")
    if recall > 0.55:
        print(f"WARNING: dataset too easy — BM25 Recall@5={recall:.3f} exceeds 0.55 ceiling")
        print("         Consider tightening lexical_gap_score thresholds or regenerating easy scenarios.")
    elif recall < 0.20:
        print(f"WARNING: dataset may be too hard — BM25 Recall@5={recall:.3f} below 0.20 floor")
    else:
        print(f"OK: BM25 Recall@5={recall:.3f} in target range [0.20–0.55]")
    print()

    now = datetime.now(timezone.utc).isoformat()
    result = {
        "meta": {
            "generated_at": now,
            "generator_model": GENERATOR,
            "verifier_1_model": VERIFIER_1,
            "verifier_2_model": VERIFIER_2,
            "total_facts": len(corpus),
            "total_scenarios": len(scenarios),
            "bm25_naive_recall5": round(recall, 4),
            "total_cost_usd": round(_total_cost(), 4),
        },
        "corpus": corpus,
        "scenarios": scenarios,
    }
    OUT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    COST_LOG_FILE.write_text(json.dumps(_cost_log, ensure_ascii=False, indent=2), encoding="utf-8")

    total_tokens = sum(int(e.get("total_tokens", 0)) for e in _cost_log)
    print(f"Written: {OUT_FILE}")
    print()
    print("=== COST SUMMARY ===")
    print(f"Total API calls : {len(_cost_log)}")
    print(f"Total tokens    : {total_tokens:,}")
    print(f"Total cost (est): ${_total_cost():.4f}")


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
