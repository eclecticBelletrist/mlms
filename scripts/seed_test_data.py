"""
Generate synthetic_100.json benchmark dataset for MLMS Recall@5 evaluation.

Pipeline per batch (5 scenarios):
  Stage 1 — GENERATE  : LLM via OpenRouter structured output (tool_use)
  Stage 2 — VERIFY    : cross-family LLM checks ground truth
  Stage 3 — VALIDATE  : structural checks + auto-fix
  Stage 4 — ACCUMULATE: append to partial checkpoint

Usage:
    python scripts/seed_test_data.py [--max-batches N] [--verbose]

Environment (.env):
    OPENROUTER_API_KEY  — OpenRouter API key (falls back to EMBEDDING_API_KEY)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# OpenRouter model IDs — override via .env (MODEL_CLAUDE / MODEL_GPT4O / MODEL_GROK)
MODELS: dict[str, str] = {
    "claude": os.getenv("MODEL_CLAUDE", "anthropic/claude-sonnet-4.6"),
    "gpt4o":  os.getenv("MODEL_GPT4O",  "openai/gpt-5.2"),
    "grok":   os.getenv("MODEL_GROK",   "x-ai/grok-4.20"),
}

# Short names for meta.generated_by / meta.verified_by fields
# Derived from the last segment of the OpenRouter path (after the slash)
MODEL_SHORT: dict[str, str] = {k: v.split("/")[-1] for k, v in MODELS.items()}

# (generator_key, verifier_key) per batch_index % 3
# Cross-family rotation: Anthropic → OpenAI → xAI → Anthropic
MODEL_ROTATION = [
    ("claude", "gpt4o"),
    ("gpt4o",  "grok"),
    ("grok",   "claude"),
]

# 5 batches × 4 categories = 20 batches total
CATEGORIES = ["lexical", "semantic_drift", "conflict", "cross_session"]
BATCH_TO_CATEGORY: list[str] = [cat for cat in CATEGORIES for _ in range(5)]

TOTAL_BATCHES = 20
SCENARIOS_PER_BATCH = 5

PARTIAL_PATH = Path("tests/fixtures/synthetic_partial.json")
FINAL_PATH = Path("tests/fixtures/synthetic_100.json")

# meta fields that are category-specific, patched after generation
_CATEGORY_META: dict[str, dict[str, bool]] = {
    "lexical":        {"tests_session_augmentation": False, "ab_test": False},
    "semantic_drift": {"tests_session_augmentation": True,  "ab_test": True},
    "conflict":       {"tests_session_augmentation": False, "ab_test": False},
    "cross_session":  {"tests_session_augmentation": False, "ab_test": False},
}

# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_GENERATE = """\
Ты генерируешь тест-кейсы для evaluation retrieval-системы.

Контекст системы:
- MLMS хранит факты агента в pgvector (cosine similarity, Qwen3 embeddings)
- Перед поиском запрос обогащается терминами из session_log (Session-Augmented RAG)
- Факты описывают технические решения, конфигурации, предпочтения IT-агента
- Домены: любой IT-контекст — DevOps, ML, networking, databases, web и т.д.

Требования к выводу:
- Отвечай через native structured output (tool use / function call)
- Поле rationale — всегда первое, пиши рассуждение до заполнения остальных полей
- Не используй домены из списка used_topics\
"""

SYSTEM_VERIFY = "Ты проверяешь корректность ground truth разметки тест-кейсов для evaluation retrieval-системы."

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "lexical": """\
Запрос содержит те же ключевые слова что и релевантный факт.
Это baseline — embedding retrieval должен справляться всегда.
Distractors: факты из той же предметной области, но про другой аспект.
session_log: пустой массив.\
""",
    "semantic_drift": """\
Запрос и релевантный факт описывают одно и то же, но разными словами.
Паттерн: факт — техническая формулировка, запрос — на языке проблемы/результата.
Session_log содержит термины-мосты, которые связывают запрос с фактом при обогащении.
Без session_log cosine similarity между запросом и фактом должна быть ниже порога.

ОБЯЗАТЕЛЬНЫЙ self-check в rationale перед заполнением полей:
  Шаг 1. Выпиши ключевые смысловые слова из query (без стоп-слов).
  Шаг 2. Выпиши ключевые смысловые слова из content релевантного факта.
  Шаг 3. Найди пересечение. Оно должно быть ПУСТЫМ или содержать не более одного слова.
          Если больше — переформулируй query или факт до выполнения условия.
  Шаг 4. Проверь session_log: каждая фраза должна быть семантическим мостом —
          содержать слова близкие одновременно к query И к факту.

ПЛОХОЙ пример:
  fact:  "настроен rate limiting через nginx на 100 req/s"
  query: "как работает rate limiting в nginx"   ← пересечение слишком большое

ХОРОШИЙ пример:
  fact:  "настроен rate limiting через nginx на 100 req/s"
  query: "как мы защитились от перегрузки сервера"   ← пересечение пустое
  session_log: ["проблема с нагрузкой", "лимиты запросов", "nginx конфиг"]\
""",
    "conflict": """\
В БД есть два факта про одно и то же: устаревший (is_relevant=false) и актуальный (is_relevant=true).
Они явно противоречат друг другу — разные значения одного параметра или решения.
Запрос ищет текущее состояние — должен вернуться только актуальный факт.
session_log: пустой массив.\
""",
    "cross_session": """\
Факты хранятся под source_session.chat_id. Запрос выполняется из query_session.chat_id.
Оба chat_id принадлежат одному project_id — факты видны глобально.
query_session.session_log: ВСЕГДА пустой массив. SAR невозможен — только чистые embeddings.

Сценарий должен иметь lexical distance как в semantic_drift: запрос и факт описывают
одно и то же разными словами. Без этого cross_session вырождается в lexical.\
""",
}

# ── Tool schemas ─────────────────────────────────────────────────────────────

_FACT_ITEM = {
    "type": "object",
    "properties": {
        "id":         {"type": "string"},
        "content":    {"type": "string"},
        "is_relevant": {"type": "boolean"},
    },
    "required": ["id", "content", "is_relevant"],
}

_META_PROPS = {
    "tests_session_augmentation": {"type": "boolean"},
    "ab_test":       {"type": "boolean"},
    "difficulty":    {"type": "string", "enum": ["easy", "medium", "hard"]},
    "generated_by":  {"type": "string"},
    "verified_by":   {"type": "string"},
    "auto_fixed":    {"type": "boolean"},
}

_EXPECTED_PROPS = {
    "relevant_ids":   {"type": "array", "items": {"type": "string"}},
    "irrelevant_ids": {"type": "array", "items": {"type": "string"}},
}

_STANDARD_SCENARIO = {
    "type": "object",
    "properties": {
        "id":       {"type": "string"},
        "category": {"type": "string"},
        "domain":   {"type": "string"},
        "setup": {
            "type": "object",
            "properties": {
                "facts":       {"type": "array", "items": _FACT_ITEM, "minItems": 4, "maxItems": 6},
                "session_log": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["facts", "session_log"],
        },
        "query":    {"type": "string"},
        "expected": {
            "type": "object",
            "properties": _EXPECTED_PROPS,
            "required": ["relevant_ids", "irrelevant_ids"],
        },
        "meta": {
            "type": "object",
            "properties": _META_PROPS,
            "required": list(_META_PROPS),
        },
    },
    "required": ["id", "category", "domain", "setup", "query", "expected", "meta"],
}

_CROSS_SESSION_SCENARIO = {
    "type": "object",
    "properties": {
        "id":       {"type": "string"},
        "category": {"type": "string"},
        "domain":   {"type": "string"},
        "setup": {
            "type": "object",
            "properties": {
                "source_session": {
                    "type": "object",
                    "properties": {
                        "chat_id": {"type": "string"},
                        "facts":   {"type": "array", "items": _FACT_ITEM, "minItems": 3, "maxItems": 6},
                    },
                    "required": ["chat_id", "facts"],
                },
                "query_session": {
                    "type": "object",
                    "properties": {
                        "chat_id":    {"type": "string"},
                        "session_log": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["chat_id", "session_log"],
                },
            },
            "required": ["source_session", "query_session"],
        },
        "query":    {"type": "string"},
        "expected": {
            "type": "object",
            "properties": _EXPECTED_PROPS,
            "required": ["relevant_ids", "irrelevant_ids"],
        },
        "meta": {
            "type": "object",
            "properties": _META_PROPS,
            "required": list(_META_PROPS),
        },
    },
    "required": ["id", "category", "domain", "setup", "query", "expected", "meta"],
}


def _generate_tool(category: str) -> dict[str, Any]:
    item_schema = _CROSS_SESSION_SCENARIO if category == "cross_session" else _STANDARD_SCENARIO
    return {
        "type": "function",
        "function": {
            "name": "generate_scenarios",
            "description": "Generate exactly 5 test scenarios for the MLMS retrieval benchmark.",
            "parameters": {
                "type": "object",
                "properties": {
                    "rationale": {
                        "type": "string",
                        "description": (
                            "Free-form reasoning: plan diversity, verify lexical distance "
                            "for semantic_drift/cross_session. Write FIRST before filling scenarios."
                        ),
                    },
                    "scenarios": {
                        "type": "array",
                        "items": item_schema,
                        "minItems": 5,
                        "maxItems": 5,
                    },
                },
                "required": ["rationale", "scenarios"],
            },
        },
    }


_VERIFY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "verify_scenario",
        "description": "Verify the ground truth labeling of a single test scenario.",
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "For each fact: is it actually relevant to the query? Explain.",
                },
                "verdict": {
                    "type": "string",
                    "enum": ["ok", "fix_needed"],
                },
                "corrected_relevant_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Corrected list when verdict=fix_needed; omit if verdict=ok.",
                },
            },
            "required": ["reasoning", "verdict"],
        },
    },
}

# ── OpenRouter API ────────────────────────────────────────────────────────────


async def _call_openrouter(
    model_key: str,
    messages: list[dict[str, Any]],
    tool: dict[str, Any],
    api_key: str,
) -> dict[str, Any]:
    model_id = MODELS[model_key]
    tool_name = tool["function"]["name"]

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:
        for attempt in range(3):
            try:
                r = await client.post(
                    f"{OPENROUTER_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://github.com/mlms",
                        "X-Title": "MLMS Benchmark Generator",
                    },
                    json={
                        "model": model_id,
                        "messages": messages,
                        "tools": [tool],
                        "tool_choice": {"type": "function", "function": {"name": tool_name}},
                        "temperature": 0.8,
                    },
                )
            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                if attempt == 2:
                    raise
                log.warning("network error (attempt %d/3): %s — retrying in 10s", attempt + 1, exc)
                await asyncio.sleep(10.0)
                continue

            if r.status_code == 429:
                wait = float(r.headers.get("retry-after", 2 ** (attempt + 1)))
                log.warning("rate limited, waiting %.0fs", wait)
                await asyncio.sleep(wait)
                continue

            r.raise_for_status()
            data = r.json()
            if "choices" not in data:
                err = data.get("error", data)
                log.warning("no choices from %s (attempt %d/3): %s", model_id, attempt + 1, err)
                await asyncio.sleep(2 ** (attempt + 1))
                continue
            tool_calls = data["choices"][0]["message"].get("tool_calls", [])
            if not tool_calls:
                raise ValueError(f"no tool_call in response from {model_id}")
            return json.loads(tool_calls[0]["function"]["arguments"])

    raise RuntimeError(f"max retries exceeded for {model_id}")


# ── Validation ───────────────────────────────────────────────────────────────


def _fact_ids(scenario: dict[str, Any], category: str) -> list[str]:
    if category == "cross_session":
        return [f["id"] for f in scenario["setup"]["source_session"]["facts"]]
    return [f["id"] for f in scenario["setup"]["facts"]]


def _validate_scenario(scenario: dict[str, Any], category: str) -> bool:
    """Fix expected IDs to match actual fact IDs. Returns False if unfixable."""
    sid = scenario.get("id", "?")

    if category == "cross_session":
        setup = scenario.get("setup", {})
        src = setup.get("source_session")
        qry = setup.get("query_session")
        if not src or not isinstance(src.get("facts"), list):
            log.error("  %s: cross_session missing setup.source_session.facts — drop", sid)
            return False
        if qry is None or "session_log" not in qry:
            log.error("  %s: cross_session missing setup.query_session.session_log — drop", sid)
            return False

    all_ids = _fact_ids(scenario, category)
    if not all_ids:
        log.error("  %s: no facts found", scenario["id"])
        return False

    relevant = [rid for rid in scenario["expected"]["relevant_ids"] if rid in all_ids]
    if not relevant:
        log.error("  %s: no valid relevant_ids", scenario["id"])
        return False

    scenario["expected"]["relevant_ids"] = relevant
    scenario["expected"]["irrelevant_ids"] = [fid for fid in all_ids if fid not in relevant]

    # Enforce session_log rules
    if category == "cross_session":
        scenario["setup"]["query_session"]["session_log"] = []
    elif category in ("lexical", "conflict"):
        scenario["setup"]["session_log"] = []

    return True


# ── Stage 1: Generate ─────────────────────────────────────────────────────────


async def _generate_batch(
    category: str,
    used_topics: list[str],
    gen_key: str,
    api_key: str,
    category_start_idx: int,
) -> list[dict[str, Any]]:
    tool = _generate_tool(category)

    user_prompt = (
        f'Сгенерируй РОВНО 5 тест-кейсов категории "{category}".\n\n'
        f"Уже использованные домены (не повторять):\n{json.dumps(used_topics, ensure_ascii=False)}\n\n"
        f"Определение категории:\n{CATEGORY_DESCRIPTIONS[category]}\n\n"
        "Требования к разнообразию (обязательно):\n"
        "- Стиль запроса: чередуй вопрос / утверждение / неполная фраза\n"
        "- Длина фактов: чередуй короткий / детальный / с числами или кодом\n"
        "- Домены: каждый сценарий — новый домен, не из used_topics\n\n"
        f"ID сценариев: {category}_{str(category_start_idx + 1).zfill(3)} … "
        f"{category}_{str(category_start_idx + 5).zfill(3)}"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_GENERATE},
        {"role": "user", "content": user_prompt},
    ]

    result = await _call_openrouter(gen_key, messages, tool, api_key)
    scenarios: list[dict[str, Any]] = result["scenarios"]

    # Normalise IDs and meta fields
    for i, s in enumerate(scenarios):
        s["id"] = f"{category}_{str(category_start_idx + i + 1).zfill(3)}"
        s["category"] = category
        s.setdefault("meta", {}).update(
            {
                "generated_by": MODEL_SHORT[gen_key],
                "verified_by": "",
                "auto_fixed": False,
                **_CATEGORY_META[category],
            }
        )

    return scenarios


# ── Stage 2: Verify ───────────────────────────────────────────────────────────


async def _verify_scenario(
    scenario: dict[str, Any],
    category: str,
    ver_key: str,
    api_key: str,
    verbose: bool = False,
) -> dict[str, Any]:
    facts = (
        scenario["setup"]["source_session"]["facts"]
        if category == "cross_session"
        else scenario["setup"]["facts"]
    )
    session_log = (
        scenario["setup"]["query_session"]["session_log"]
        if category == "cross_session"
        else scenario["setup"]["session_log"]
    )

    facts_text = "\n".join(
        f"{idx + 1}. [{f['id']}] {f['content']}" for idx, f in enumerate(facts)
    )

    extra = ""
    if category in ("semantic_drift", "cross_session"):
        extra = (
            "\nЕсли категория semantic_drift или cross_session — дополнительно проверь:\n"
            "4. Запрос и релевантный факт не имеют более одного общего ключевого слова.\n"
            "   Если имеют — это плохой сценарий (naive найдёт без SAR), отметь как issue."
        )

    user_prompt = (
        f'Проверь корректность ground truth разметки этого тест-кейса.\n\n'
        f'Запрос: "{scenario["query"]}"\n'
        f'Категория: "{category}"\n'
        f"Session_log: {json.dumps(session_log, ensure_ascii=False)}\n\n"
        f"Факты в БД:\n{facts_text}\n\n"
        f"Заявленные релевантные: {json.dumps(scenario['expected']['relevant_ids'])}\n\n"
        "Критерии релевантности:\n"
        "1. Факт семантически отвечает именно на этот запрос (не просто похож по теме)\n"
        "2. Пользователь получил бы нужную информацию из этого факта\n"
        "3. Факт НЕ является просто distractor'ом похожей тематики"
        f"{extra}"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_VERIFY},
        {"role": "user", "content": user_prompt},
    ]

    result = await _call_openrouter(ver_key, messages, _VERIFY_TOOL, api_key)

    if verbose:
        log.info(
            "  VERIFY %s  verdict=%s\n    reasoning: %s",
            scenario["id"],
            result["verdict"],
            result.get("reasoning", "")[:400],
        )

    scenario["meta"]["verified_by"] = MODEL_SHORT[ver_key]

    if result["verdict"] == "fix_needed":
        corrected: list[str] = result.get("corrected_relevant_ids") or []
        all_ids = _fact_ids(scenario, category)
        valid_corrected = [rid for rid in corrected if rid in all_ids]
        if valid_corrected:
            scenario["expected"]["relevant_ids"] = valid_corrected
            scenario["expected"]["irrelevant_ids"] = [
                fid for fid in all_ids if fid not in valid_corrected
            ]
            # sync is_relevant flags in facts array
            if category == "cross_session":
                for f in scenario["setup"]["source_session"]["facts"]:
                    f["is_relevant"] = f["id"] in valid_corrected
            else:
                for f in scenario["setup"]["facts"]:
                    f["is_relevant"] = f["id"] in valid_corrected
            scenario["meta"]["auto_fixed"] = True
            log.info("    auto_fixed: %s", scenario["id"])

    return scenario


async def _verify_batch(
    scenarios: list[dict[str, Any]],
    category: str,
    ver_key: str,
    api_key: str,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    verified: list[dict[str, Any]] = []
    auto_fixed = 0

    for s in scenarios:
        s = await _verify_scenario(s, category, ver_key, api_key, verbose=verbose)
        if s["meta"]["auto_fixed"]:
            auto_fixed += 1
        verified.append(s)

    return verified, auto_fixed


# ── Checkpoint I/O ────────────────────────────────────────────────────────────


def _load_partial() -> tuple[list[dict[str, Any]], set[str]]:
    if not PARTIAL_PATH.exists():
        return [], set()
    data: dict[str, Any] = json.loads(PARTIAL_PATH.read_text(encoding="utf-8"))
    scenarios: list[dict[str, Any]] = data.get("scenarios", [])
    used_topics: set[str] = {s["domain"] for s in scenarios if s.get("domain")}
    return scenarios, used_topics


def _save_partial(scenarios: list[dict[str, Any]]) -> None:
    PARTIAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    for s in scenarios:
        counts[s["category"]] += 1
    payload = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(scenarios),
        "categories": {cat: counts.get(cat, 0) for cat in CATEGORIES},
        "models_used": list(MODEL_SHORT.values()),
        "scenarios": scenarios,
    }
    PARTIAL_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Orchestration ─────────────────────────────────────────────────────────────


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-batches", type=int, default=TOTAL_BATCHES,
                        help="Stop after N batches (default: all 20)")
    parser.add_argument("--verbose", action="store_true",
                        help="Log full scenario JSON and verify reasoning")
    args = parser.parse_args()
    max_batches: int = args.max_batches
    verbose: bool = args.verbose

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("EMBEDDING_API_KEY")
    if not api_key:
        sys.exit("ERROR: set OPENROUTER_API_KEY in .env")

    scenarios, used_topics = _load_partial()
    start_batch = len(scenarios) // SCENARIOS_PER_BATCH
    end_batch = min(start_batch + max_batches, TOTAL_BATCHES)

    if start_batch >= TOTAL_BATCHES:
        log.info("Already complete (%d scenarios). Delete partial to regenerate.", len(scenarios))
        return

    if start_batch > 0:
        log.info("Resuming from batch %d (%d scenarios already done)", start_batch + 1, len(scenarios))

    for batch_idx in range(start_batch, end_batch):
        category = BATCH_TO_CATEGORY[batch_idx]
        gen_key, ver_key = MODEL_ROTATION[batch_idx % 3]
        category_start = sum(1 for s in scenarios if s["category"] == category)

        log.info(
            "Batch %02d/%d  category=%-14s  gen=%-26s  ver=%s",
            batch_idx + 1,
            TOTAL_BATCHES,
            category,
            MODELS[gen_key],
            MODELS[ver_key],
        )

        # Stage 1 — generate
        batch = await _generate_batch(
            category, list(used_topics), gen_key, api_key, category_start
        )

        if verbose:
            for s in batch:
                log.info("  GENERATED %s\n%s", s["id"], json.dumps(s, ensure_ascii=False, indent=2))

        # Stage 3 — validate (before verify, so verifier sees clean IDs)
        batch = [s for s in batch if _validate_scenario(s, category)]
        if len(batch) < SCENARIOS_PER_BATCH:
            log.warning("  only %d valid scenarios after validation (expected 5)", len(batch))

        # Stage 2 — verify
        batch, auto_fixed = await _verify_batch(batch, category, ver_key, api_key, verbose=verbose)

        # Stage 3 — re-validate after potential fixes
        batch = [s for s in batch if _validate_scenario(s, category)]

        log.info("  auto_fixed=%d  valid=%d", auto_fixed, len(batch))

        # Stage 4 — accumulate
        scenarios.extend(batch)
        for s in batch:
            if s.get("domain"):
                used_topics.add(s["domain"])

        _save_partial(scenarios)

    if len(scenarios) < TOTAL_BATCHES * SCENARIOS_PER_BATCH:
        log.info("Partial run done (%d/%d scenarios). synthetic_100.json not written.",
                 len(scenarios), TOTAL_BATCHES * SCENARIOS_PER_BATCH)
        return

    # Assemble final file
    counts: dict[str, int] = defaultdict(int)
    for s in scenarios:
        counts[s["category"]] += 1

    final_payload = {
        "version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total": len(scenarios),
        "categories": {cat: counts.get(cat, 0) for cat in CATEGORIES},
        "models_used": list(MODEL_SHORT.values()),
        "scenarios": scenarios,
    }
    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_PATH.write_text(json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote %s  (%d scenarios total)", FINAL_PATH, len(scenarios))

    # ── Stats ─────────────────────────────────────────────────────────────────
    print("\n=== auto_fixed stats ===")
    grand_total = 0
    for cat in CATEGORIES:
        fixed = sum(1 for s in scenarios if s["category"] == cat and s["meta"].get("auto_fixed"))
        total = counts.get(cat, 0)
        grand_total += fixed
        print(f"  {cat:<20s}: {fixed:3d} / {total:3d}")
    print(f"  {'TOTAL':<20s}: {grand_total:3d} / {len(scenarios):3d}")


if __name__ == "__main__":
    asyncio.run(main())
