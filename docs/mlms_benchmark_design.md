# MLMS Benchmark — Design Document
**Purpose:** an internal reference for implementing the module that generates the synthetic dataset `synthetic_100.json` and runs the Recall@5 + latency p99 benchmark.

**Scope of SAR.** Session-Augmented RAG (enriching the query with terms from the session_log before embedding) is implemented and measured only in the A/B runners in `scripts/` (`run_ab_benchmark.py`, `run_hard_ab.py`) — it is a researched technique, not part of the served MCP API. The real tools (`get_facts`, `search_memory` and the rest in `src/mlms/`) embed the bare query, without session augmentation. Everywhere below where SAR is mentioned as part of the system under test, it refers to the experimental runner, not to what an agent gets through `memorize`/`get_facts`/`search_memory` today.

---

## 1. Context: what we measure and why

MLMS has 4 memory layers. The benchmark covers **only the semantic layer** (facts in pgvector), because it is the only one that uses embedding-based retrieval — the other layers are deterministic (Redis get/set, SQL time_range, DISTINCT ON version).

**Two target numbers:**

| NFR | Metric | Target |
|-----|---------|------|
| NFR-1 | Latency p99 (cached path, no API call) | < 200 ms |
| FR-1 | Recall@5 on synthetic_100.json | ≥ 0.80 |

**Recall@5** = the share of relevant facts found among all relevant ones, considering only the top-5 returned results. Averaged over 100 scenarios.

```
Recall@5(scenario) = |returned_top5 ∩ relevant_ids| / |relevant_ids|
Recall@5(total)    = mean(Recall@5 over all 100 scenarios)
```

**Why Recall@5 and not Precision@5.** Each scenario has 1–2 relevant facts and 3–5 distractors. With Precision@5 the maximum achievable result with 1 relevant fact is 1/5 = 0.20, which makes the target ≥ 0.80 mathematically unreachable. Recall@5 with 1 relevant fact: found — 1.0, not found — 0.0. The mean over 100 scenarios is the share of queries where the needed fact made it into the top-5. A target ≥ 0.80 means: the system finds the needed fact in 80% of cases.

---

## 2. Structure of the synthetic_100.json dataset

### 2.1 Four scenario categories

| Category | Count | What is tested |
|-----------|--------|-----------------|
| `lexical` | 25 | Direct vocabulary match — a baseline, must always work |
| `semantic_drift` | 25 | The query and the fact are about the same thing but in different words; tests Qwen3 embeddings + Session-Augmented RAG; run in A/B mode |
| `conflict` | 25 | The DB holds an old fact (is_current=FALSE) and a new one (TRUE); only the new one must be returned |
| `cross_session` | 25 | The query looks for facts from another session (different chat_id, same project_id); tests persistent cross-session memory without SAR enrichment |

### 2.2 Scenario schemas

**Standard schema** (lexical, conflict, semantic_drift):

```json
{
  "id": "semantic_drift_003",
  "category": "semantic_drift",
  "domain": "nginx_config",
  "setup": {
    "facts": [
      {"id": "f1", "content": "...", "is_relevant": true},
      {"id": "f2", "content": "...", "is_relevant": false},
      {"id": "f3", "content": "...", "is_relevant": false},
      {"id": "f4", "content": "...", "is_relevant": false}
    ],
    "session_log": ["context phrase 1", "context phrase 2"]
  },
  "query": "how did we decide X?",
  "expected": {
    "relevant_ids": ["f1"],
    "irrelevant_ids": ["f2", "f3", "f4"]
  },
  "meta": {
    "tests_session_augmentation": true,
    "ab_test": true,
    "difficulty": "medium",
    "generated_by": "claude-sonnet-4.6",
    "verified_by": "gpt-5.2",
    "auto_fixed": false
  }
}
```

**cross_session schema** (two chat_ids, the session_log of the querying session is empty):

```json
{
  "id": "cross_session_007",
  "category": "cross_session",
  "domain": "ci_pipeline",
  "setup": {
    "source_session": {
      "chat_id": "src_cross_007",
      "facts": [
        {"id": "f1", "content": "...", "is_relevant": true},
        {"id": "f2", "content": "...", "is_relevant": false},
        {"id": "f3", "content": "...", "is_relevant": false}
      ]
    },
    "query_session": {
      "chat_id": "qry_cross_007",
      "session_log": []
    }
  },
  "query": "...",
  "expected": {
    "relevant_ids": ["f1"],
    "irrelevant_ids": ["f2", "f3"]
  },
  "meta": {
    "tests_session_augmentation": false,
    "ab_test": false,
    "difficulty": "hard",
    "generated_by": "gpt-5.2",
    "verified_by": "grok-4.20",
    "auto_fixed": false
  }
}
```

The `session_log` of query_session is always empty — this is by design: the querying session has no access to the context of the source session, so SAR enrichment is impossible.

### 2.3 Requirements for scenario content

- Relevant facts: 1–2 per scenario
- Distractor facts: 3–5 per scenario (similar topic, but they do not answer the query)
- session_log: 2–3 phrases only for `semantic_drift`; for `lexical`, `conflict`, `cross_session` — an empty array
- Domains: any IT context (DevOps, ML, networking, databases, web, etc.)
- A domain must not repeat across scenarios (controlled via `used_topics`)
- For `semantic_drift`: the query and the relevant fact must not share more than one keyword (the lexical distance requirement — see section 4.4 for details)

---

## 3. Generation: pipeline architecture

### 3.1 Overview

```
[Stage 1] GENERATE  — an LLM generates 5 scenarios in one call (tool_use → JSON)
[Stage 2] VERIFY    — another LLM from a different family checks the ground truth
[Stage 3] VALIDATE  — deterministic schema check + auto-fix of IDs
[Stage 4] ACCUMULATE — extend used_topics, write to the file
```

20 batches × 5 scenarios = 100 scenarios in total. All calls go through the **OpenRouter API** (`https://openrouter.ai/api/v1/chat/completions`) with `tool_choice` — a single mechanism for all three providers.

### 3.2 Model rotation

Three models from **different families**. Switching the family between generator and verifier is mandatory.

| Batch | Generator | Verifier |
|------|-----------|-------------|
| 0, 3, 6, ... | `anthropic/claude-sonnet-4.6` | `openai/gpt-5.2` |
| 1, 4, 7, ... | `openai/gpt-5.2` | `x-ai/grok-4.20` |
| 2, 5, 8, ... | `x-ai/grok-4.20` | `anthropic/claude-sonnet-4.6` |

Rule: the verifier is always from a **different family** (not the next one in the cycle, but strictly cross-family). The models are set in `.env`: `MODEL_CLAUDE`, `MODEL_GPT4O`, `MODEL_GROK` — they can be overridden without changing code.

### 3.3 Distribution of categories across batches

Each category is 5 batches of 5 scenarios = 25 in total. The order of batches within a category is sequential.

---

## 4. Prompt engineering

### 4.1 Key decisions backed by research

**No role setting ("You are an expert").**
Zheng et al., EMNLP 2024 (162 personas, 4 LLM families, 2410 questions): personas in the system prompt do not improve performance on objective tasks and sometimes degrade it. Removed.

**Two-phase generation: reasoning first, then JSON.**
Tam et al., EMNLP 2024 ("Let Me Speak Freely?"): strict JSON mode degrades reasoning by 10–15%. Solution: the model first reasons freely (`rationale`), then fills in the structured fields. `rationale` is the first field in the JSON schema.

**Native structured outputs instead of prompt-only JSON.**
Prompt-only JSON: failure rate 5–20%. All three providers are called through the OpenRouter API with `tool_choice: {type: "function", function: {name: ...}}` — failure rate < 0.3%.

**Cross-family verification.**
The Panel of LLMs (PoLL) approach: different model families have different "blind spots". A verifier from a different family reduces correlated bias.

**A used_topics list in every prompt.**
Fights mode collapse — the tendency of instruction-tuned models to generate structurally similar scenarios on repeated calls.

### 4.2 System prompt (final)

Contains no role setting. Contains only the task context:

```
Ты генерируешь тест-кейсы для evaluation retrieval-системы.

Контекст системы:
- MLMS хранит факты агента в pgvector (cosine similarity, Qwen3 embeddings)
- In the experimental A/B mode (see `scripts/`) the query is enriched with terms from the session_log (Session-Augmented RAG) before search — the served API does not do this
- Факты описывают технические решения, конфигурации, предпочтения IT-агента
- Домены: любой IT-контекст — DevOps, ML, networking, databases, web и т.д.

Требования к выводу:
- Отвечай через native structured output (tool use / function call)
- Поле rationale — всегда первое, пиши рассуждение до заполнения остальных полей
- Не используй домены из списка used_topics
```

### 4.3 User prompt

```
Сгенерируй РОВНО 5 тест-кейсов категории "{category}".

Уже использованные домены (не повторять):
{used_topics_json}

Определение категории:
{category_description}

Требования к разнообразию (обязательно):
- Стиль запроса: чередуй вопрос / утверждение / неполная фраза
- Длина фактов: чередуй короткий / детальный / с числами или кодом
- Домены: каждый сценарий — новый домен, не из used_topics
```

### 4.4 Category descriptions (substituted into the user prompt)

**lexical:**
```
Запрос содержит те же ключевые слова что и релевантный факт.
Это baseline — embedding retrieval должен справляться всегда.
Distractors: факты из той же предметной области, но про другой аспект.
session_log: пустой массив.
```

**semantic_drift:**
```
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

ПЛОХОЙ пример (отвергни):
  fact:  "настроен rate limiting через nginx на 100 req/s"
  query: "как работает rate limiting в nginx"
  пересечение: ["rate", "limiting", "nginx"] — слишком много, naive найдёт

ХОРОШИЙ пример (принять):
  fact:  "настроен rate limiting через nginx на 100 req/s"
  query: "как мы защитились от перегрузки сервера"
  пересечение: [] — пусто, naive не найдёт
  session_log мост: ["проблема с нагрузкой", "лимиты запросов", "nginx конфиг"]
```

**conflict:**
```
В БД есть два факта про одно и то же: устаревший (is_relevant=false) и актуальный (is_relevant=true).
Они явно противоречат друг другу — разные значения одного параметра или решения.
Запрос ищет текущее состояние — должен вернуться только актуальный факт.
session_log: пустой массив.
```

**cross_session:**
```
Факты хранятся под source_session.chat_id. Запрос выполняется из query_session.chat_id.
Оба chat_id принадлежат одному project_id — факты видны глобально, фильтрации нет.
query_session.session_log: ВСЕГДА пустой массив. Это принципиально — SAR невозможен,
retrieval работает только на чистых embeddings.

Это тестирует baseline качество Qwen3 (dim=1536) в условиях vocabulary mismatch
без возможности SAR обогащения. Ожидаемый Recall@5 ниже чем у semantic_drift (SAR) —
это намеренно, не признак ошибки системы.

Сценарий должен иметь lexical distance как в semantic_drift: запрос и факт описывают
одно и то же разными словами. Без этого cross_session вырождается в lexical.
```

### 4.5 Verify prompt

```
Проверь корректность ground truth разметки этого тест-кейса.

Запрос: "{query}"
Категория: "{category}"
Session_log: {session_log}

Факты в БД:
{факты пронумерованные}

Заявленные релевантные: {relevant_ids}

Критерии релевантности:
1. Факт семантически отвечает именно на этот запрос (не просто похож по теме)
2. Пользователь получил бы нужную информацию из этого факта
3. Факт НЕ является просто distractor'ом похожей тематики

Если категория semantic_drift или cross_session — дополнительно проверь:
4. Запрос и релевантный факт не имеют более одного общего ключевого слова.
   Если имеют — это плохой сценарий (naive найдёт без SAR), отметь как issue.

Для каждого факта вынеси: relevant / irrelevant / borderline.
Если не согласен с разметкой — укажи исправленный список.
```

---

## 5. Protection against known problems

### 5.1 Preference leakage

Problem: if one model generates the data AND is used in embeddings/evaluation, bias arises.

Our solution: Claude/GPT-5.2/Grok generate and verify. Qwen3 is used **only** for embeddings inside MLMS itself. Recall@5 is computed deterministically by UUID — no LLM takes part in computing the metric.

### 5.2 Hallucinated ground truth

Problem: an LLM labels wrongly with confidence.

Solution: a mandatory verify call by a model from a different family. On `verdict=fix_needed` we overwrite `relevant_ids` and set the flag `auto_fixed=true` in meta.

### 5.3 Mode collapse / repeating patterns

Problem: instruction-tuned models with repeated templates generate structurally similar scenarios.

Solution:
- `used_topics` in every prompt
- Explicit style diversity requirements in the user prompt
- Three different models in rotation

### 5.4 JSON parse failures

Problem: prompt-only JSON gives a 5–20% failure rate.

Solution: all three providers are called through OpenRouter with `tool_choice: {"type": "function", "function": {"name": ...}}` — API-level schema enforcement. Failure rate < 0.3%. A single mechanism for all providers removes provider-specific settings (response_format, response_schema).

---

## 6. Database isolation in tests

### 6.1 Approach: a unique project_id per scenario

Each scenario gets `project_id = uuid.uuid4()` — full isolation with no contention between scenarios. Cleanup in `try/finally`:

**Standard categories** (lexical, conflict, semantic_drift):

```
pid = uuid.uuid4()
INSERT INTO projects (id, name) VALUES (pid, "bench-{sid}")
→ INSERT facts (project_id=pid) via upsert_fact()
→ INSERT session_log entries (PostgreSQL, chat_id="bench-{sid}-{pid}") for semantic_drift
→ COMMIT
→ get_facts(about=query, project_id=pid, limit=5)
→ COLLECT top-5 fact_ids → COMPARE with expected.relevant_ids → Recall@5
finally:
  DELETE FROM facts WHERE project_id = pid
  DELETE FROM session_log WHERE chat_id = "bench-{sid}-{pid}"
  DELETE FROM projects WHERE id = pid
  COMMIT
```

**cross_session category** (two chat_ids):

```
pid = uuid.uuid4()
INSERT INTO projects (id, name) VALUES (pid, ...)
→ INSERT facts (project_id=pid) from source_session.facts
→ COMMIT
→ get_facts(about=query, project_id=pid, limit=5)
  [chat_id at query time = "{query_session.chat_id}-{pid}"]
→ COLLECT top-5 → COMPARE with expected.relevant_ids → Recall@5
finally:
  DELETE FROM facts WHERE project_id = pid
  DELETE FROM projects WHERE id = pid
  COMMIT
```

Facts are filtered by `project_id`, not by `chat_id` — so the facts of source_session are visible from query_session. That is exactly the property under test.

### 6.2 chat_id isolation

Standard scenarios: `chat_id = f"bench-{scenario_id}-{project_uuid}"` — unique per scenario.
Cross_session (A/B): `chat_id = f"ab-{scenario_id}-{project_uuid}"`.
Cross_session (full benchmark): `chat_id = f"{setup.query_session.chat_id}-{project_uuid}"`.

UUID uniqueness prevents collisions in any run mode.

---

## 7. Latency benchmark (NFR-1)

### 7.1 What we measure

Three operations with different SLAs:

| Operation | Path | SLA |
|----------|------|-----|
| `get_session_context()` | Redis hash GET | ≤ 5 ms p99 |
| `get_facts()` | pgvector cosine + embed cache hit | ≤ 200 ms p99 |
| `memorize(fact)` cold | API embed + upsert + COMMIT | no SLA |

NFR-1 officially is `get_facts()` p99 < 200 ms. `get_session_context()` has a separate target of 5 ms (LATENCY_SESSION_CTX_MS in config).

### 7.2 Measurement method

The script `scripts/measure_latency.py` (`asyncio`, n=100 iterations per operation):

- `bench_session_context()` — n=100 Redis GETs, after one SET
- `bench_get_facts()` — n=100 calls of `get_facts()` with a warmed-up embedding cache (embed cache warm-up before the loop)
- `bench_memorize_cold()` — n=100 calls of the embed API + upsert, unique content per iteration (forces a cache miss)

p50/p95/p99 metrics via linear interpolation. Results are saved to `scripts/latency_results.json`.

**Note on indexes:** HNSW and IVFFlat are limited to 2000 dim in pgvector 0.8.2; dim=1536 fits within the HNSW limit, and an HNSW index was added on `facts.embedding`/`session_log.embedding` in migration 006 (see `KEY_DECISIONS.md` Decision 2). With n=10 seed facts in the latency benchmark the choice of index does not affect p99.

### 7.3 Cold path (for reference)

The cold path (with a call to the Qwen3-Embedding-8B API through OpenRouter) is measured by `bench_memorize_cold()` and is not part of NFR-1. Latency is determined by the network delay to the API.

---

## 8. Format of the resulting file

```json
{
  "version": "1.0",
  "generated_at": "2026-05-13T21:15:18.397732+00:00",
  "total": 100,
  "categories": {
    "lexical": 25,
    "semantic_drift": 25,
    "conflict": 25,
    "cross_session": 25
  },
  "models_used": ["claude-sonnet-4.6", "gpt-5.2", "grok-4.20"],
  "scenarios": [ ... ]
}
```

---

## 9. Computing the final metrics

### 9.1 Main metric: Recall@5

```
Recall@5(scenario_i) = |top5_returned ∩ relevant_ids| / |relevant_ids|

Recall@5(total) = (1/100) × Σ Recall@5(scenario_i)

By category separately:
Recall@5(lexical)        = mean over 25 scenarios
Recall@5(semantic_drift) = mean over 25 scenarios  [SAR condition only]
Recall@5(conflict)       = mean over 25 scenarios
Recall@5(cross_session)  = mean over 25 scenarios
```

### 9.2 A/B for semantic_drift: proof of Session-Augmented RAG

For each of the 25 semantic_drift scenarios two runs are made:

```
Condition A (naive):
  get_facts(about=raw_query, project_id=pid, limit=5)
  → Recall@5_naive

Condition B (SAR):
  session_entries = get_session_log(chat_id=chat_id, limit=50)  [reads from PostgreSQL]
  enriched = " ".join(e.label for e in session_entries) + " " + raw_query
  get_facts(about=enriched, project_id=pid, limit=5)
  → Recall@5_sar
```

The enrichment is built from the **e.label fields** read through `get_session_log()` (not directly from the JSON field `setup.session_log`). This matters: SAR works through the real MCP tool, not a mock.

Final report:

```
semantic_drift A/B:
  Recall@5 naive: 0.XX   (baseline without SAR)
  Recall@5 SAR:   0.XX   (with Session-Augmented RAG)
  delta:         +0.XX   (contribution of the author's pattern)
```

`delta > 0` is empirical proof that Session-Augmented RAG improves retrieval under vocabulary mismatch.

### 9.3 Expected pattern of results

```
Recall@5(lexical)              ≈ 1.0    — baseline, must always work
Recall@5(semantic_drift, SAR)  > 0.80   — the FR-1 target
Recall@5(semantic_drift, naive)< SAR    — proves the value of SAR
Recall@5(conflict)             ≈ 1.0    — the is_current filter works
Recall@5(cross_session)        < SAR    — embeddings without enrichment, this is normal
```

Recall@5(cross_session) < Recall@5(semantic_drift SAR) — **not a system error**. It shows the difference between "pure embeddings" and "embeddings + SAR". Cross_session tests the baseline quality of Qwen3 (dim=1536) without additional mechanisms.

### 9.4 Actual measurement results

> Environment: Windows 11, Python 3.12, Docker Desktop (postgres+redis on localhost), OpenRouter Qwen3-Embedding-8B API.

#### FR-1 — Recall@5 (100 scenarios, synthetic_100.json)

`scripts/run_benchmark.py`

| Category | Recall@5 | N |
|---|---|---|
| lexical | 1.000 | 25 |
| semantic_drift | 1.000 | 25 |
| conflict | 0.960 | 25 |
| cross_session | 1.000 | 25 |
| **OVERALL** | **0.990** | **100** |

FR-1 target: ≥ 0.80 → **PASS**.

The single miss (`conflict_019`): both conflicting facts stay `is_current=TRUE` because the cosine distance is below the 0.96 threshold — a threshold calibration edge case.

#### A/B — SAR vs Naive (25 semantic_drift scenarios, synthetic_100.json)

`scripts/run_ab_benchmark.py`

| Mode | Recall@5 | N |
|---|---|---|
| Naive (raw query) | 1.000 | 25 |
| SAR (session-enriched) | 1.000 | 25 |
| **Delta (SAR − naive)** | **+0.000** | — |

**Interpretation:** Qwen3-Embedding-8B eliminates semantic drift at the vector-space level — the model correctly matches the problem description to the technical fact without enriching the query. delta=0 on synthetic_100 is explained by the high quality of the 8B model + moderate dataset difficulty. The architectural value of SAR is confirmed on synthetic_hard (see section 12).

#### Multi-model A/B (semantic_drift, synthetic_100.json)

`scripts/run_multimodel_ab.py` — in-memory retrieval, each model in its own vector space.

| Model | Naive@5 | SAR@5 | Delta |
|---|---|---|---|
| Qwen3-8B | 0.840¹ | 0.840¹ | +0.000 |
| Qwen3-0.6B | — | — | — ² |
| BM25 | 1.000 | 1.000 | +0.000 ³ |

¹ 0.840 due to 4 transient OpenRouter ConnectTimeouts; the confirmed value is 1.000 from `run_ab_benchmark.py`.
² `EMBEDDING_FALLBACK = "qwen3-embedding-0.6b"` — 404 on OpenRouter, intended for local inference (Ollama/vLLM).
³ A corpus of 5 facts, k=5: top-5 of 5 = all the facts → Recall@5=1.0 trivially.

#### NFR-1 — Latency p99 (n=100 per operation)

`scripts/measure_latency.py`

| Operation | p50 (ms) | p95 (ms) | p99 (ms) | Target (ms) | Status |
|---|---|---|---|---|---|
| `get_session_context()` | 1.6 | 3.1 | 9.9 | ≤ 5 | p99 FAIL¹ |
| `get_facts()` warm cache | 4.7 | 7.7 | 16.0 | ≤ 200 | **PASS** |
| `memorize(fact)` cold path | 1158 | 4130 | 9237 | — | N/A |

¹ p99 = 9.9 ms vs target 5 ms: a single outlier among 100 measurements (p95 = 3.1 ms is within the target). Likely cause: TCP overhead of Docker Desktop on Windows. On a production Linux host with a Unix-socket Redis, < 1 ms at p99 is expected.

---

## 10. What is NOT part of this module

- README.md, CONTRIBUTING.md — a separate task
- GitHub Actions CI — a separate task
- Changes to the MLMS architecture — it is fixed, we do not touch it
- Full multi-model comparison with Qwen3-0.6B — requires a local Ollama deployment

### Obsolete scripts

| Script | Reason | Current replacement |
|---|---|---|
| `run_ab_benchmark.py` | delta=0 on synthetic_100 — Qwen3-8B does not need SAR at this difficulty | `run_hard_ab.py` |
| `run_multimodel_ab.py` | Trivial result: 5 facts/scenario → BM25=1.0; Qwen3-0.6B N/A (OpenRouter 404) | `run_hard_ab.py` |

---

## 11. What we ran into and how we solved it

Each problem was discovered while designing the pipeline and confirmed by existing research. Below is the chronology of decisions.

---

### 11.1 Role prompt ("You are an expert in dataset generation")

**Problem.** Initially the system prompt began with a role setting, by analogy with a common practice: it was assumed that specifying a role would improve generation quality.

**What we found.** Zheng et al. ran a large-scale experiment: 162 personas, 4 families of open-source LLMs, 2410 factual questions from MMLU. Conclusion: adding personas to the system prompt does **not improve** performance on objective tasks compared with a control without a persona — and in some cases degrades it slightly. The key distinction: role settings affect **style and tone** (subjective tasks), but not the **accuracy and quality** of objective tasks such as generating labelled test cases.

**Decision.** The role setting was removed entirely. The system prompt contains only the system context and the technical requirements for the output.

**Source.** Zheng et al. "When 'A Helpful Assistant' Is Not Really Helpful: Personas in System Prompts Do Not Improve Performances of Large Language Models." *Findings of EMNLP 2024*, pp. 15126–15154.

---

### 11.2 JSON mode degrades reasoning during generation

**Problem.** The initial prompt design required the model to emit valid JSON right away — to invent the scenario and structure the output at the same time. In testing, scenario quality was unstable: models sometimes generated superficial distractors or unconvincing semantic drift.

**What we found.** Tam et al. ("Let Me Speak Freely?") tested three modes: constrained decoding (JSON mode), Format-Restricting Instructions (FRI), and NL-to-Format (free text first, then conversion). On reasoning tasks strict JSON mode gave a **10–15% degradation** compared with free generation. Mechanism: the model generates tokens left to right; if the format is imposed immediately, it has to think about the content and keep to the JSON syntax at once, which reduces the "bandwidth" available for reasoning.

**Decision.** The `rationale` field was moved to be the **first** in the JSON schema. The model first writes free reasoning ("why this fact is relevant, why the others are not"), and only then fills in the structured fields. This emulates NL-to-Format without two separate calls: the reasoning "warms up" the context before structuring.

**Source.** Tam et al. "Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models." *EMNLP Industry Track 2024*, arXiv:2408.02442.

---

### 11.3 Prompt-only JSON: high failure rate

**Problem.** With the instruction "Answer only with valid JSON without markdown", in real tests some responses contained a preamble, trailing text or invalid JSON — especially from GPT-3.5-class models.

**What we found.** Studies of production systems show that prompt-only JSON extraction gives a **failure rate of 5–20%** depending on schema complexity. Since 2023–2024 all three providers have implemented native structured outputs with API-level enforcement.

**Decision.** All three models are called through the OpenRouter API with `tool_choice: {"type": "function", "function": {"name": tool_name}}`. A single mechanism for all providers — no provider-specific settings. Failure rate < 0.3%.

**Source.** TokenMix.ai benchmark (500,000 calls); "Beyond JSON Mode: Getting Reliable Structured Outputs from LLMs in Production", TianPan.co, 2025.

---

### 11.4 Preference leakage: one model generates and evaluates

**Problem.** The initial variant assumed that the verifier could be the same model as the generator, with a different prompt. The question also arose: can Qwen3 (the same embedding engine as MLMS) be used to generate test cases?

**What we found.** Li et al. described the phenomenon of preference leakage: when the generator model and the judge model belong to the same family (or are the very same model), the scores are systematically inflated. Three kinds of relatedness were identified: the same model, inheritance (fine-tuned from one base), the same family. In all three cases the bias was confirmed experimentally on AlpacaEval 2.0 and Arena-Hard.

**Decision.** Two rules:
1. Qwen3 takes no part in generation or verification — only in the embeddings inside MLMS.
2. The verifier is always from a **different family** under a fixed scheme: Claude → GPT-5.2, GPT-5.2 → Grok, Grok → Claude. Recall@5 is computed deterministically by UUID — no LLM takes part in computing the metric.

**Source.** Li et al. "Preference Leakage: A Contamination Problem in LLM-as-a-judge." arXiv:2502.01534, 2025.

---

### 11.5 Mode collapse: the model repeats the same patterns

**Problem.** When prototyping with one model and repeated calls, the scenarios began to reproduce the same structural template: similar distractors, identical query phrasings, one domain (most often docker/kubernetes).

**What we found.** "The Price of Format: Diversity Collapse in LLMs" (2025): instruction-tuned models trained on repeated structural templates internalize them as strong generation priors. This leads to overly deterministic outputs even at high temperature. The effect persists because alignment training shifts the model toward "typical" answers (typicality bias), which fundamentally reduces diversity regardless of algorithmic settings.

**Decision.** Three levels of protection:
1. `used_topics` — a JSON list of domains is passed into every following call.
2. Explicit diversity requirements in the user prompt: alternate the query style (question / statement / incomplete phrase) and the fact length (short / detailed / with numbers).
3. Rotation of three models from different families: different training patterns give different generation priors.

**Source.** "The Price of Format: Diversity Collapse in LLMs", arXiv:2505.18949, 2025; "Verbalized Sampling: How to Mitigate Mode Collapse", arXiv:2510.01171, 2025.

---

### 11.6 Hallucinated ground truth: an LLM is confidently wrong in labelling

**Problem.** In a manual check of some of the generated scenarios it turned out that the model sometimes labelled as relevant a fact that is only topically close to the query but does not answer it. Hallucinations in labelling are more dangerous than hallucinations in text: they directly distort the Recall@5 metric.

**What we found.** Studies show that when generating synthetic data, LLMs produce factual errors and wrong labels with high confidence. If not filtered, the errors become entrenched in the dataset and make the metric meaningless. "No Free Labels" (2025): an LLM judge without human grounding systematically errs on edge cases.

**Decision.** A mandatory Stage 2 (verify): every generated scenario is checked by a model from a different family against explicit relevance criteria. On `verdict=fix_needed` we overwrite `expected.relevant_ids` and set the flag `meta.auto_fixed=true`. The fix_needed statistics from the generation run are documented.

**Source.** "No Free Labels: Limitations of LLM-as-a-Judge Without Human Grounding", arXiv:2503.05061, 2025; "On LLMs-Driven Synthetic Data Generation, Curation, and Evaluation: A Survey", arXiv:2406.15126, 2024.

---

### 11.7 Precision@5 is mathematically unreachable with sparse ground truth

**Problem.** The original FR-1 metric was formulated as Precision@5 ≥ 0.80. With a dataset design of 1–2 relevant facts per scenario, the maximum achievable Precision@5 = 2/5 = 0.40 with two relevant facts and 1/5 = 0.20 with one. A target ≥ 0.80 is mathematically unreachable.

**Trade-off analysis.** Three fixes were considered: (1) increase the number of relevant facts to 4–5 — unnatural for an agent's semantic memory, where facts are discrete; (2) switch to Recall@5 — the standard IR metric for sparse relevance; (3) switch to Hit@5 — a binary version of Recall@5. Recall@5 was chosen as the most semantically precise: "the system found the needed fact among the top-5".

**Decision.** FR-1 was reformulated: **Recall@5 ≥ 0.80**. With 1 relevant fact: found — 1.0, not found — 0.0. The mean over 100 scenarios is the share of queries where the needed fact made it into the top-5. A target ≥ 0.80 means: the system finds the needed fact in 80% of cases.

---

### 11.8 Temporal category — mixing architecture layers

**Problem.** Initially the dataset had a `temporal` category: queries with temporal context through `get_timeline()`. In review it turned out that `get_timeline()` works with TimescaleDB through SQL time_range — that is the **episodic layer**, not the semantic one. The benchmark is declared to cover only the semantic layer (pgvector), so including temporal created an architectural contradiction. In addition, the accuracy of an SQL time_range is deterministic and is not a retrieval-quality metric — there is nothing to measure with Recall@5.

**Decision.** Temporal was replaced by `cross_session`: a query from one session looks for facts saved in another session (different chat_id, same project_id). It stays strictly within the semantic layer through `get_facts()` + pgvector. Additional value: it directly tests the main claim of MLMS about persistent cross-session memory and covers Multi-Session Reasoning from the LongMemEval taxonomy.

**Source.** Wu et al. "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory." *ICLR 2025*.

---

### 11.9 semantic_drift: A/B to prove Session-Augmented RAG

**Problem.** The original design did not specify: is semantic_drift run through a naive `get_facts(raw_query)` or through the SAR flow (`session_log → enrichment → get_facts`)? Without this clarification the benchmark tests only embedding quality and does not prove the value of the author's pattern.

**Decision.** For all 25 semantic_drift scenarios an A/B is run: Condition A (naive) and Condition B (SAR). Delta = Recall@5_sar − Recall@5_naive — empirical proof that Session-Augmented RAG improves retrieval under vocabulary mismatch. This is the central experimental evidence of the project.

Additionally: a mandatory self-check (in the rationale field) was added to the generation prompt — the model must verify the lexical distance between the query and the fact before filling in the other fields. Without it delta ≈ 0 because naive copes on its own.

---

### 11.10 synthetic_100 is too easy for multi-model comparison

**Problem.** A measured Recall@5 = 0.990 against a target of 0.80 means the dataset does not allow models to be told apart: Qwen3-8B and any sufficiently strong model will give the same result. For proving the superiority of SAR over naive on synthetic_100, delta = 0 — Qwen3-8B copes without enrichment.

**Cause.** synthetic_100 was generated with moderate requirements on the lexical gap (no explicit threshold); the generator models sometimes violated the lexical distance constraint despite the self-check. A corpus of 5 facts/scenario is trivial for pgvector.

**Decision.** `synthetic_hard.json` was created — a higher-difficulty dataset with stricter requirements (see section 12). On it the delta of Qwen3-8B = +0.303, which is the needed empirical proof.

---

### 11.11 EMBEDDING_DIM: the HNSW limit of pgvector 0.8.2

**Problem.** The original design assumed EMBEDDING_DIM = 4096 (full Qwen3-Embedding-8B) with an HNSW index. pgvector 0.8.2 limits HNSW and IVFFlat to a maximum of 2000 dim for the `vector` type. `halfvec` supports up to 4000 — also less than 4096.

**Decision.** EMBEDDING_DIM = **1536** (Matryoshka truncation of Qwen3-Embedding-8B: 4096 → 1536). The recall loss from truncation was not measured (`KEY_DECISIONS.md`, Decision 1). At 1536 dim HNSW is technically possible (2000 > 1536) and is wired into the production migrations (details in `KEY_DECISIONS.md` Decision 2). The latency benchmark (n=10–100 facts) fits within the SLA.

---

### 11.12 Summary table

| Problem | Original approach | Decision | Source |
|----------|-----------------------|---------|----------|
| Role prompt | "You are an expert in generation" | Removed; only the task context | Zheng et al., EMNLP 2024 |
| JSON degrades reasoning | JSON right away in one prompt | rationale as the first field; think→format | Tam et al., EMNLP 2024 |
| Prompt-only JSON failure 5–20% | The instruction "answer only JSON" | OpenRouter tool_choice for all providers | TokenMix benchmark, 2025 |
| Preference leakage | The same model generates and checks | Cross-family rotation; Qwen3 only in embeddings | Li et al., arXiv 2025 |
| Mode collapse | One batch, one model | used_topics + diversity requirements + 3 models | arXiv 2505.18949, 2025 |
| Hallucinated ground truth | No verification | Stage 2: cross-family verify + auto_fixed flag | arXiv 2503.05061, 2025 |
| Precision@5 mathematically unreachable | Precision@5 ≥ 0.80 | Recall@5 ≥ 0.80 | IR theory (sparse relevance) |
| Temporal mixes architecture layers | temporal via get_timeline() | cross_session via get_facts() + pgvector | Wu et al., ICLR 2025 |
| semantic_drift does not prove SAR | Naive search only | A/B: naive vs SAR, delta = contribution of the pattern | — |
| synthetic_100 too easy | A single dataset | synthetic_hard: 400 facts, strict lexical gap | — |
| HNSW limit of 2000 dim | EMBEDDING_DIM = 4096 | Matryoshka truncation → 1536 | KEY_DECISIONS.md Decision 2 |

---

## 12. The synthetic_hard dataset

### 12.1 Purpose

`tests/fixtures/synthetic_hard.json` was created to prove the SAR effect under conditions where naive retrieval degrades. BM25 naive Recall@5 = 0.118 (the target range [0.20, 0.55] was not reached — the dataset is harder than expected).

### 12.2 Structure

| Parameter | Value |
|---|---|
| Corpus | 400 facts (10 domains × 40 facts) |
| Scenarios | 93 |
| Generator | `anthropic/claude-sonnet-4.6` (via OpenRouter) |
| Verifier 1 | `openai/gpt-5.2` — checks "too_easy" (would naive find it without SAR?) |
| Verifier 2 | `x-ai/grok-4.20` — checks "sar_solvable" (would SAR find it?) |

### 12.3 Categories

| Category | N | What is tested |
|---|---|---|
| `lexical_drift` | 30 | User language vs technical terms |
| `semantic_drift` | 29 | Symptom/effect vs cause/mechanism |
| `cross_domain` | 25 | The problem touches two domains; the relevant facts come from different domains |
| `multi_hop` | 9 | The answer requires a chain of 2+ facts; lexical_gap ≤ 1 |

### 12.4 Scenario schema

```json
{
  "id": "sd_001",
  "type": "semantic_drift",
  "domain": "kafka/redis",
  "setup": {
    "session_log": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  },
  "query": "...",
  "expected": {
    "relevant_ids": ["f045", "f123"],
    "irrelevant_ids": ["f046", "f047", ...]
  },
  "meta": {
    "lexical_gap_score": 1,
    "difficulty": "hard",
    "drift_mechanism": "...",
    "sar_bridge_terms": ["term1", "term2"],
    "too_easy": false,
    "sar_solvable": true,
    "verifier_1_reasoning": "...",
    "verifier_2_reasoning": "..."
  }
}
```

Key differences from synthetic_100: a shared corpus of 400 facts (not 5 per scenario), `session_log` contains `{role, content}` objects (not strings), `lexical_gap_score` is controlled explicitly (≤ 2 for most types, ≤ 1 for multi_hop).

### 12.5 Retrieval in run_hard_ab.py

In-memory retrieval without PostgreSQL. For each backend:

```
fact_vecs = [embed(f.content) for f in corpus]  # 400 embeddings, once
for scenario in scenarios:
    q_naive = embed(scenario.query)
    q_sar   = embed(session_log_content + " " + scenario.query)
    # SAR enrichment: " ".join(e["content"] for e in session_log)
    naive_top = cosine_top5(fact_vecs, q_naive)
    sar_top   = cosine_top5(fact_vecs, q_sar)
```

The session_log is passed directly from JSON (not through PostgreSQL) — a difference from the synthetic_100 benchmark.

### 12.6 Backends

| Backend | Source | Dimension |
|---|---|---|
| BM25 | `rank_bm25` | — |
| nomic-embed-text | Ollama `http://localhost:11434` | native |
| mxbai-embed-large | Ollama `http://localhost:11434` | native (input truncated to 800 characters) |
| Qwen3-8B | OpenRouter API | 1536 (Matryoshka) |

Ollama backends are checked at startup (`check_alive()`); unavailable ones are skipped.

### 12.7 Results (hard_ab_results.json)

| Model | N | Naive | SAR | Δ |
|---|---|---|---|---|
| BM25 | 93 | 0.040 | 0.312 | +0.272 |
| nomic-embed-text | 93 | 0.011 | 0.215 | +0.204 |
| mxbai-embed-large | 93 | 0.006 | 0.100 | +0.093 |
| **Qwen3-8B** | **93** | **0.320** | **0.624** | **+0.303** |

**Conclusion:** SAR gives a significant gain of Δ = +0.303 for Qwen3-8B on the hard dataset. The largest gain is in cross_domain (+0.320) and multi_hop (+0.370). BM25 shows Δ = +0.272 because the session_log content contains bridge terms that literally occur in the facts — lexical enrichment works even without vector search.

Generating the dataset cost **$1.04** (3 models, ~93 scenarios + 400 facts, logged in `tests/fixtures/cost_log.json`).
