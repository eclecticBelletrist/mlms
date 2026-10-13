# MLMS retrieval benchmark

This document describes how the retrieval quality and latency of MLMS were measured: the two
synthetic datasets, how each was built and why the second one exists, the results, and what
those results do and do not show. All numbers below come from files in this repository
(`scripts/*.json`, `tests/fixtures/*.json`) unless a section says otherwise.

In short: on `synthetic_100` Recall@5 is 0.990, but that dataset cannot fail: almost
every scenario has no more facts than the five that are returned. On `synthetic_hard` the
Session-Augmented RAG (SAR) experiment raises Recall@5 for Qwen3-Embedding-8B from 0.320 to 0.624,
but the generator was told to put the words of the relevant facts into the session log, and
scenarios where SAR would not help were filtered out. Neither number is evidence of retrieval
quality on real data. Section 6 explains both in detail, and section 8 lists what the next
version of the benchmark changes.

## 1. What is measured

The benchmark covers only the semantic layer: facts stored in pgvector and read with
`get_facts`. The other reads are deterministic (Redis hash, SQL time range, latest version per
skill) or, for cross-session `session_log` search, not measured here.

| Target | Metric | Value |
|---|---|---|
| FR-1 | Recall@5 on `synthetic_100` | ≥ 0.80 |
| NFR-1 | `get_facts` p99, embedding served from cache | < 200 ms |
| | `get_session_context` p99 | ≤ 5 ms |

```
Recall@5(scenario) = |top-5 returned ∩ relevant_ids| / |relevant_ids|
Recall@5(dataset)  = mean over scenarios
```

The target was first written as Precision@5 ≥ 0.80. With one or two relevant facts per scenario
the best possible Precision@5 is 0.20 or 0.40, so the target could never be met, and it was
changed to Recall@5. With one relevant fact Recall@5 equals Hit@5 (found or not found).

SAR (Session-Augmented RAG) prepends the text of the current chat's session log to the query
before embedding it. It exists only in the benchmark runners in `scripts/`. The served MCP tools
embed the bare query.

## 2. synthetic_100: the first dataset

### 2.1 Design

100 scenarios, 25 in each of four categories. Each scenario is an isolated project with its own
facts, a query, and the ids of the relevant facts.

| Category | What it is meant to test |
|---|---|
| `lexical` | The query uses the same words as the fact. Baseline. |
| `semantic_drift` | The query describes the same thing in other words. Run twice, without and with SAR. |
| `conflict` | An outdated and a current version of the same fact; the query asks for the current state. |
| `cross_session` | Facts written from one chat, queried from another chat of the same project. |

An earlier draft had a `temporal` category served by `get_timeline`. It was replaced by
`cross_session` because a time-range query is plain SQL with no ranking to measure, and it is the
episodic layer, not the semantic one.

### 2.2 Generation

`scripts/seed_test_data.py` builds the dataset in 20 batches of 5 scenarios through the OpenRouter
API. The decisions in the pipeline, with the work each one rests on:

- Generator and verifier from different model families. Three models rotate:
  `claude-sonnet-4.6` is checked by `gpt-5.2`, `gpt-5.2` by `grok-4.20`, `grok-4.20` by
  `claude-sonnet-4.6`. A judge from the generator's own family tends to rate its output higher
  (Li et al., 2025, "Preference Leakage"). Qwen3, the embedding model under test, takes no part in
  generation or labeling, and Recall@5 is computed from ids, with no LLM in the metric.
- No persona in the system prompt. Personas do not improve accuracy on objective tasks
  (Zheng et al., Findings of EMNLP 2024).
- Reasoning before structure. The output schema starts with a free-text `rationale` field, so
  the model reasons before it fills the structured fields. Tam et al. (EMNLP 2024, "Let Me Speak
  Freely?") report that strict format constraints can hurt reasoning. The output itself is still
  forced through `tool_choice`, so this is a mitigation inside constrained output, not a separate
  free-text step.
- Structured output through `tool_choice` for all three providers, instead of asking for JSON
  in the prompt. The parse failure rate of this pipeline was not recorded.
- Diversity controls. Every call receives the list of domains already used and has to vary
  query style and fact length, against repeated templates across calls (arXiv:2505.18949).
- Lexical distance self-check for `semantic_drift`: before filling the fields the generator
  lists the content words of the query and of the relevant fact, and the overlap must be at most
  one word. Without it the naive query already found the fact and the A/B showed no difference.
- Verification. The verifier reads each scenario and returns `ok` or `fix_needed` with a
  corrected list of relevant ids; on `fix_needed` the scenario's labels are overwritten and
  `meta.auto_fixed` is set.

The prompts are in the script, in Russian, as they were sent.

### 2.3 Results

`scripts/run_benchmark.py`, Windows 11, Docker Desktop, Qwen3-Embedding-8B through OpenRouter.
The result file of this run was not kept.

| Category | Recall@5 | N |
|---|---|---|
| lexical | 1.000 | 25 |
| semantic_drift | 1.000 | 25 |
| conflict | 0.960 | 25 |
| cross_session | 1.000 | 25 |
| overall | 0.990 | 100 |

`semantic_drift` without and with SAR (`scripts/run_ab_benchmark.py`): 1.000 and 1.000, delta 0.

### 2.4 Why these results say little

The dataset cannot fail. A project holds 3 to 6 facts, 97 of the 100 scenarios
hold five or fewer, and `get_facts` returns the top 5 with no similarity cut-off. In those
scenarios every current fact is returned whatever the ranking, so Recall@5 is 1.0 for any
embedding model, with or without SAR. This is why three categories score exactly 1.000 and why the
SAR delta is 0, and it is why the second dataset was built.

## 3. synthetic_hard: the second dataset

### 3.1 Design

The fix for the saturation was a shared corpus much larger than k, and queries that word match
cannot solve.

- Corpus: 400 facts, 40 in each of 10 domains (kubernetes, postgresql, redis, kafka,
  fastapi, react, docker, elasticsearch, nginx, pytorch), shared by all scenarios.
- Scenarios: 93. Four types: `lexical_drift` (30, user language against technical terms),
  `semantic_drift` (29, symptom against cause), `cross_domain` (25, relevant facts from two
  domains), `multi_hop` (9, the answer needs a chain of facts). The plan was 100 (30/30/25/15);
  scenarios that failed three generation attempts were dropped.
- Relevant facts: 2 to 5 per scenario, 3 in 73 of them, so Recall@5 here is a real fraction,
  not a hit or miss.
- Session log: 2 to 4 `{role, content}` turns per scenario.

### 3.2 Generation

`scripts/generate_hard_dataset.py`: `claude-sonnet-4.6` writes the corpus and the scenarios. A
scenario is regenerated, up to three times, when:

- the word overlap between the query and the relevant facts (`lexical_gap_score`, computed by the
  script) is above 2, or above 1 for `multi_hop`;
- `gpt-5.2` judges it too easy, that is, solvable without the session log;
- `grok-4.20` judges that SAR could not solve it.

After generation the script checks BM25 on the raw queries against a target range of 0.20 to 0.55
(share of scenarios with at least one relevant fact in the top 5). It got 0.118, printed a
warning and kept the dataset. Generation cost $1.04 (`tests/fixtures/cost_log.json`).

### 3.3 Results

`scripts/run_hard_ab.py` (`scripts/hard_ab_results.json`): in-memory cosine search over the 400
facts, k=5, without Postgres. The SAR query is the session log text followed by the query.

| Backend | Naive | SAR | Delta |
|---|---|---|---|
| BM25 (`rank_bm25`) | 0.040 | 0.312 | +0.272 |
| nomic-embed-text (Ollama) | 0.011 | 0.215 | +0.204 |
| mxbai-embed-large (Ollama, input cut to 800 characters) | 0.006 | 0.100 | +0.093 |
| Qwen3-Embedding-8B, 1536 dimensions | 0.320 | 0.624 | +0.303 |

Qwen3-Embedding-8B by type:

| Type | N | Naive | SAR | Delta |
|---|---|---|---|---|
| lexical_drift | 30 | 0.362 | 0.669 | +0.307 |
| semantic_drift | 29 | 0.402 | 0.667 | +0.264 |
| cross_domain | 25 | 0.123 | 0.443 | +0.320 |
| multi_hop | 9 | 0.463 | 0.833 | +0.370 |

BM25 shows 0.040 here and 0.118 in section 3.2 because the two scripts compute different things:
the runner reports Recall@5, the generator's check counts a scenario as passed when any relevant
fact is in the top 5.

## 4. Latency

`scripts/measure_latency.py`, n=100 per operation, Windows 11 with Docker Desktop
(`scripts/latency_results.json`). Percentiles are read from the sorted samples by index without
interpolation, so with n=100 the p99 is the slowest sample.

| Operation | p50 ms | p95 ms | p99 ms | Target |
|---|---|---|---|---|
| `get_session_context` | 1.6 | 3.1 | 9.9 | ≤ 5 ms, missed |
| `get_facts`, cached query embedding | 4.7 | 7.7 | 16.0 | < 200 ms, met |
| embed through the API + `upsert_fact` | 1158 | 4130 | 9237 | none |

The `get_facts` run repeats one query 100 times after warming the embedding cache, over 10 facts.
It measures the database and cache part of a read, not a read with a new query. The third row
calls `embed` and `upsert_fact` directly, not the full `memorize` tool. The `get_session_context`
miss is one slow sample out of 100; it was not measured on Linux.

## 5. Test isolation

Each scenario gets a new `project_id` (`uuid4`), and its facts, session log rows and project are
deleted in a `finally` block, so scenarios never see each other's data. For `cross_session` the
facts are filtered by project, not by chat, which is the behavior the category is meant to
exercise.

## 6. What the numbers do and do not show

FR-1 on `synthetic_100` is not evidence. As section 2.4 explains, the result is 1.0 for almost
any retriever. On top of that, `run_benchmark.py` scores each `semantic_drift` scenario as the
better of the naive and the SAR run (`max(naive, sar)`), and a scenario that raises an error is
scored 0 instead of being reported as an error.

The SAR gain on `synthetic_hard` is built into the dataset. The generator prompt asks for a
session log that contains exactly the technical words of the relevant facts and a query that
contains none of them, and the second verifier keeps only scenarios that SAR can solve. Measured on
the fixture: in 72 of the 93 scenarios a term from `sar_bridge_terms` appears both in the session
log and in a relevant fact, and in 85 of 93 the session log shares three or more words longer than
four letters with the relevant facts. Prepending the session log therefore adds the answer's
vocabulary to the query, which is also why BM25 gains +0.272 without any embedding. What the +0.303
does show: when the chat history already contains the terms of the stored facts, adding it to the
query lets Qwen3-Embedding-8B find about twice as many relevant facts in a 400-fact corpus. It does
not show how often real chat histories contain those terms. There is one run, no confidence
interval, and `multi_hop` has 9 scenarios.

The `conflict` category does not test version handling. The verifier changed the labels of
14 of the 25 conflict scenarios, and each of those 14 now has more than one relevant fact: the
outdated version is labeled relevant next to the current one (for example `conflict_005`,
`conflict_021`). Recall@5 counts either version as a hit. The `is_current` behavior is covered by
the integration tests instead.

The single miss in `conflict` is probably the opposite of a missed conflict. `conflict_019` has five facts.
If both versions had stayed current, all five would be in the top 5 and the scenario would score
1.0. A miss means the current fact was not returned, and the likely reason is that the threshold
did fire: the facts are inserted in listed order, the current version first, so the outdated one
arrived second and retired the current one. The similarity values of that run were not saved, so this is not
confirmed. Separately, on a labeled set of 76 pairs that is not in this repository the 0.96
threshold detected no conflicts at all (`docs/known-issues.md`, item 2); the threshold is
uncalibrated.

`cross_session` does not record the source chat. Facts are stored per project and the
runner does not keep the source `chat_id`, so the category checks that facts are visible project
wide, which `lexical` already shows.

A quarter of the `synthetic_100` labels were written by the verifier. 25 of 100 scenarios are
`auto_fixed` (conflict 14, cross_session 9, semantic_drift 2), and no labels were checked by a
person.

All runs are single runs on Windows with Docker Desktop.

## 7. Reproducing

All runners need Postgres and Redis from `docker-compose.yml` and an embedding key
(`EMBEDDING_API_KEY`). Generation needs `OPENROUTER_API_KEY` and the model ids in `MODEL_CLAUDE`,
`MODEL_GPT4O` (holds the GPT model id; the name is historical) and `MODEL_GROK`. OpenRouter model
ids change over time, so regenerated datasets will differ from the committed ones.

```bash
python scripts/seed_test_data.py                         # regenerate synthetic_100.json
python scripts/generate_hard_dataset.py                  # regenerate synthetic_hard.json
MLMS_INTEGRATION=1 python scripts/run_benchmark.py       # section 2.3
MLMS_INTEGRATION=1 python scripts/run_ab_benchmark.py    # section 2.3, A/B
MLMS_INTEGRATION=1 python scripts/run_hard_ab.py         # section 3.3 (Ollama backends optional)
MLMS_INTEGRATION=1 python scripts/measure_latency.py     # section 4
```

`scripts/run_multimodel_ab.py` ran the `synthetic_100` A/B with several backends. Its results are
not used: every backend ties for the reason in section 2.4, and failed calls were recorded as 0
(`EMBEDDING_FALLBACK`, `qwen3-embedding-0.6b`, is not served by OpenRouter).

## 8. Next version

- A corpus much larger than k for every category, and Recall@k for several k plus MRR, with a
  random ranker as a control that must fail the target.
- A session log written without seeing the relevant facts (by another model, or from real chat
  transcripts), and results reported on the unfiltered set, with the number of rejected scenarios.
- A sample of labels checked by hand, and the verifier's original verdicts kept.
- A conflict metric that penalizes an outdated version in the results, and the `is_current` state
  logged after the facts are inserted.
- Errors reported separately from misses, no `max` over conditions, and bootstrap confidence
  intervals for every delta.
- Latency on Linux, with new queries (cache misses) and a corpus of at least 10,000 facts.
