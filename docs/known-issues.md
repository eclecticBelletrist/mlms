# Known issues and debt — MLMS

## Deviations from targets

### 1. p99 session_context FAIL

`get_session_context()` p99=9.9ms vs target ≤5ms. p95=3.1ms is within the target.
**Cause:** a single outlier among 100 measurements, TCP overhead of Docker Desktop on Windows.
**Expectation:** on Linux with a Unix-socket Redis → < 1ms p99.
**Status:** does not block the release, likely cause identified, no fix required.

### 2. conflict_019 — calibration edge case

The only failing scenario in synthetic_100. Two facts with cosine < 0.96 — both stay `is_current=TRUE`.
**Cause:** the 0.96 threshold is a calibration starting point, not a final parameter.
**Needed:** independent threshold calibration for Qwen3-Embedding-8B.
**Additional data (measured on a separate branch that is not included in this repository):** on an independent labeled dataset of 76 pairs, the 0.96 threshold gives precision/recall/F1 = 0.000 — the detector never fired. The best threshold found there is 0.65-0.66 (F1=0.835). This diverges noticeably from the "only failing scenario out of 100" above — either `synthetic_100` is not sensitive enough to this problem, or the two datasets cover different conflict patterns. `main` has the same formula and the same 0.96 value, but there is neither its own run on the 76-pair dataset nor an analysis of this discrepancy — an open question, a fact neither confirmed nor refuted.

### 3. (FIXED 2026-09-16) Tool count in the docs

CLAUDE.md said "7 tools", but there were actually 9: `revert_fact` and `export_project` had been added. Fixed: the counts in the documentation are updated.

## Architectural limitations

### 4. Skills race condition

Two parallel `insert_skill` calls with the same name → two rows with the same `version`. Deliberate: the agent writes one at a time. Fix: advisory lock (FW).

### 5. (FIXED 2026-05-09) No HNSW on session_log.embedding

With `vector(4096)` neither HNSW nor IVFFlat could be built (pgvector limit of 2000 dim), so there were no indexes. Migration 006, after truncation to 1536, added HNSW on `facts.embedding` and `session_log.embedding`; details in `KEY_DECISIONS.md`, Decision 2.

### 6. session_log has no UNIQUE constraint

Hypertable + UNIQUE is a complicated construction in TimescaleDB. The current version allows duplicates.

### 7. project_id in session_log is nullable

The column is nullable. The code handles NULL correctly.

### 8. Benchmarks only on Windows + Docker Desktop

Production Linux behavior is not verified. p99 of session_context may be better on Linux.

### 9. memorize() — the Redis part is not atomic (dual-write + RMW race)

`src/mlms/tools/memorize.py`: in all four branches (`fact`/`event`/`skill`/`session_log`) the Postgres write completes before Redis is updated by a separate `_refresh_redis()` call — not one transaction, not an outbox, not a saga. In the `event`/`skill` branches there is one more Postgres insert (`insert_session_entry()`) between the main Postgres write and `_refresh_redis()` — so there it is `Postgres → Postgres → Redis`, not a plain two-step, one more partial-failure point. If `_refresh_redis()` fails after the Postgres part has committed, the fact/event/skill is already in the database, but the working memory in Redis does not reflect it; the client sees an error and may decide that the write did not go through (and retry, or continue with a desynchronized state).
**Plus, a separate class of bug — not a Postgres/Redis desync, but a race inside Redis itself:** `_refresh_redis()` is a non-atomic read-modify-write (`HGETALL` → mutation → `HSET`) over the same Redis hash `chat_id`. A concurrent `memorize()` for one `chat_id` from two devices can lose an update (lost update). A related non-atomicity in the same code is item 10 below (`hset`/`expire` inside `set_session_context`, which `_refresh_redis()` also calls).
**Status:** deliberately not fixed — working memory in Redis is treated as a best-effort cache, not a source of truth.

### 10. session_ctx — non-atomic hset/hgetall + expire pairs

`src/mlms/storage/session_ctx.py`, `get_session_context`/`set_session_context` (called, among others, from `_refresh_redis()` in item 9 above): a read is `hgetall`, then a separate `expire` call; a write is `hset`, then a separate `expire` call. Both are two independent round-trips to Redis, not `MULTI/EXEC`, not Lua.
The narrow window between the two calls is not a systemic guarantee but a rare failure exactly in that gap (process killed, network to Redis briefly unavailable): on a read, the TTL is not refreshed and the session expires earlier than real activity; on a write, the key stays in Redis without a TTL until the next successful write for the same `chat_id` (until then — a memory leak, Redis will not clean such a key up by itself).
**Status:** deliberately not fixed in this round.

### 11. Errors are plain Python exceptions, not MCPError

Throughout `src/mlms/tools/*.py` and `src/mlms/storage/*.py` — `KeyError`/`ValueError`/`RuntimeError` instead of `MCPError` from the MCP SDK. The SDK distinguishes protocol-level errors (`MCPError`, visible to the host) from execution-level ones (a plain exception, visible to the model as text in the tool response). For business errors ("`chat_id` and `semantic_query` cannot be used together") execution-level is appropriate — the model should see it and adjust the call. For infrastructure failures — for example `RuntimeError("Both embedding models are unavailable")` in `src/mlms/embedding.py:110`, when the service itself is unavailable rather than the call arguments being invalid — the model gets an unclear text error and may blindly repeat the call, instead of the host receiving a clear protocol signal.
**Status:** deliberately not fixed across the whole codebase in this round; a full migration is a task for the future.

## Cut from scope

### 12. synthetic_100 is too easy for the SAR proof

Delta=0 on synthetic_100. Qwen3-8B copes without enrichment. Solution: synthetic_hard (93 scenarios, 400 facts, delta=+0.303).

### 13. No local inference for the fallback

`EMBEDDING_FALLBACK = "qwen3-embedding-0.6b"` — 404 on OpenRouter. Meant for Ollama/vLLM. Not implemented. Its native dimension is 1024, below `EMBEDDING_DIM = 1536`, so it cannot be used with the current column (inserts fail with "expected 1536 dimensions") or mixed with 8B vectors until local serving and a dimension strategy exist.

### 14. No MCP smoke test

10 tools were not checked through a real MCP client (Claude, MCPJam). Only directly from Python.

### 15. export_project and get_skills are not scoped by project_id

`export_project`: `SELECT id, name, content, domain, version, created_at FROM skills` — no `WHERE`. An export for a specific `project_id` returns all skills in the system, labelled with the same `project_id` in the top-level JSON — misleading. `get_skills` (`src/mlms/tools/get_skills.py`) does not accept a `project_id` parameter at all — there is no way to select the skills of a specific project through the MCP interface. The root cause and the full architectural rationale are in `KEY_DECISIONS.md`, Decision 24 (`skills` has no `project_id` column in the schema, scoping does not exist architecturally). The `skill` layer in `search_memory` (`src/mlms/tools/search_memory.py`) also searches across all skills: `_search_skills` calls `get_skills(search=..., limit=...)` without a project.

### 16. `get_timeline` cannot return more than `limit` events — the cursor is not wired up

`get_events()` in `storage/timeline.py` implements full keyset pagination `(time, id)`, covered by tests. But the MCP tool `tools/get_timeline.py` does not use it — it bypasses it with its own inline SQL (`ORDER BY time DESC, id DESC LIMIT %s`), without a cursor and without `has_next`. In practice: if `timeline_events` has accumulated more events than `limit`, only the first `limit` by descending time are reachable through the MCP tool, and the rest cannot be reached by any call. Deliberately not wired up for now — it requires changing the tool's response model (`next_cursor` exposed), a risk under publication haste.

## CI

### 17. (FIXED 2026-09-19) CI does not run the integration tests

It was based on a wrong premise: `timescale/timescaledb-ha:pg16` and `redis:7-alpine` are
ordinary Docker images, `services:` in GitHub Actions accepts any image from a registry, no
custom image was needed. Fixed: `ci.yml` starts Postgres+Redis as
`services:` (credentials and images mirror `docker-compose.yml`), runs `alembic upgrade head`,
then `pytest -m unit` and `MLMS_INTEGRATION=1 pytest -m integration` — all 31 integration
tests are set up to run in CI on every push/PR to `main`, not only manually. The steps were reproduced
locally; the run on GitHub Actions itself was not checked (Decision 19).

### 18. (FIXED 2026-09-16) `pytest -m "not integration"` filtered nothing in CI

Until this date no test file set the `unit`/`integration`/`benchmark` markers
declared in `pyproject.toml`. This meant the CI step `pytest -m "not integration"`
(`.github/workflows/ci.yml`) did not exclude `tests/integration/*` at all — `testpaths =
["tests"]` includes both folders, and without markers the condition "not integration" is true for
any test. In practice CI collected all 31 integration tests together with the unit tests
(without `MLMS_INTEGRATION=1` they skip themselves, `tests/integration/conftest.py`), and the filter
by marker excluded nothing. This was aggravated by a separate finding (Decision 27 in KEY_DECISIONS.md): one of the transport tests
may hang on leaving `TestClient` (10 of 10 runs with Postgres reachable, 6 of 20 with nothing
listening), which with the 10-minute `timeout-minutes` in the workflow meant a real risk that CI,
on reaching that test, would hang until the runner timeout instead of failing within seconds.

Fixed: added `pytest_collection_modifyitems` in `tests/conftest.py`,
which sets `unit`/`integration` automatically from the test path. `pytest -m unit`
(the call CI uses) now really selects 244/275 (31 deselected) — verified with a live
collect-only run.

## Search

### 19. Full-text search is hardcoded to the `english` configuration

Two queries use PostgreSQL full-text search with the `english` text search configuration fixed in the SQL: `get_facts` (`src/mlms/tools/get_facts.py`, `to_tsvector('english', content) @@ plainto_tsquery('english', %s)`) and the event layer of `search_memory` (`src/mlms/tools/search_memory.py`, the same pair over `title || ' ' || content`). The configuration decides how text is split into words, which stop words are dropped and how word forms are reduced to a stem. For text that is not English, only the splitting and lower-casing apply.

Measured on PostgreSQL 16.15 (the image used by `docker-compose.yml`):

- Russian word forms are not matched to each other: a document containing `запросы` is not found by the query `запрос`; the `russian` configuration finds it. A document with `файл файлы файла` produces three separate lexemes.
- An exact Russian word, in any letter case, is found (`кофе` finds `Я люблю кофе`).
- Russian stop words are not dropped (`я` and `и` stay in the index and in the query). English stop words are dropped, and a query made only of them (`the`) is ignored with a notice, so it matches nothing.
- English stemming works (`run test` finds `running tests`), and identifiers such as `3f2a-9b1c` and mixed text such as `PostgreSQL` inside Russian prose are found.

Reach: this is the secondary path. Vector search is the main retrieval path and does not depend on the language of the text. In `get_facts` the full-text query runs only when the `about` text looks like an identifier or a version, as decided by `_EXACT_ID_RE`, and its results are appended after the vector results. In `search_memory` it is the only way events are searched (events have no embedding), so for non-English events the keyword match is by exact word form. Skills are searched with `ILIKE` and are not affected.

Not fixed: the fix is either a configurable language per project or a language-agnostic match, and both need a decision about the schema and the query. Until then, non-English content is reliably retrievable by vector search and by exact word form.

## TODO/FIXME in code

There are no direct `TODO`/`FIXME` comments in the code — the project follows the "no over-commenting" principle. But:

- `COSINE_CONFLICT_THRESHOLD = 0.96` — calibration starting point, needs independent calibration
- `_EXACT_ID_RE` in `get_facts.py` — regex that activates FTS, may need extending
- `search_memory.py` coverage 56% — covered by integration tests
