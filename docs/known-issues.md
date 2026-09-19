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

### 5. No HNSW on session_log.embedding

Migration 006 added HNSW on `facts.embedding` and `session_log.embedding`. But `KEY_DECISIONS.md` records that IVFFlat was planned first, then a sequential scan. Current state: HNSW exists on both tables.

### 6. session_log has no UNIQUE constraint

Hypertable + UNIQUE is a complicated construction in TimescaleDB. The current version allows duplicates.

### 7. project_id in session_log is nullable

Added in migration 007. Rows that existed before the migration have NULL. The code handles this correctly.

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

`EMBEDDING_FALLBACK = "qwen3-embedding-0.6b"` — 404 on OpenRouter. Meant for Ollama/vLLM. Not implemented.

### 14. No MCP smoke test

10 tools were not checked through a real MCP client (Claude, MCPJam). Only directly from Python.

### 15. export_project and get_skills are not scoped by project_id

`export_project`: `SELECT id, name, content, domain, version, created_at FROM skills` — no `WHERE`. An export for a specific `project_id` returns all skills in the system, labelled with the same `project_id` in the top-level JSON — misleading. `get_skills` (`src/mlms/tools/get_skills.py`) does not accept a `project_id` parameter at all — there is no way to select the skills of a specific project through the MCP interface. The root cause and the full architectural rationale are in `KEY_DECISIONS.md`, Decision 24 (`skills` has no `project_id` column in the schema, scoping does not exist architecturally).

### 16. `get_timeline` cannot return more than `limit` events — the cursor is not wired up

`get_events()` in `storage/timeline.py` implements full keyset pagination `(time, id)`, covered by tests. But the MCP tool `tools/get_timeline.py` does not use it — it bypasses it with its own inline SQL (`ORDER BY time DESC, id DESC LIMIT %s`), without a cursor and without `has_next`. In practice: if `timeline_events` has accumulated more events than `limit`, only the first `limit` by descending time are reachable through the MCP tool, and the rest cannot be reached by any call. Deliberately not wired up for now — it requires changing the tool's response model (`next_cursor` exposed), a risk under publication haste.

## CI

### 18. (FIXED 2026-09-16) `pytest -m "not integration"` filtered nothing in CI

Until this date no test file set the `unit`/`integration`/`benchmark` markers
declared in `pyproject.toml`. This meant the CI step `pytest -m "not integration"`
(`.github/workflows/ci.yml`) did not exclude `tests/integration/*` at all — `testpaths =
["tests"]` includes both folders, and without markers the condition "not integration" is true for
any test. In practice CI tried to run all 31 integration tests (which
need a live Postgres+Redis) without a running `docker compose` — a scenario aggravated by
a separate finding (Decision 27 in KEY_DECISIONS.md): one of the transport tests
may hang on leaving `TestClient` (10 of 10 runs with Postgres reachable, 6 of 20 with nothing
listening), which with the 10-minute `timeout-minutes` in the workflow meant a real risk that CI,
on reaching that test, would hang until the runner timeout instead of failing within seconds.

Fixed: added `pytest_collection_modifyitems` in `tests/conftest.py`,
which sets `unit`/`integration` automatically from the test path. `pytest -m unit`
(the call CI uses) now really selects 244/275 (31 deselected) — verified with a live
collect-only run.

### 17. (FIXED 2026-09-19) CI does not run the integration tests

It was based on a wrong premise: `timescale/timescaledb-ha:pg16` and `redis:7-alpine` are
ordinary Docker images, `services:` in GitHub Actions accepts any image from a registry, no
custom image was needed. Fixed: `ci.yml` starts Postgres+Redis as
`services:` (credentials and images mirror `docker-compose.yml`), runs `alembic upgrade head`,
then `pytest -m unit` and `MLMS_INTEGRATION=1 pytest -m integration` — all 31 integration
tests now really run in CI on every push/PR to `main`, not only manually.

## TODO/FIXME in code

There are no direct `TODO`/`FIXME` comments in the code — the project follows the "no over-commenting" principle. But:

- `COSINE_CONFLICT_THRESHOLD = 0.96` — calibration starting point, needs independent calibration
- `_EXACT_ID_RE` in `get_facts.py` — regex that activates FTS, may need extending
- `search_memory.py` coverage 53% — covered by integration tests
