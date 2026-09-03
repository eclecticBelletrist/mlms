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

### 3. Tools = 9, not 7

CLAUDE.md says "7 tools", but there are actually 9: `revert_fact` and `export_project` were added. Hard-coded counts in the documentation are stale.

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

## Cut from scope

### 9. synthetic_100 is too easy for the SAR proof

Delta=0 on synthetic_100. Qwen3-8B copes without enrichment. Solution: synthetic_hard (93 scenarios, 400 facts, delta=+0.303).

### 10. No local inference for the fallback

`EMBEDDING_FALLBACK = "qwen3-embedding-0.6b"` — 404 on OpenRouter. Meant for Ollama/vLLM. Not implemented.

### 11. No MCP smoke test

9 tools were not checked through a real MCP client (Claude, MCPJam). Only directly from Python.

### 12. export_project does not filter skills by project_id

`SELECT id, name, content, domain, version, created_at FROM skills` — no `WHERE`. An export for a specific `project_id` returns all skills in the system, labelled with the same `project_id` in the top-level JSON — misleading. Root cause: in this version of the schema `skills` has no `project_id` column at all (see `alembic/versions/005_create_skills.py`) — scoping does not exist architecturally, not only in this query.

### 13. `get_timeline` cannot return more than `limit` events — the cursor is not wired up

`get_events()` in `storage/timeline.py` implements full keyset pagination `(time, id)`, covered by tests. But the MCP tool `tools/get_timeline.py` does not use it — it bypasses it with its own inline SQL (`ORDER BY time DESC LIMIT %s`), without a cursor and without `has_next`. In practice: if `timeline_events` has accumulated more events than `limit`, only the first `limit` by descending time are reachable through the MCP tool, and the rest cannot be reached by any call. Deliberately not wired up for now — it requires changing the tool's response model (`next_cursor` exposed), a risk under publication haste.

## CI

### 14. CI does not run the integration tests

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs only `pytest -m "not integration"` — lint (ruff), types (mypy), unit tests with an 80% coverage gate. `tests/integration/*` require `docker compose up` (Postgres with pgvector+TimescaleDB, Redis) — a non-standard image, not covered by the default GitHub Actions `services:` without a separate custom image. Deliberately not wired up for now — the integration tests are still run manually (`MLMS_INTEGRATION=1 pytest tests/integration -v` with a locally started `docker compose up`).

## TODO/FIXME in code

There are no direct `TODO`/`FIXME` comments in the code — the project follows the "no over-commenting" principle. But:

- `COSINE_CONFLICT_THRESHOLD = 0.96` — calibration starting point, needs independent calibration
- `_EXACT_ID_RE` in `get_facts.py` — regex that activates FTS, may need extending
- `search_memory.py` coverage 53% — covered by integration tests
