# Known issues and debt

## Deviations from targets

### 1. p99 session_context FAIL

`get_session_context()` p99=9.9ms vs target ≤5ms. p95=3.1ms is within the target.
**Likely cause (not verified):** a single outlier among 100 measurements, TCP overhead of Docker Desktop on Windows. With 100 samples the script's p99 is the slowest one (it picks a sample by index, without interpolation).
**Expectation:** on Linux with a Unix-socket Redis, p99 under 1ms.
**Status:** does not block the release; not measured on Linux.

### 2. The conflict threshold is uncalibrated

The 0.96 threshold is a calibration starting point. On an independent labeled dataset of 76 pairs, which is not included in this repository, it gives precision/recall/F1 = 0.000: the detector never fired. That run used the same formula and the same 0.96 value as this code; the best threshold on that dataset is 0.65-0.66 (F1=0.835).
The only miss in `synthetic_100`, `conflict_019`, does not contradict this and does not measure it: the scenario has five facts and `limit=5`, so the miss means the current fact was retired, most likely by the outdated version inserted after it. The similarity values of that run were not saved. The `conflict` category of `synthetic_100` cannot measure the threshold (`docs/mlms_benchmark_design.md`, section 6).
**Needed:** independent threshold calibration for Qwen3-Embedding-8B.

## Architectural limitations

### 3. Skills race condition

Two parallel `insert_skill` calls with the same name read the same maximum version. The `UNIQUE (name, version)` constraint (migration 001) rejects the second insert, so one writer fails with a unique-violation error and no duplicate version is stored. Deliberate: the agent writes one at a time. A possible fix is an advisory lock; it is not implemented.

### 4. session_log does not prevent duplicate entries

The primary key is `(id, time)` with a generated `id`, so writing the same entry twice stores it twice. A unique constraint on a TimescaleDB hypertable has to include the time column, so no constraint over the entry content was added.

### 5. project_id in session_log is nullable

The column is nullable. The code handles NULL correctly.

### 6. Benchmarks only on Windows + Docker Desktop

Production Linux behavior is not verified. p99 of session_context may be better on Linux.

### 7. memorize(): the Redis part is not atomic (dual-write + RMW race)

`src/mlms/tools/memorize.py`: in all four branches (`fact`/`event`/`skill`/`session_log`) the Postgres write completes before Redis is updated by a separate `_refresh_redis()` call. There is no single transaction, outbox or saga. In the `event`/`skill` branches there is one more Postgres insert (`insert_session_entry()`) between the main Postgres write and `_refresh_redis()`, so there it is `Postgres → Postgres → Redis`, one more partial-failure point. If `_refresh_redis()` fails after the Postgres part has committed, the fact/event/skill is already in the database, but the working memory in Redis does not reflect it; the client sees an error and may decide that the write did not go through (and retry, or continue with a desynchronized state).
A second problem is a race inside Redis itself, separate from the Postgres/Redis desync: `_refresh_redis()` is a non-atomic read-modify-write (`HGETALL` → mutation → `HSET`) over the same Redis hash `chat_id`. A concurrent `memorize()` for one `chat_id` from two devices can lose an update. A related non-atomicity in the same code is item 8 below (`hset`/`expire` inside `set_session_context`, which `_refresh_redis()` also calls).
**Status:** deliberately not fixed. Working memory in Redis is treated as a best-effort cache, not a source of truth.

### 8. session_ctx: non-atomic hset/hgetall + expire pairs

`src/mlms/storage/session_ctx.py`, `get_session_context`/`set_session_context` (called, among others, from `_refresh_redis()` in item 7 above): a read is `hgetall`, then a separate `expire` call; a write is `hset`, then a separate `expire` call. Each pair is two independent round-trips to Redis, without `MULTI/EXEC` or Lua.
A failure exactly between the two calls (process killed, network to Redis briefly unavailable) is rare. When it happens, on a read the TTL is not refreshed and the session expires earlier than real activity; on a write the key stays in Redis without a TTL until the next successful write for the same `chat_id`, and until then it leaks memory, because Redis will not clean such a key up by itself.
**Status:** deliberately not fixed in this round.

### 9. Errors are plain Python exceptions, not MCPError

`src/mlms/tools/*.py` and `src/mlms/storage/*.py` raise `KeyError`/`ValueError`/`RuntimeError` instead of `MCPError` from the MCP SDK. The SDK distinguishes protocol-level errors (`MCPError`, visible to the host) from execution-level ones (a plain exception, visible to the model as text in the tool response). For business errors ("`chat_id` and `semantic_query` cannot be used together") execution-level is appropriate, because the model should see the error and adjust the call. For infrastructure failures, for example `RuntimeError("Both embedding models are unavailable")` in `src/mlms/embedding.py`, which is raised when the service itself is unavailable rather than the call arguments being invalid, the model gets an unclear text error and may blindly repeat the call, instead of the host receiving a clear protocol signal.
**Status:** not fixed across the whole codebase in this round; migrating all of it is future work.

## Cut from scope

### 10. synthetic_100 is too easy for the SAR comparison

SAR (Session-Augmented RAG) shows delta=0 on synthetic_100, because nearly every scenario has at most five facts and k=5, so any retriever returns all of them. On synthetic_hard (93 scenarios, 400 facts) the delta is +0.303, but the session logs were generated to contain the words of the relevant facts and scenarios that SAR could not solve were filtered out. Details and the plan for an unbiased set are in `docs/mlms_benchmark_design.md`, sections 6 and 8.

### 11. No local inference for the fallback

`EMBEDDING_FALLBACK = "qwen3-embedding-0.6b"` returns 404 on OpenRouter. It is meant for Ollama/vLLM, which is not implemented. Its native dimension is 1024, below `EMBEDDING_DIM = 1536`, so it cannot be used with the current column (inserts fail with "expected 1536 dimensions") or mixed with 8B vectors until local serving and a dimension strategy exist.

### 12. No automated MCP smoke test

No test in CI starts the server and calls the tools through an MCP client; the tools are tested directly from Python. The server has been driven through the MCP SDK's stdio client (tool listing and the calls of a store, change and recall scenario) and through Claude Code 2.1.310 in print mode with Haiku and Sonnet, on a scenario of four chats (store, change, recall, revert; `KEY_DECISIONS.md`, Decision 30) and on the two prompts of the README walkthrough. Those runs are manual and were not repeated in other clients, such as Claude Desktop or MCPJam.

### 13. export_project and get_skills are not scoped by project_id

`export_project`: `SELECT id, name, content, domain, version, created_at FROM skills` has no `WHERE`. An export for a specific `project_id` returns all skills in the system (in the `skills_global` section of the export), next to the same top-level `project_id`, which is misleading. `get_skills` (`src/mlms/tools/get_skills.py`) does not accept a `project_id` parameter at all, so there is no way to select the skills of a specific project through the MCP interface. The root cause and the full architectural rationale are in `KEY_DECISIONS.md`, Decision 24 (`skills` has no `project_id` column in the schema, so scoping does not exist architecturally). The `skill` layer in `search_memory` (`src/mlms/tools/search_memory.py`) also searches across all skills: `_search_skills` calls `get_skills(search=..., limit=...)` without a project.

### 14. `get_timeline` has no cursor for events beyond `limit`

`get_events()` in `storage/timeline.py` implements full keyset pagination `(time, id)`, covered by tests. But the MCP tool `tools/get_timeline.py` does not use it. It has its own inline SQL (`ORDER BY time DESC, id DESC LIMIT %s`), without a cursor and without `has_next`. In practice: if `timeline_events` has accumulated more events than `limit`, one call returns only the first `limit` by descending time. Older rows are still reachable with a larger `limit` or by moving the `end` of `time_range` before the oldest event returned; what is missing is a bounded cursor that pages through them. Not wired up because it requires changing the tool's response model (exposing `next_cursor`).

## Search

### 15. Full-text search is hardcoded to the `english` configuration

Two queries use PostgreSQL full-text search with the `english` text search configuration fixed in the SQL: `get_facts` (`src/mlms/tools/get_facts.py`, `to_tsvector('english', content) @@ plainto_tsquery('english', %s)`) and the event layer of `search_memory` (`src/mlms/tools/search_memory.py`, the same pair over `title || ' ' || content`). The configuration decides how text is split into words, which stop words are dropped and how word forms are reduced to a stem. For text that is not English, only the splitting and lower-casing apply.

Measured on PostgreSQL 16.15 (the image used by `docker-compose.yml`):

- Russian word forms are not matched to each other: a document containing `запросы` is not found by the query `запрос`; the `russian` configuration finds it. A document with `файл файлы файла` produces three separate lexemes.
- An exact Russian word, in any letter case, is found (`кофе` finds `Я люблю кофе`).
- Russian stop words are not dropped (`я` and `и` stay in the index and in the query). English stop words are dropped, and a query made only of them (`the`) is ignored with a notice, so it matches nothing.
- English stemming works (`run test` finds `running tests`), and identifiers such as `3f2a-9b1c` and mixed text such as `PostgreSQL` inside Russian prose are found.

Reach: this is the secondary path. Vector search is the main retrieval path and does not depend on the language of the text. In `get_facts` the full-text query runs only when the `about` text looks like an identifier or a version, as decided by `_EXACT_ID_RE`; its hits are added to the vector results and displace the lowest-ranked vector rows when the page is full. In `search_memory` it is the only way events are searched (events have no embedding), so for non-English events the keyword match is by exact word form. Skills are searched with `ILIKE` and are not affected.

Not fixed: the fix is either a configurable language per project or a language-agnostic match, and both need a decision about the schema and the query. Until then, the path that serves each layer differs: facts are found by vector search plus, for identifier-like queries only, the keyword match by exact word form; events are found by the keyword match alone, by exact word form for non-English text; skills are found by substring match.

### 16. Other accepted gaps

Smaller gaps, accepted for now. A write that has already committed to Postgres can still report failure when the Redis refresh after it fails, and a retry may then store the fact twice (entry 7). `revert_fact` running at the same moment as a write to the same fact can leave two current versions. Redis working-memory updates are not atomic (entries 7 and 8). A filtered vector search can return fewer rows than `limit`, because the project and `is_current` filters apply after the HNSW index scan (Decision 17 describes the iterative scan that reduces this). Embedding retries cover rate limits and transport errors, while the fallback to the second model triggers on timeouts, HTTP errors and connect errors, so the two sets differ. A `Retry-After` header is obeyed with no upper bound, and a malformed embedding response is cached before it is validated.

## Deployment

### 17. One connection pool per MCP session

The Postgres pool is opened in the SDK lifespan, which runs once per client session, not once per process (`KEY_DECISIONS.md`, Decision 3). Over HTTP, a session the client never closes with `DELETE` keeps its pool and at least one connection open until the SDK closes it after 30 minutes without requests. The SDK allows up to 10,000 open sessions and Postgres allows 100 connections by default, so many clients that drop sessions without closing them can exhaust `max_connections` within that window. A fix is one pool per process, shared by all sessions; it is not implemented.

### 18. `MCP_HOST=::1` passes the secret check but not the Host allowlist

`::1` is treated as loopback by the shared-secret check, so the secret is optional on that bind. `_LOOPBACK_HOSTS` in `src/mlms/server.py` lists only `localhost` and `127.0.0.1`, so a request with `Host: [::1]` gets `421`. Use `127.0.0.1`, or add `[::1]` and `[::1]:*` to the list.

## Open points in code

The code has no `TODO`/`FIXME` comments. The open points are:

- `COSINE_CONFLICT_THRESHOLD = 0.96`: calibration starting point, needs independent calibration
- `_EXACT_ID_RE` in `get_facts.py`: regex that activates FTS, may need extending
