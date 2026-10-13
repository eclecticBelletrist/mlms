# Key architectural decisions

Each entry records what was decided, why, and which alternatives were rejected.

Dates are the days the decisions were made. Commit history before 2026-08 was reconstructed, so the early dates are approximate. Entry numbers are stable identifiers, not chronological order; numbers that are missing belong to entries that were merged into others or removed.

| # | Decision |
|---|---|
| 29 | Postgres (pgvector, TimescaleDB) for durable memory, Redis for working memory |
| 1 | Embeddings truncated to 1536 dimensions |
| 2 | HNSW index on the embedding columns |
| 3 | Connection pool instead of one shared connection |
| 4 | DNS rebinding protection, loopback-only allowlist |
| 5 | One `timescaledb-ha` image for Postgres |
| 6 | Conflict detection by cosine distance in a SERIALIZABLE transaction |
| 9 | `MCP_TRANSPORT` switch between stdio and HTTP |
| 10 | `revert_fact` follows the `replaces_id` chain |
| 14 | Keyset pagination on `(time, id)` |
| 15 | `memorize` content limit: hard 10,000 characters, soft 500 |
| 16 | MCP tool annotations |
| 17 | Filters in SQL before `LIMIT`, HNSW iterative scan |
| 18 | `psycopg[binary]`, no `types-redis` stub |
| 19 | CI: lint, types, unit and integration tests, image build |
| 20 | Container: install from `uv.lock`, HTTP transport, migrations on start |
| 20.5 | Postgres credentials from `.env`, Redis unauthenticated |
| 21 | Backup and restore with custom-format `pg_dump`, no `-j` |
| 22 | Separate Origin allowlist with schemes |
| 23 | Shared-secret gate for the HTTP transports |
| 24 | `skills` are global |
| 25 | Plain `CREATE INDEX` for the HNSW indexes |
| 26 | No SERIALIZABLE `SET` on a connection already in a transaction |
| 27 | Global `pytest-timeout` against a shutdown hang |
| 28 | Migrations squashed into one initial schema |

---

## Decision 29: Postgres with pgvector and TimescaleDB for durable memory, Redis for working memory

**Decided:** 2026-05-08

**Decision:** facts, timeline events, the session log and skills live in one PostgreSQL database. Vectors are stored in it with pgvector, and `timeline_events` and `session_log` are TimescaleDB hypertables. The working memory of a chat (`session:{chat_id}`) is a Redis hash with a 48-hour TTL that is renewed on every access.

**Rationale:**
- Conflict detection (Decision 6) reads the nearest current fact by vector and then retires it and inserts the new version. With vectors and rows in one database this is one transaction; with a separate vector store the two writes could diverge.
- One database means one backup and one restore procedure (Decision 21) and one place where a project's data can be exported.
- Timeline and session-log reads are time-range queries over append-only rows, the workload hypertables are partitioned for. At the current data volume the gain over a plain table has not been measured.
- Working memory is small, per chat and short-lived. A Redis hash gives sub-millisecond reads (the target for `get_session_context` is 5 ms) and expiry without a cleanup job. It is treated as a best-effort cache: the durable record of what happened in a chat is `session_log` in Postgres.

**Alternatives:**
- A dedicated vector database next to Postgres: a second service to run and back up, and no transaction across the vector lookup and the version write.
- Plain Postgres tables instead of hypertables: workable at small volume; TimescaleDB was chosen for the time-range workload, and its image also ships pgvector (Decision 5).
- Working memory in a Postgres table: needs its own expiry job and adds a database round-trip to every read.

**Affected files:** `docker-compose.yml`, `alembic/versions/001_initial_schema.py`, `src/mlms/storage/`

**See also:** Decision 5, Decision 6, Decision 21.

---

## Decision 1: EMBEDDING_DIM = 1536 (Matryoshka truncation)

**Decided:** 2026-05-09

**Last updated:** 2026-10-09 (migrations squashed into one initial schema, Decision 28)

**Decision:** `EMBEDDING_DIM = 1536` instead of the native 4096 of Qwen3-Embedding-8B.

**Rationale:** pgvector (0.8.7 in the pinned image, Decision 5) limits HNSW and IVFFlat indexes on `vector` to 2000 dimensions, and `halfvec` to 4000 ([pgvector README](https://github.com/pgvector/pgvector)). Neither fits 4096. Qwen3-Embedding-8B supports Matryoshka Representation Learning with output dimensions from 32 to 4096 ([model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B); technique: [Kusupati et al., 2022](https://arxiv.org/abs/2205.13147)), so its vectors can be truncated to a prefix with a limited loss of quality. Truncating to 1536 brings the column under the 2000-dimension limit and makes an HNSW index possible (see Decision 2). The API request asks for 1536 dimensions, and `embedding.py` truncates any longer vector and L2-normalizes it. The recall loss from truncation has not been measured in this project. The choice is expensive to undo: the full vectors are not stored, so changing the dimension means re-embedding every fact.

**Alternatives:**
- Sequential scan with no index on `vector(4096)`: exact (recall 100%), but query cost grows linearly with table size.
- `halfvec(4096)`: the 4000-dimension limit is below 4096.
- Expression index on a 1536-dimension prefix (`subvector(embedding, 1, 1536)`, documented in the pgvector README for models that support it): a valid option, not chosen. It keeps the full 4096 dimensions in storage (16392 bytes per row versus 6152 at 1536, from the README formula `4 * dimensions + 8`), and every query has to use the same expression to hit the index.

**Affected files:** `alembic/versions/001_initial_schema.py` (the `vector(1536)` column type and the HNSW indexes), `src/mlms/config.py`, `src/mlms/embedding.py`

**See also:** Decision 2.

---

## Decision 2: HNSW index on embedding columns

**Decided:** 2026-05-09

**Last updated:** 2026-10-09 (migrations squashed into one initial schema, Decision 28)

**Decision:** HNSW index on `facts.embedding` and `session_log.embedding`.

**Rationale:** without an ANN index, similarity search is a sequential scan, so its cost grows linearly with table size. At the native 4096 dimensions neither HNSW nor IVFFlat can be built (both are limited to 2000), so truncating to 1536 (Decision 1) is what makes the index possible.

**Alternatives:**
- Sequential scan: exact, but linear cost.
- IVFFlat: same 2000-dimension limit at 4096.

**Trade-offs:** HNSW search is approximately logarithmic in table size, at the price of higher memory use and a slower index build. At the expected volume (up to about 100K rows, an estimate rather than a measurement) this is acceptable.

**Affected files:** `alembic/versions/001_initial_schema.py`

**See also:** Decision 1, Decision 17, Decision 25.

---

## Decision 3: Connection pool instead of a single shared connection

**Decided:** 2026-08-24

**Last updated:** 2026-10-03

**Decision:** the server uses a `psycopg_pool.AsyncConnectionPool` instead of one `AsyncConnection` held for the whole lifespan. Each tool call checks out a connection with `async with pool.connection()`.

**Consequence:** the pool is created per MCP session, not per process. The SDK enters the lifespan in `Server.run` ([`lowlevel/server.py#L657`](https://github.com/modelcontextprotocol/python-sdk/blob/v1.27.2/src/mcp/server/lowlevel/server.py#L657)), and the streamable-HTTP session manager runs `Server.run` once for every new session ([`streamable_http_manager.py#L299`](https://github.com/modelcontextprotocol/python-sdk/blob/v1.27.2/src/mcp/server/streamable_http_manager.py#L299)); the links are pinned to python-sdk `v1.27.2`, the version in `uv.lock`. The `sse` transport works the same way: its handler calls `Server.run` once per SSE connection ([`fastmcp/server.py#L845`](https://github.com/modelcontextprotocol/python-sdk/blob/v1.27.2/src/mcp/server/fastmcp/server.py#L845)). Each client session therefore opens its own pool and holds at least one Postgres connection, up to `max_size`. stdio is a single session. Over HTTP, a session that the client never ends with `DELETE` keeps its pool open, because the SDK sets no idle timeout (measured in Decision 27: 20 such sessions held 20 connections), so the number of connections has no fixed bound.

**Rationale:** psycopg runs queries on one connection one at a time ([psycopg docs, "Concurrent operations"](https://www.psycopg.org/psycopg3/docs/advanced/async.html#concurrent-operations)), and the SDK handles each incoming message in its own task ([`lowlevel/server.py#L678`](https://github.com/modelcontextprotocol/python-sdk/blob/v1.27.2/src/mcp/server/lowlevel/server.py#L678)). Several tool calls sent at once by one client therefore queue on a single connection. The pool lets them run in parallel. Concurrency is small (one person, a few clients) but real.

**Configuration** (`server.py`):
- `min_size=1, max_size=5` per pool, sized for one person's clients.
- No `check=AsyncConnectionPool.check_connection`: a psycopg discussion reports `PoolTimeout` when it is combined with background tasks ([psycopg #985](https://github.com/psycopg/psycopg/discussions/985)). Connections are recycled passively with `max_lifetime=3600, max_idle=300`.
- `pool.open(wait=True, timeout=10)`. By default the pool accepts clients before `min_size` connections are ready ([psycopg_pool docs](https://www.psycopg.org/psycopg3/docs/advanced/pool.html#pool-startup-check)), so an unreachable database would give a "healthy" start and fail only on the first tool call. With `wait=True` the server fails at startup instead.
- `PoolTimeout` and `OperationalError` are caught in `_acquire()` and raised as a short `PoolUnavailable` message; the log line carries `pool.get_stats()` and no stack trace.

**Alternative not taken:** one pool per process, created outside the SDK lifespan and shared by all sessions. It would bound the connections whatever the number of sessions. It is not implemented; the per-session pool is recorded in `docs/known-issues.md` (#17).

**Known issue, not fixed here:** `_refresh_redis()` in `memorize.py` does a non-atomic HGETALL → mutate → HSET on the session hash. With the pool, two overlapping `memorize()` calls with the same `chat_id` can overwrite each other's session-context write, within one session as well as across sessions. Currently `_refresh_redis()` is the only code that writes the hash and the only field it changes is `last_activity`, so the practical loss is a slightly older timestamp; the race would start losing real data if another code path began writing `open_todos`, `active_entities` or `recent_decisions`. Redis working memory is treated as a best-effort cache, so this is left unfixed; recorded in `docs/known-issues.md` (#7).

**Affected files:** `src/mlms/server.py` (`_AppCtx`, `_lifespan`, `_acquire`), `pyproject.toml` (added `psycopg_pool>=3.2`), `tests/integration/test_pool.py`.

**See also:** Decision 6, Decision 27.

---

## Decision 4: DNS rebinding protection enabled, allowlist limited to loopback

**Decided:** 2026-08-29

**Last updated:** 2026-10-03 (upper bound `<2` added)

**Decision:** enable DNS rebinding protection explicitly for the HTTP transports, with `allowed_hosts` limited to loopback (`localhost` and `127.0.0.1`, with or without a port) and `allowed_origins` to the same hosts with a scheme (Decision 22), and raise the dependency floor to `mcp>=1.23.0` (later raised to `mcp>=1.27.2`, see Decision 23).

**Rationale:** CVE-2025-66416 ([GHSA-9h52-p55h-vw2f](https://github.com/advisories/GHSA-9h52-p55h-vw2f)): the MCP Python SDK did not enable DNS rebinding protection by default (fixed in 1.23.0). It affects only `sse` and `streamable-http`; the default `stdio` transport opens no network port. The SDK now enables the protection automatically, but only when the host is `127.0.0.1`, `localhost` or `::1` ([`fastmcp/server.py`](https://github.com/modelcontextprotocol/python-sdk/blob/v1.23.0/src/mcp/server/fastmcp/server.py)). MLMS defaults `mcp_host` to `0.0.0.0`, so the setting has to be explicit. `allowed_origins` is required too, because the SDK validates `Origin` inside the same gate ([`transport_security.py`](https://github.com/modelcontextprotocol/python-sdk/blob/v1.23.0/src/mcp/server/transport_security.py)). The floor guards against a re-resolve of `uv.lock` (currently `1.27.2`) going below the fixed version. The range is capped at `<2`: mcp 2.x renames `FastMCP` to `MCPServer`, which breaks the `mcp.server.fastmcp` import (Decision 20), so a resolution outside `uv.lock` stays on 1.x until the code is migrated.

**Consequence:** if `MCP_HOST` points outside loopback (LAN, VPS, domain), every request gets `421 Invalid Host Header` until that host is added to `_LOOPBACK_HOSTS` in `server.py` (the Origin list is built from it, Decision 22). This is deliberate; the same confusion is reported in other MCP servers ([blockscout/mcp-server #333](https://github.com/blockscout/mcp-server/issues/333)). A regression test (`tests/unit/test_server_transport_security.py`) fails if the protection is turned off again.

**Alternatives:** a wider allowlist for hypothetical proxy or domain deployments was rejected as speculative. Adding a host later means editing `_LOOPBACK_HOSTS` in `server.py` (and rebuilding the image for the container).

**Affected files:** `src/mlms/server.py`, `pyproject.toml`, `uv.lock`, `tests/unit/test_server_transport_security.py`

**See also:** Decision 9, Decision 20, Decision 22, and Decision 23.

---

## Decision 5: Single `timescaledb-ha` image for PostgreSQL

**Decided:** 2026-05-08

**Last updated:** 2026-10-13 (image tags pinned; a tag is not an immutable image identity)

**Decision:** `docker-compose.yml` uses one image, `timescale/timescaledb-ha`, instead of `postgres` plus separately installed extensions. The tag is pinned to `pg16.15-ts2.30.2` (PostgreSQL 16.15, TimescaleDB 2.30.2, pgvector 0.8.7 as checked in that image) so that a new upstream release cannot move the database to a later version under an unchanged checkout. A tag is not an immutable image identity: the registry can re-point it, and no digest is pinned. `redis` is pinned the same way (`7.4.11-alpine`). Dependabot proposes tag bumps.

**Consequence:** the data directory in this image is `/home/postgres/pgdata/data`, not the stock `/var/lib/postgresql/data`. Switching to another Postgres image means changing the volume mapping too.

**Rationale:** the image already ships TimescaleDB and pgvector (the [Dockerfile](https://github.com/timescale/timescaledb-docker-ha/blob/master/Dockerfile#L186) installs `postgresql-${pg}-pgvector`), so there are no init scripts to mount and no extensions to compile.

**Affected files:** `docker-compose.yml`

**See also:** Decision 19, Decision 20.5, and Decision 21.

---

## Decision 6: Conflict detection by cosine distance, inside a SERIALIZABLE transaction

**Decided:** 2026-05-10

**Last updated:** 2026-10-09 (migrations squashed into one initial schema, Decision 28)

**Decision:** when a fact is written, its embedding is compared with the nearest `is_current=TRUE` fact of the same `project_id`. If the cosine distance is at most `1 - COSINE_CONFLICT_THRESHOLD`, the old row gets `is_current=FALSE` and the new row is inserted with `version + 1` and `replaces_id` pointing at the old row. Both statements run in one transaction whose first statement is `SET TRANSACTION ISOLATION LEVEL SERIALIZABLE`. A `SerializationFailure` restarts the whole transaction, at most 3 attempts in total and with no backoff; after the third failure the error propagates. If the connection is already inside a transaction, the `SET` is skipped and the write runs at that transaction's isolation level (Decision 26).

**Rationale:** the transaction makes retire-and-insert atomic: a failure between the two statements cannot leave the old fact retired with no replacement. The isolation level is raised because at the default `READ COMMITTED` the check and the insert are separate statements, so two concurrent `memorize()` calls with near-identical content both pass the check and both insert an `is_current=TRUE` row. `get_facts` and `search_memory` filter on `is_current=TRUE`, so they would return both versions.

**Verification:** checked with an ad hoc script that is not in the repository, not by a repeatable test. The script ran two concurrent `upsert_fact` calls with the same embedding, 60 times per case (with and without an existing fact), against `timescale/timescaledb-ha:pg16`, on a `facts` table and HNSW index shaped like the ones in the initial migration. At `READ COMMITTED` both rows stayed `is_current=TRUE` in every trial; at `SERIALIZABLE` exactly one did, and neither writer got an error. No test in the repository covers this: the unit tests mock the connection and check only the order of statements, and the integration tests call `upsert_fact` sequentially.

**Known limitation:** the retries are bounded. In the same check, four concurrent writers on an existing fact produced 60 `SerializationFailure` errors over 60 trials (one per trial on average), and every trial still ended with exactly one current row. That is beyond the intended use of one person with a few clients (Decision 3), so there is no backoff and no larger bound. `SerializationFailure` is a subclass of psycopg's `OperationalError`, so `_acquire()` (Decision 3) reports it as the generic "database temporarily unavailable, try again shortly" and logs it as a pool problem.

**Threshold:** `COSINE_CONFLICT_THRESHOLD = 0.96` is a starting point that has not been calibrated against real data. On a separate labeled set of 76 pairs it detected no conflicts at all (`docs/known-issues.md`, #2).

**Affected files:** `src/mlms/storage/facts.py` (`upsert_fact`), `src/mlms/config.py`

**See also:** Decision 3, Decision 10, Decision 26.

---

## Decision 9: `MCP_TRANSPORT` switch between stdio and HTTP

**Decided:** 2026-05-16

**Last updated:** 2026-09-04

**Decision:** `MCP_TRANSPORT` (default `stdio`) selects the transport in `server.py`. For `sse` and `streamable-http` the server also reads `MCP_HOST` / `MCP_PORT`.

**Rationale:** `stdio` is enough when the MCP client launches the server as a child process, and it opens no network port. HTTP is needed when the client cannot start the server itself, for example a server on another machine.

**Affected files:** `src/mlms/config.py`, `src/mlms/server.py`

**See also:** Decision 4 (DNS rebinding applies only to the HTTP transports), Decision 20 (the container uses `streamable-http`).

---

## Decision 10: `revert_fact` follows the `replaces_id` chain and blocks cross-project reverts

**Decided:** 2026-05-20

**Last updated:** 2026-10-09 (migrations squashed into one initial schema, Decision 28)

**Decision:** each fact version stores `replaces_id`, a reference to the version it replaced (column `facts.replaces_id`, set on conflict as in Decision 6). `revert_fact()` retires the given fact and restores its predecessor in one transaction, one step back. It first checks that the given fact is the current version and that both versions have the same `project_id`, and refuses otherwise.

**Rationale:** chains are built inside one project, so a mismatch can only come from inconsistent data (a manual edit or a bug). The project check stops a revert from reactivating a fact in another project in that case. The `is_current` check refuses to revert a version that is no longer current: reverting an already replaced version would restore its predecessor while the newer version stays current. Both checks run before the transaction starts, so a concurrent `memorize()` can still change the chain between check and write. The tool takes only `fact_id`, so it does not check that the caller owns the fact.

**Affected files:** `alembic/versions/001_initial_schema.py`, `src/mlms/tools/revert_fact.py`

**See also:** Decision 6, Decision 16.

---

## Decision 14: Keyset pagination on `(time, id)` in `get_events()`

**Decided:** 2026-05-13

**Last updated:** 2026-10-13 (what the tool can reach beyond `limit`)

**Decision:** the cursor is the pair `(time, id)` with `WHERE time > %s OR (time = %s AND id > %s)`, ordered by `time, id`, instead of `time` alone.

**Rationale:** timestamps are not unique. With a `time`-only cursor, events sharing a timestamp across a page boundary are skipped or repeated. `id` gives a strict tie-break.

**Known limitation:** pagination is implemented and unit-tested in `storage/timeline.py`, but the `get_timeline` tool does not use it. It runs its own SQL without a cursor, so one call returns at most `limit` events; older rows need a larger `limit` or a different `time_range`. This is a limit of the tool, not of the pagination code; it is recorded in `docs/known-issues.md` (#14).

**Affected files:** `src/mlms/storage/timeline.py` (`get_events`)

---

## Decision 15: `memorize(content)` length limit: hard 10,000 characters, soft 500

**Decided:** 2026-08-30

**Last updated:** 2026-09-01

**Decision:** `content` above `MEMORY_CONTENT_MAX_LENGTH = 10_000` characters is rejected, not truncated. The check runs in the tool signature through a `BeforeValidator`, so the error reports the actual length and the limit; `Field(max_length=...)` stays alongside so the limit also appears in the tool's JSON Schema. The tool docstring recommends about 500 characters per fact (guidance only, not enforced), so composite facts are split across several calls.

**Rationale:** for facts the whole text is sent to the embedding API. At an assumed 1.5 characters per token for Cyrillic (an estimate, not measured with the real tokenizer), 10,000 characters is about 6,700 tokens. That is far below the 32k context of both Qwen3-Embedding models ([8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B), [0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)) and within the `max_length = 8192` used in both model cards' usage examples. The margin is deliberate: the fallback `qwen3-embedding-0.6b` is intended for local serving (not implemented yet), where the usable context is set by the server.

**Alternatives:**
- Truncate silently: loses part of a fact without warning, while an error lets the agent shorten or split the text.
- Different limits per `memorize()` type: only `fact` content and the `session_log` label (already capped at 100 characters) reach the embedding API, so `event` and `skill` are not restricted in practice, and per-type limits would only add complexity.

**Affected files:** `src/mlms/server.py` (`_check_content_length`, `_MemoryContent`, `memorize` signature and docstring), `src/mlms/config.py`, `tests/unit/test_memorize_content_limit.py`

---

## Decision 16: MCP tool annotations: minimal set on read-only tools, all four fields on write tools

**Decided:** 2026-09-01

**Last updated:** 2026-09-19 (`create_project` annotations added)

**Decision:** every tool carries `ToolAnnotations`. Read-only tools set only `readOnlyHint=True` and `openWorldHint`, because `destructiveHint` and `idempotentHint` are meaningful only when `readOnlyHint` is false ([MCP schema](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/schema/2025-06-18/schema.ts)). `memorize`, `revert_fact` and `create_project` set all four fields explicitly: a missing `destructiveHint` defaults to `true` for a tool that is not read-only, and none of the three deletes rows. Only `create_project` sets `idempotentHint=True`, because it is get-or-create by name.

**`openWorldHint`** follows whether a tool can reach the external embedding API, not the tool type. It is `True` for `get_facts` (with `about`), `get_session_log` (with `semantic_query`), `memorize` and `search_memory`, and `False` for `get_timeline`, `get_session_context`, `get_skills`, `revert_fact`, `create_project` and `export_project`. The hint is static per tool, so a tool that calls the API only for some arguments is marked `True`. The schema default is `true`, so the `False` values are the ones that matter.

**`revert_fact` hints:**
- `destructiveHint=False` is a judgment call. No row is deleted (`is_current` is flipped), but for the calling agent a reverted fact disappears from the default `get_facts` and `search_memory` results, which both filter on `is_current=TRUE`.
- `idempotentHint=False` is set conservatively: a concurrent `memorize()` can change the version chain between two calls.

**Affected files:** `src/mlms/server.py`, `tests/unit/test_server_tool_annotations.py`

**See also:** Decision 10.

---

## Decision 17: Filters applied in SQL before `LIMIT`, HNSW iterative scan for filtered session-log search

**Decided:** 2026-09-01

**Last updated:** 2026-09-19

**Decision:** `get_session_log` (`entry_type`) and `get_skills` (`domain`) filter inside the SQL query, before `LIMIT`. For the cross-session (vector) search in `get_session_log`, `SET LOCAL hnsw.iterative_scan = 'strict_order'` is enabled.

**Rationale:**
- Filtering in Python after the storage layer has applied `LIMIT` loses matching rows ranked below the first `LIMIT` rows, even though they exist.
- The HNSW index has the same effect: pgvector applies the filter after scanning the index's candidate list (`hnsw.ef_search`, default 40), so a selective filter can return fewer than `LIMIT` rows ([README, filtering](https://github.com/pgvector/pgvector#filtering)). Iterative scans (pgvector 0.8.0 and later) keep scanning until enough rows are found or `hnsw.max_scan_tuples` (20,000 by default) is reached ([README, iterative index scans](https://github.com/pgvector/pgvector#iterative-index-scans)).

**Implementation note:** the pool runs with `autocommit=True` (`server.py`), so every bare `execute()` is its own transaction and `SET LOCAL` would be gone before the query runs. The `SET LOCAL` and the search `SELECT` therefore share one explicit transaction, the same pattern as `upsert_fact` (Decision 6). A plain `SET` is session-scoped ([PostgreSQL docs](https://www.postgresql.org/docs/current/sql-set.html)) and would stay on the pooled connection for the next borrower; the pool has no `reset` callback configured.

**Alternatives:** exact (non-indexed) search on `session_log` avoids approximate-recall loss but gives up the HNSW index chosen in Decision 2. It was not pursued and not measured; it remains an option if approximate misses become visible.

**Affected files:** `src/mlms/storage/session_log.py`, `src/mlms/storage/skills.py`, `src/mlms/tools/get_session_log.py`, `src/mlms/tools/get_skills.py`, `tests/unit/test_session_log.py`, `tests/unit/test_skills.py`, `tests/unit/test_get_session_log.py`, `tests/unit/test_get_skills.py`, `tests/integration/test_session_rag.py` (runs only with `MLMS_INTEGRATION=1`, against the Compose stack or the CI service containers)

**See also:** Decision 2, Decision 19.

---

## Decision 18: `psycopg[binary]` instead of a system libpq, and no `types-redis` stub

**Decided:** 2026-09-03

**Decision:** `psycopg` is installed with the `binary` extra, and the `types-redis` dev dependency is removed.

**Rationale:** without the `binary` extra, psycopg needs client libraries on the machine ([psycopg installation docs](https://www.psycopg.org/psycopg3/docs/basic/install.html)), and `uv.lock` never contained `psycopg-binary`. A fresh install from the lock therefore depended on what the machine happened to have installed. The binary package is self-contained.

**Trade-off:** the same docs prefer a local installation for production, linked against the system libpq and libssl so that OS updates apply to psycopg too. The binary wheel bundles its own copies, so a production image may build psycopg from source instead.

**Stub removal:** the `types-redis` stub targets redis 4.6, and its [PyPI page](https://pypi.org/project/types-redis/) says redis ships its own annotations since 5.0.0 and asks to uninstall the stub. The locked redis-py 8.0.0 has `py.typed`. The stub caused false mypy errors: it has no `aclose` (which exists at runtime), and it makes `Redis` generic in its string type. Without the stub, `embedding.py` annotates the client as plain `Redis` and needs no `# type: ignore`.

**Affected files:** `pyproject.toml`, `uv.lock`, `src/mlms/embedding.py`

---

## Decision 19: CI covers lint, types, unit and integration tests and the image build

**Decided:** 2026-09-03

**Last updated:** 2026-10-13 (integration tests, docs check and image build in CI; image tags pinned)

**Decision:** the GitHub Actions workflow (`.github/workflows/ci.yml`) runs `ruff check`, `ruff format --check`, `mypy src`, the documentation consistency check (`scripts/check_docs.py`), `alembic upgrade head`, the unit tests with the coverage gate (`pytest -m unit`, `fail_under = 80` in `pyproject.toml`), the integration tests (`MLMS_INTEGRATION=1 pytest -m integration`) and `docker build` of the server image. Postgres and Redis run as job `services:` containers from the images in `docker-compose.yml` (`timescale/timescaledb-ha:pg16.15-ts2.30.2`, Decision 5, and `redis:7.4.11-alpine`) with the same credentials; a comment in the workflow says to keep the two files in sync.

**Rationale:** the integration tests cover what unit tests with mocked connections cannot: pool behavior, the conflict flow, and filtered session-log search with the HNSW iterative scan (Decision 17). Without them in CI, a green run would say nothing about those paths. A hook in `tests/conftest.py` marks every test under `tests/unit/` as `unit` and every test under `tests/integration/` as `integration`, so the two `-m` selections do not overlap. Without `MLMS_INTEGRATION=1` the integration tests skip themselves, which keeps a plain `pytest` run from reaching for a database that is not there.

**Consequence:** the coverage gate applies to the unit run only; the integration run passes no `--cov`. The images and credentials exist in both `docker-compose.yml` and `ci.yml`, and no test checks that they agree. The workflow's steps were reproduced locally against the same two images. A run on GitHub Actions itself was not checked.

**Reproducibility and hardening:**
- `uv sync --locked --extra dev` fails the build if `uv.lock` is out of sync with `pyproject.toml`.
- Actions are pinned to exact tags (`actions/checkout@v7.0.1`, `astral-sh/setup-uv@v10.0.1`), not moving major tags. From v8.0.0 `setup-uv` no longer publishes major or minor tags ([release notes](https://github.com/astral-sh/setup-uv/releases/tag/v8.0.0)); `checkout` is pinned the same way for consistency. A commit SHA would be stricter (the same release notes recommend it) and was not used.
- `permissions: contents: read` limits the workflow token, `timeout-minutes: 10` stops a hung job, and `ubuntu-24.04` replaces the floating `ubuntu-latest`.
- A per-test limit of 30 seconds is set in the pytest configuration (Decision 27).
- A new push to the same pull request cancels its previous run. Runs on push to `main` are grouped per commit and never cancelled.
- One job on Python 3.11, the lowest version allowed by `requires-python`.

**Affected files:** `.github/workflows/ci.yml`, `tests/conftest.py`, `README.md` (test commands and CI description)

**See also:** Decision 5, Decision 17, Decision 20, Decision 27.

---

## Decision 20: Dockerfile and compose service: install from `uv.lock`, HTTP transport, migrations on start

**Decided:** 2026-09-04

**Last updated:** 2026-10-13 (healthcheck over TCP, cause of the startup race identified; `mlms` service behind the `app` profile; image build copies `README.md` and `LICENSE`)

**Decision:** the MLMS server runs in a container built from `Dockerfile`, as the `mlms` service of `docker-compose.yml`. The image installs the project with `uv sync --locked --no-editable` and starts through `entrypoint.sh`, which runs `alembic upgrade head` (with retries) and then the server. The compose service sets `MCP_TRANSPORT=streamable-http`. Inside the Compose network `DATABASE_URL` and `REDIS_URL` use the service names `postgres` and `redis`, not `localhost`, which differs deliberately from the host-run defaults in `.env.example`.

**Install from `uv.lock`, not `pip install .`:** `uv sync --locked` installs exactly the versions in `uv.lock`, the set the CI runs against (Decision 19), and fails if the lock is out of sync with `pyproject.toml`. `pip install .` ignores `uv.lock` and resolves from PyPI. `pyproject.toml` declares `mcp>=1.27.2,<2`, so that resolution stays on 1.x too: mcp 2.x renamed `FastMCP` to `MCPServer`, so `from mcp.server.fastmcp import FastMCP` raises `ModuleNotFoundError` ([migration guide](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/migration.md)), and an unbounded range would make the container fail on start with that error. The `<2` cap follows the SDK's own README, which recommends it until an application has migrated ([v1.x README](https://github.com/modelcontextprotocol/python-sdk/blob/v1.x/README.md)); it stays until the code moves to `MCPServer` (see Decision 4). `--no-editable` installs the package into the virtualenv as a regular copy instead of a link to `src/`.

**Transport:** `stdio` needs the MCP client to start the server as a child process and attach its stdin. A detached Compose service has no such stdin, so the container must use an HTTP transport. `streamable-http` was chosen over `sse` because it replaces the HTTP+SSE transport in the MCP specification ([transports](https://modelcontextprotocol.io/specification/2025-06-18/basic/transports)); `config.py` accepts both.

**Migrations on start, with retry:** a fresh `postgres_data` volume has no schema, so without an upgrade the server would fail on its first database call. Running the migrations manually was rejected for that reason, so `alembic.ini` and `alembic/` are copied into the image and every container start runs them. The `postgres` healthcheck runs `pg_isready -h 127.0.0.1` (TCP). On a new volume the image starts a temporary server that listens only on the unix socket, runs its init scripts, stops it and starts the real one; a check over the socket would pass against the temporary server and mark `postgres` healthy before the restart, so the first `alembic upgrade head` could fail with a dropped connection. Only the real server answers over TCP. Three consecutive starts on fresh volumes ran the migration straight after `docker compose up --wait` with no failure. A Postgres that restarts later, or a slow start, can still make the first attempt fail, and `restart: unless-stopped` would recover from that only through an uncapped crash loop with a traceback. So the entrypoint retries `alembic upgrade head` itself, up to 10 attempts 2 seconds apart, and exits with status 1 after the last one. It then `exec`s `python -m mlms.server`, so `SIGTERM` reaches the server process, not a shell. The retry does not tell a startup race from a real failure (wrong `DATABASE_URL`, a broken migration). A misconfigured container therefore waits at least 18 seconds (nine pauses), longer if each attempt has to time out, before it exits.

**`HEALTHCHECK`:** a TCP connect to `MCP_PORT` in Python. The `python:slim` image does not ship `curl`, and the check needs no MCP-specific request. It shows that the port accepts connections, not that the MCP endpoint works. Under Docker Compose the container reaches `healthy`.

**Trade-off:** no dependency-only layer. The build copies `pyproject.toml`, `uv.lock`, `alembic.ini`, `README.md`, `LICENSE`, `alembic/`, `src/` and `entrypoint.sh` (`pyproject.toml` names the README and the licence file as package metadata, so the build needs both) and runs one `uv sync`, so any code change reinstalls all dependencies. A separate layer is possible with `uv sync --no-install-project`; it was left out to keep the Dockerfile simple. The image is not published to a registry; users build it with `docker compose --profile app up -d --build`.

**Affected files:** `Dockerfile`, `entrypoint.sh`, `.dockerignore`, `.gitignore`, `docker-compose.yml`, `README.md`

**See also:** Decision 4, Decision 9, Decision 19, Decision 20.5, and Decision 25.

---

## Decision 20.5: Postgres credentials read from `.env`, Redis stays unauthenticated

**Decided:** 2026-09-05

**Last updated:** 2026-09-07

**Decision:** `docker-compose.yml`'s `postgres` service and the `mlms` service's `DATABASE_URL` both read `POSTGRES_DB`/`POSTGRES_USER`/`POSTGRES_PASSWORD` from environment variables with the same `mlms`/`mlms`/`mlms` default, declared in `.env.example`. The `postgres` healthcheck reads the same variables from the container's own environment (`$$POSTGRES_USER`/`$$POSTGRES_DB`, a literal not interpolated by Compose, per the [Compose spec](https://github.com/compose-spec/compose-spec/blob/master/12-interpolation.md)) instead of repeating the default a third time.

**Rationale:** the credentials appear in three places: `postgres`'s own environment, the `mlms` service's `DATABASE_URL` and the healthcheck command. Reading them from one set of variables means a password change is made once. `.env.example`'s variables are read only by Compose's interpolation; the application's own `Settings` (`extra="ignore"`) never reads `POSTGRES_*`, so the shipped default there is set equal to the app's `DATABASE_URL` default to avoid the two silently drifting apart.

**Alternatives:** parametrizing `redis` the same way was not pursued. `redis` has no authentication enabled at all in this compose file, so there is no credential to expose through `.env`; that would mean turning auth on (`requirepass`), which is left for later.

**Known limitation:** changing `POSTGRES_PASSWORD` only takes effect against a fresh `postgres_data` volume, because the image applies `POSTGRES_*` only at first initialization. An existing volume needs `docker compose down -v` or a manual `ALTER ROLE`. The characters `` @ : / % # `` in the password are not URL-encoded when composed into `DATABASE_URL`, so a password containing one of them produces an invalid connection string, and the resulting error does not point at the password.

**Affected files:** `docker-compose.yml`, `.env.example`, `README.md`

**See also:** Decision 5, Decision 20.

---

## Decision 21: Backup/restore uses custom-format `pg_dump`/`pg_restore`, not plain SQL, and never `-j`

**Decided:** 2026-09-06

**Last updated:** 2026-10-12 (image tags pinned)

**Decision:** the backup/restore guide in README uses `pg_dump -Fc` (custom format) with `timescaledb_pre_restore()` / `pg_restore` / `timescaledb_post_restore()`, not a plain-text dump restored with `psql`. `pg_restore` runs without `-j` (parallel restore).

**Rationale:** TimescaleDB's own backup guide requires the custom format plus the pre/post-restore pair to reconstruct hypertable metadata (chunk structure) as well as row data ([Logical backup](https://www.tigerdata.com/docs/deploy/self-hosted/backup-and-restore/logical-backup)). The same page recommends against dumping individual tables with `pg_dump -t`: even naming the hypertable and every one of its chunks explicitly is not enough to recreate the hypertable on restore. `pg_restore -j` is documented there as not correctly restoring the TimescaleDB catalogs, with a long-standing report of restores failing on foreign-key errors in the catalog tables under it ([timescaledb#1584](https://github.com/timescale/timescaledb/issues/1584)); official support for `-j` is still an open feature request ([#698](https://github.com/timescale/timescaledb/issues/698)), so the guide does not use it.

**Restore targets a fresh database, not the live one:** the server's startup migration has already created the schema, and `pg_restore` does not overwrite an existing schema, so restoring into the live database fails with "relation already exists". The guide runs `DROP DATABASE` / `CREATE DATABASE` first. No extra step is needed for extensions: `timescaledb` and `timescaledb_toolkit` are already installed into `template1` by the image's own init script ([`post_init.sh`](https://github.com/timescale/timescaledb-docker-ha/blob/master/scripts/post_init.sh)), so a freshly created database has them automatically, and `vector` is created by the dump itself (`pg_dump` includes `CREATE EXTENSION IF NOT EXISTS vector`). Restore was verified end-to-end: a memorized fact survived a dump, drop, recreate and restore with the same `id` and content, checked through a real `get_timeline` call.

**Redis:** `SAVE` (not `BGSAVE`) runs synchronously before the file is copied out. That is simpler to reason about in a manual runbook, at the cost of blocking the server for the duration. `dump.rdb` is safe to copy while Redis keeps running: Redis writes the new snapshot to a temporary file and renames it into place only once it is complete ([Redis persistence docs](https://redis.io/docs/latest/operate/oss_and_stack/management/persistence/)), so a copy taken mid-write still reads a complete, previous snapshot. `dump.rdb`/`/data` (`dbfilename`/`dir`) were checked with `CONFIG GET` against the running container. Restore was verified the same way: a deleted `session:{chat_id}` key reappeared with its original content after restoring the snapshot.

**Alternatives:**
- `redis-cli --rdb <file>` run from the host: rejected because the host's `redis-cli` version can drift from the `redis:7.4.11-alpine` version in the container, and the replication protocol used to stream the RDB is version-sensitive.
- The same command from a matching container (`docker run --network <compose-net> redis:7.4.11-alpine redis-cli -h redis --rdb -`) avoids that mismatch and was confirmed to work, but needs the compose network's name; `SAVE` + `docker compose cp` was kept as the shorter default.

**Affected files:** `README.md`

---

## Decision 22: Origin allowlist needs a scheme-qualified list, separate from the Host allowlist

**Decided:** 2026-09-07

**Last updated:** 2026-09-19

**Decision:** `allowed_origins` gets its own list, `_LOOPBACK_ORIGINS` (every entry of `_LOOPBACK_HOSTS` with `http://` and with `https://`), instead of reusing the Host list.

**Rationale:** a `Host` header never carries a scheme; an `Origin` header always does (`"http://127.0.0.1"`, per the [Fetch spec](https://fetch.spec.whatwg.org/#origin-header)). Checked against the scheme-less Host list, no `Origin` header can match, including a legitimate loopback one, and every request that carries one is rejected with `403`. Only browser-based clients send `Origin`. A CLI or agent client that omits it passes anyway, because the SDK treats a missing `Origin` as same-origin, and `stdio` never runs this code (Decision 4).

**Test:** The regression test drives a real HTTP request through `TestClient` rather than calling the SDK's private `_validate_origin` method directly, so it keeps working across an SDK version that renames or removes that method. The app under test is a `FastMCP` instance that carries the real server's `transport_security` settings and a no-op lifespan, so the test does not open a database pool (Decision 27).

The README section "Homelab/VPS hardening" describes the deployment side: firewall, tunnels, a reverse proxy, and the `curl` recipe that checks the `421` and `403` responses. Whether a tunnel forwards the original `Host`/`Origin` was not verified against a live tunnel, and the README says so.

**Known gap, accepted:** the allowlist match is exact and case-sensitive with no host normalization, so `Host: [::1]`, `LOCALHOST`, or a trailing-dot `localhost.` fail the check. This fails toward rejecting a legitimate request, not toward accepting an illegitimate one, so it is documented in the README rather than fixed.

**Alternatives:** an `MCP_ALLOWED_HOSTS` environment variable instead of editing `_LOOPBACK_HOSTS` directly. Rejected as configuration for a case (a non-loopback deployment) that most users of a single-user project will not hit.

**Affected files:** `src/mlms/server.py` (`_LOOPBACK_ORIGINS`), `tests/unit/test_server_transport_security.py`, `README.md`

**See also:** Decision 4, Decision 23, and Decision 27.

---

## Decision 23: Shared-secret gate for sse/streamable-http, `MCP_SHARED_SECRET`

**Decided:** 2026-09-08

**Last updated:** 2026-10-11 (compose: `app` profile, empty default for the secret)

**Decision:** `sse` and `streamable-http` are protected by one shared secret, `MCP_SHARED_SECRET`. `stdio` has no network port and stays unauthenticated. An ASGI middleware (`_SharedSecretMiddleware`) wraps the app returned by `sse_app()`/`streamable_http_app()` and rejects any `http`-scope request with `401` unless its `Authorization: Bearer <token>` matches the secret (`secrets.compare_digest`). The Host/Origin allowlist from Decision 4 is built inside those app factories, so it stays in effect underneath. When a secret is set, the server runs the wrapped app with `uvicorn.run` instead of `mcp.run()`. A non-loopback `MCP_HOST` without a secret refuses to start (`_require_secret_on_non_loopback_bind`); on a loopback bind (`127.0.0.1`/`localhost`/`::1`) the secret is optional.

**Rationale:** the SDK's own `token_verifier`/`AuthSettings` path was rejected: `AuthSettings.issuer_url`/`.resource_server_url` are both required, and setting them makes the server publish OAuth discovery metadata even with no real authorization server behind it. A client that sees that metadata can ignore a configured static `Authorization` header and fall into a broken OAuth flow instead, as reported for Claude Code's MCP client ([anthropics/claude-code#59467](https://github.com/anthropics/claude-code/issues/59467)). The middleware does a flat token compare instead, with no discovery surface at all.

**Known limitations, accepted:**
- CVE-2026-52869 ([GHSA-jpw9-pfvf-9f58](https://github.com/advisories/GHSA-jpw9-pfvf-9f58), fixed in `mcp` 1.27.2, raising the floor set by Decision 4) closes a session-hijack gap by tracking each session's owning principal and rejecting a mismatched requester, but only for the SDK's own `BearerAuthBackend`, which sets `scope["user"]`. This project's middleware never sets it, so the SDK's ownership check compares `None != None` and never rejects; the advisory itself says as much ("Deployments using a custom authentication backend other than the built-in `BearerAuthBackend` should enforce an equivalent check themselves"). It is accepted because the data model has no per-user isolation to protect regardless (`project_id`/`chat_id` are plain tool arguments, not an identity scope), so a hijacked session exposes nothing a valid secret holder couldn't already reach directly. Multi-user auth, token rotation, per-user scoping and rate limiting are out of scope for the same reason.
- Confused-deputy and token passthrough (community discussion [modelcontextprotocol/modelcontextprotocol#1247](https://github.com/modelcontextprotocol/modelcontextprotocol/discussions/1247)) do not apply: the server holds its own state and never passes a client's token to a downstream service.

**Consequence:** `docker-compose.yml`'s `mlms` service binds `MCP_HOST=0.0.0.0` (Decision 20) with an empty default for `MCP_SHARED_SECRET` (unlike a password, an auth secret has no safe default), so the container refuses to start until one is set in `.env`. The `mlms` service sits behind the `app` profile, so a plain `docker compose up` (Postgres and Redis for the `stdio` setup) needs no secret. A required `${MCP_SHARED_SECRET:?...}` is not used: Compose substitutes variables in the whole file before it applies profiles, so it would block the infrastructure-only start as well. A cross-file test (`tests/unit/test_docker_compose_shared_secret.py`) parses the `mlms` block of `docker-compose.yml` directly (no `pyyaml`, not a declared dependency) and fails if a non-loopback `MCP_HOST` is set there without `MCP_SHARED_SECRET` wired through, a drift that neither file's own tests would see.

**Alternatives:**
- SDK `token_verifier`/`AuthSettings`: rejected above.
- A `middleware:` constructor parameter on `FastMCP`: it does not exist in this SDK; that parameter belongs to the unrelated third-party `fastmcp` (jlowin) package, not `mcp.server.fastmcp`.

**Affected files:** `src/mlms/server.py` (`_SharedSecretMiddleware`, `_BIND_LOOPBACK_HOSTS`, `_require_secret_on_non_loopback_bind`), `src/mlms/config.py` (`mcp_shared_secret`), `pyproject.toml` (`mcp>=1.27.2`), `.env.example`, `docker-compose.yml`, `README.md`, `tests/unit/test_server_shared_secret.py`, `tests/unit/test_docker_compose_shared_secret.py`

**See also:** Decision 4 (the loopback allowlist underneath), Decision 20 (the compose bind this gates), Decision 22 (the same README hardening section).

---

## Decision 24: `skills` are global: no `project_id` column

**Decided:** 2026-05-08

**Last updated:** 2026-10-09 (migrations squashed into one initial schema, Decision 28)

**Decision:** `skills` stays global: there is no `project_id` column, so no tool can scope skills by project. The initial schema (`alembic/versions/001_initial_schema.py`) creates `skills` with only `id, name, domain, content, trigger_conditions, version, created_at`, while `facts` and `timeline_events` carry a `project_id` foreign key from the start. The behavior follows from the schema; it is not a forgotten `WHERE`. `export_project` names its section `skills_global`.

**Rationale:** without a column there is nothing to filter on, and adding one would also decide whether skills are per-project or deliberately shared across projects, an open design question. It is left for a redesign.

**Consequence:** with several `project_id`s in one database, `export_project(project_id="X")` returns every skill in the system in its `skills_global` section, although the top-level `project_id` of the JSON is still X. `get_skills` has no `project_id` parameter, and the skill layer of `search_memory` searches all skills through the same storage function. The `export_project`/`get_skills` symptom is recorded in `docs/known-issues.md` (#13).

**Alternatives:**
- Add a `project_id` column to `skills` in a new migration: needs the per-project-or-global question answered first.

**Affected files:** `src/mlms/tools/export_project.py`, `src/mlms/tools/get_skills.py`, `src/mlms/tools/search_memory.py`, `src/mlms/storage/skills.py`, `alembic/versions/001_initial_schema.py`, `docs/known-issues.md`

---

## Decision 25: HNSW indexes are created with a plain `CREATE INDEX`, `CONCURRENTLY` is not available on `session_log`

**Decided:** 2026-09-16

**Last updated:** 2026-10-09 (rewritten after the migrations were squashed, Decision 28)

**Decision:** the initial migration creates `facts_embedding_idx` and `session_log_embedding_idx` with a plain `CREATE INDEX`.

**Rationale:** the tables are empty when the migration runs, so a build that blocks writes costs nothing. `session_log` is a TimescaleDB hypertable, and TimescaleDB does not support `CREATE INDEX CONCURRENTLY` on hypertables ([docs](https://docs.tigerdata.com/api/latest/hypertable/create_index/)). On `timescale/timescaledb-ha:pg16` the statement fails with `FeatureNotSupported`, "hypertables do not support concurrent index creation", whether or not the hypertable has chunks. `facts` is a plain table and can be rebuilt concurrently. The limit comes from TimescaleDB, not from pgvector or HNSW.

**Consequence:** rebuilding either index on a populated database blocks writes while it builds, except for `facts`, where `CREATE INDEX CONCURRENTLY` works outside a migration transaction. If the `session_log` index ever has to be rebuilt while writes continue, TimescaleDB offers `CREATE INDEX ... WITH (timescaledb.transaction_per_chunk)`, which locks one chunk at a time; in a manual check it accepted an HNSW index on the image above. No migration uses it.

**Alternatives:** `CREATE INDEX CONCURRENTLY` on `session_log`: rejected by TimescaleDB, as above.

**Affected files:** `alembic/versions/001_initial_schema.py`

**See also:** Decision 2, Decision 28.

---

## Decision 26: `upsert_fact` skips the SERIALIZABLE `SET` on a connection already in a transaction

**Decided:** 2026-09-16

**Decision:** before its retry loop, `upsert_fact` checks `conn.info.transaction_status == IDLE`. Only then does the transaction start with `SET TRANSACTION ISOLATION LEVEL SERIALIZABLE` (Decision 6). On a connection that is already inside a transaction the statement is skipped, silently, and the write runs at the isolation level of that transaction: `READ COMMITTED` by default, `SERIALIZABLE` only if the caller set it before its first statement.

**Rationale:** PostgreSQL allows changing the isolation level only before the first query of a transaction ([docs](https://www.postgresql.org/docs/16/sql-set-transaction.html)). On a connection already in a transaction, `conn.transaction()` opens a savepoint, and the `SET` inside it fails with `ActiveSqlTransaction` (SQLSTATE 25001). Skipping it lets such callers work instead of crashing.

**Consequence:** the conflict-detection guarantee of Decision 6 holds only on an idle connection. That is every call through the server: the pool runs with `autocommit=True` (`server.py`), so its connections are idle between statements. The skip path is reached by callers that hold a connection without autocommit and have already run a statement on it, such as the integration-test fixture; two integration tests fail with `ActiveSqlTransaction` without it. No unit test covers the skip path (the mocked connections report `IDLE`).

**Affected files:** `src/mlms/storage/facts.py`, `tests/unit/test_facts.py`, `tests/unit/test_conflict_detection.py`

**See also:** Decision 6.

---

## Decision 27: Global `pytest-timeout` (signal method, 30 s) as a guard against a shutdown hang

**Decided:** 2026-09-16

**Last updated:** 2026-09-19 (the test no longer opens the real lifespan)

**Decision:** `pyproject.toml` sets `timeout = 30` and `timeout_method = "signal"` for pytest, and `pytest-timeout` is a dev dependency. A test that hangs fails after 30 seconds with a traceback instead of blocking the whole run. The only test that enters the `with` block of a `TestClient`, `test_legitimate_loopback_origin_is_not_rejected` (`tests/unit/test_server_transport_security.py`), builds its own `FastMCP` instance with the real server's `transport_security` settings and a no-op lifespan, so it never opens a database pool; the Host and Origin checks it asserts on do not use the database. It passes in 10 of 10 runs with nothing listening. The shared-secret tests skip the `with` block, and the integration tests do not use `TestClient`. The hang itself, in the pool's shutdown, is not fixed, and `_lifespan()` is unchanged.

**The hang:** a test that drives the real ASGI app with the real lifespan through `TestClient(mcp.streamable_http_app())` can block forever when its `with` block exits. Traced with a stack dump of the stuck process and the library sources (mcp 1.27.2, psycopg_pool 3.3.1, the locked versions):
- Every MCP session runs `_lifespan()` (`server.py`), and in streamable-http a request without an `mcp-session-id` header starts a new session, so each such request opens its own `AsyncConnectionPool` (`pool-1`, `pool-2`, ... in the logs).
- On exit `TestClient` stops the anyio portal thread, whose `asyncio.Runner` cancels all remaining tasks once and then waits for them. A pool worker task (`pool-N-worker-0`) cancelled while it runs a connection attempt does not end: psycopg_pool catches `asyncio.CancelledError` together with `Exception` (`CLIENT_EXCEPTIONS` in `pool_async.py`), and the worker goes back to waiting on its queue. The wait never returns, so the main thread stays in `Thread.join()`.
- No outage is needed, only a worker cancelled in the middle of a connection attempt, so the hang depends on timing. With Postgres reachable it occurred in 10 of 10 runs (Python 3.11 and 3.12). With nothing listening on the port it occurred in 6 of 20 runs (2 of 10 on 3.12, 4 of 10 on 3.11), and the test passed in about a second otherwise.

**Why `signal`:** per the [pytest-timeout documentation](https://github.com/pytest-dev/pytest-timeout#readme), the `thread` method ends the whole process with `os._exit()` and no teardown, so the first hung test would abort the entire run; `signal` (SIGALRM) interrupts only that test. A test hung this way fails with `Failed: Timeout (>30.0s)` after about 31 seconds. The anyio portal thread is a daemon thread, so the thread left behind does not keep the interpreter from exiting.

**On the running server:** measured on the real app served by uvicorn over streamable-http (mcp 1.27.2, Postgres and Redis in containers), by counting backend connections in `pg_stat_activity` and tasks from `asyncio.all_tasks()` in the server process:
- Sessions are stateful and have no idle timeout (`session_idle_timeout` is left at the SDK default of none). 20 `initialize` calls without a session header gave 20 sessions, 20 backend connections and about 8 more tasks per session (164 tasks against a baseline of 4), unchanged 30 seconds later. A session that is never ended with `DELETE` keeps its pool and connection.
- `DELETE` closes a session's pool cleanly: after ending all 20, connections were back to 0 and tasks to the baseline.
- A session ended while its pool is in the middle of a connection attempt leaves the worker task behind. With 5 sessions, their connections terminated and Postgres paused, `DELETE` left 5 `pool-N-worker-0` tasks alive for the 58 seconds observed, and they were gone 40 seconds after Postgres was resumed. An orphan therefore lasts as long as the connection attempt it was in. Whether orphans pile up under long real traffic was not measured.

**Consequence:** the limit is global, so any test that runs longer than 30 seconds fails. The unit and integration suites finish in seconds. A new test that opens the real lifespan through `TestClient` can hit the hang again; the timeout turns that into a failure after 30 seconds.

**Alternatives:** patching psycopg_pool or the SDK's lifespan handling: not done, the cause is in library code outside this project.

**Affected files:** `pyproject.toml`, `uv.lock`, `tests/unit/test_server_transport_security.py`

**See also:** Decision 3, Decision 19, Decision 22.

---

## Decision 28: Migrations 001-014 squashed into one initial schema

**Decided:** 2026-10-09

**Decision:** `alembic/versions/` holds a single `001_initial_schema.py` that creates the final schema directly. It replaces the fourteen incremental migrations written during development. Later schema changes are new migrations numbered from `002`.

**Rationale:** nothing had been released, so no database existed that needed an upgrade path, and the incremental chain mostly recorded the development order: a `vector(4096)` first version, an `ALTER COLUMN TYPE` that did nothing on a new database, a concurrent rebuild of an index on an empty table, constraints added one at a time. Keeping it would make every new deployment replay those steps for no benefit, and make the schema harder to read.

**Verification:** the schema built by the old chain and the one built by `001_initial_schema.py` were compared with `pg_dump --schema-only` on `timescale/timescaledb-ha:pg16`. Tables, constraints, indexes and their names are identical. The only difference is column order in `session_log`: `id` is now the first column and `project_id` follows `chat_id`. The code names columns explicitly, so the order does not matter. `alembic downgrade base` followed by `upgrade head` and the integration tests also pass on the new schema.

**Consequence:** a database created by the old chain records one of the revisions `001`-`014`, which no longer match the files, so `alembic upgrade head` fails there. Its schema is the same, so it can keep its data: set `version_num` in the `alembic_version` table to `001`. Recreating the database also works.

**Alternatives:** keeping the fourteen migrations: rejected, there is no upgrade path to protect. Keeping them and adding a squashed baseline next to them: Alembic would still have to replay or skip the old files, which defeats the point.

**Affected files:** `alembic/versions/001_initial_schema.py`

**See also:** Decision 1, Decision 2, Decision 25.

---
