# Key Architectural Decisions — MLMS

Each entry records what was decided, why, and which alternatives were rejected.

Dates are the days the decisions were made. Commit history before 2026-08 was reconstructed afterwards, so early dates are approximate. Entry numbers are stable identifiers, not chronological order; gaps are intentional.

---

## Decision 1: EMBEDDING_DIM = 1536 (Matryoshka truncation)

**Decided:** 2026-05-09

**Decision:** `EMBEDDING_DIM = 1536` instead of the native 4096 of Qwen3-Embedding-8B.

**Rationale:** pgvector 0.8.2 limits HNSW and IVFFlat indexes on `vector` to 2000 dimensions, and `halfvec` to 4000 ([pgvector README](https://github.com/pgvector/pgvector)). Neither fits 4096. Qwen3-Embedding-8B supports Matryoshka Representation Learning with output dimensions from 32 to 4096 ([model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B); technique: [Kusupati et al., 2022](https://arxiv.org/abs/2205.13147)), so its vectors can be truncated to a prefix with a limited loss of quality. Truncating to 1536 brings the column under the 2000-dimension limit and makes an HNSW index possible (see Decision 2). The recall loss from truncation has not been measured in this project.

**Alternatives:**
- Sequential scan with no index on `vector(4096)`: exact (recall 100%), but query cost grows linearly with table size.
- `halfvec(4096)`: the 4000-dimension limit is below 4096.
- Expression index on a 1536-dimension prefix (`subvector(embedding, 1, 1536)`, documented in the pgvector README for models that support it): a valid option, not chosen. It keeps the full 4096 dimensions in storage (16392 bytes per row versus 6152 at 1536, from the README formula `4 * dimensions + 8`), and every query has to use the same expression to hit the index.

**Affected files:** `alembic/versions/006_change_embedding_dim.py`

**See also:** Decision 2.

---

## Decision 2: HNSW index on embedding columns

**Decided:** 2026-05-09

**Last updated:** 2026-08-25

**Decision:** HNSW index on `facts.embedding` and `session_log.embedding`.

**Rationale:** without an ANN index, similarity search is a sequential scan, so its cost grows linearly with table size. The schema initially used `VECTOR(4096)` with no index (the first versions of `002_create_facts.py` and `004_create_session_log.py`). IVFFlat was tried first and hit the same 2000-dimension limit as HNSW, so no ANN index was possible at 4096 dimensions. Truncating to 1536 (Decision 1) removed the limit and allowed the HNSW index.

**Alternatives:**
- Sequential scan: exact, but linear cost.
- IVFFlat: same 2000-dimension limit at 4096.

**Trade-offs:** HNSW search is approximately logarithmic in table size, at the price of higher memory use and a slower index build. At the expected volume (up to about 100K rows, an estimate rather than a measurement) this is acceptable.

**Affected files:** `alembic/versions/002_create_facts.py`, `alembic/versions/004_create_session_log.py`, `alembic/versions/006_change_embedding_dim.py`

**See also:** Decision 1.

---

## Decision 3: Connection pool instead of a single shared connection

**Decided:** 2026-08-24

**Last updated:** 2026-08-29

**Decision:** the server uses a `psycopg_pool.AsyncConnectionPool` instead of one `AsyncConnection` held for the whole lifespan. Each tool call checks out a connection with `async with pool.connection()`.

**Consequence:** the pool is created per MCP session, not per process. The SDK enters the lifespan in `Server.run` ([`lowlevel/server.py#L657`](https://github.com/modelcontextprotocol/python-sdk/blob/v1.27.2/src/mcp/server/lowlevel/server.py#L657)), and the streamable-HTTP session manager runs `Server.run` once for every new session ([`streamable_http_manager.py#L299`](https://github.com/modelcontextprotocol/python-sdk/blob/v1.27.2/src/mcp/server/streamable_http_manager.py#L299)); the links are pinned to python-sdk `v1.27.2`, the version in `uv.lock`. Each client session therefore opens its own pool and holds at least one Postgres connection, up to `max_size`. stdio is a single session.

**Rationale:** psycopg runs queries on one connection one at a time ([psycopg docs, "Concurrent operations"](https://www.psycopg.org/psycopg3/docs/advanced/async.html#concurrent-operations)), and the SDK handles each incoming message in its own task ([`lowlevel/server.py#L678`](https://github.com/modelcontextprotocol/python-sdk/blob/v1.27.2/src/mcp/server/lowlevel/server.py#L678)). Several tool calls sent at once by one client therefore queue on a single connection. The pool lets them run in parallel. Concurrency is small (one person, a few clients) but real.

**Configuration** (`server.py`):
- `min_size=1, max_size=5` per pool, sized for one person's clients, not for multi-tenant traffic.
- No `check=AsyncConnectionPool.check_connection`: a psycopg discussion reports `PoolTimeout` when it is combined with background tasks ([psycopg #985](https://github.com/psycopg/psycopg/discussions/985)). Connections are recycled passively with `max_lifetime=3600, max_idle=300`.
- `pool.open(wait=True, timeout=10)`. By default the pool accepts clients before `min_size` connections are ready ([psycopg_pool docs](https://www.psycopg.org/psycopg3/docs/advanced/pool.html#pool-startup-check)), so an unreachable database would give a "healthy" start and fail only on the first tool call. The old single `connect()` failed at startup, and `wait=True` keeps that.
- `PoolTimeout` and `OperationalError` are caught in `_acquire()` and raised as a short `PoolUnavailable` message with no stack trace; `pool.get_stats()` is logged.

**Known issue, not fixed here:** `_refresh_redis()` in `memorize.py` does a non-atomic HGETALL → mutate → HSET on the session hash. Within one session, the single connection used to serialize the database part of tool calls, which made this race unlikely. The pool removes that serialization, so two overlapping `memorize()` calls with the same `chat_id` in one session can now lose a session-context update. Calls from different sessions never shared a connection, so for devices on separate sessions the race existed before the pool.

**Affected files:** `src/mlms/server.py` (`_AppCtx`, `_lifespan`, `_acquire`), `pyproject.toml` (added `psycopg_pool>=3.2`), `tests/integration/test_pool.py` (new).

---

## Decision 4: DNS rebinding protection enabled, allowlist limited to loopback

**Decided:** 2026-08-29

**Decision:** enable DNS rebinding protection explicitly for the HTTP transports, with `allowed_hosts` and `allowed_origins` limited to loopback (`localhost`, `127.0.0.1`), and raise the dependency floor to `mcp>=1.23.0`.

**Rationale:** CVE-2025-66416 ([GHSA-9h52-p55h-vw2f](https://github.com/advisories/GHSA-9h52-p55h-vw2f)): the MCP Python SDK did not enable DNS rebinding protection by default (fixed in 1.23.0). It affects only `sse` and `streamable-http`; the default `stdio` transport opens no network port. The SDK now enables the protection automatically, but only when the host is `127.0.0.1`, `localhost` or `::1` ([`fastmcp/server.py`](https://github.com/modelcontextprotocol/python-sdk/blob/v1.23.0/src/mcp/server/fastmcp/server.py)). MLMS defaults `mcp_host` to `0.0.0.0`, so the setting has to be explicit. `allowed_origins` is required too, because the SDK validates `Origin` inside the same gate ([`transport_security.py`](https://github.com/modelcontextprotocol/python-sdk/blob/v1.23.0/src/mcp/server/transport_security.py)). The floor guards against a re-resolve of `uv.lock` (currently `1.27.2`) going below the fixed version.

**Consequence:** if `MCP_HOST` points outside loopback (LAN, VPS, domain), every request gets `421 Invalid Host Header` until that host is added to both lists in `server.py`. This is deliberate; the same confusion is reported in other MCP servers ([blockscout/mcp-server #333](https://github.com/blockscout/mcp-server/issues/333)). A regression test (`tests/unit/test_server_transport_security.py`) fails if the protection is turned off again.

**Alternatives:** a wider allowlist for hypothetical proxy or domain deployments was rejected as speculative, since adding a host later is a one-line change.

**Affected files:** `src/mlms/server.py`, `pyproject.toml`, `uv.lock`, `tests/unit/test_server_transport_security.py`

**See also:** Decision 9.

---

## Decision 5: Single `timescaledb-ha` image for PostgreSQL

**Decided:** 2026-05-08

**Last updated:** 2026-08-29

**Decision:** `docker-compose.yml` uses one image, `timescale/timescaledb-ha:pg16`, instead of `postgres` plus separately installed extensions.

**Consequence:** the data directory in this image is `/home/postgres/pgdata/data`, not the stock `/var/lib/postgresql/data`. Switching to another Postgres image means changing the volume mapping too.

**Rationale:** the image already ships TimescaleDB and pgvector (the [Dockerfile](https://github.com/timescale/timescaledb-docker-ha/blob/master/Dockerfile#L186) installs `postgresql-${pg}-pgvector`), so there are no init scripts to mount and no extensions to compile.

**Affected files:** `docker-compose.yml`

---

## Decision 6: Conflict detection by cosine distance, inside a transaction

**Decided:** 2026-05-10

**Last updated:** 2026-08-29

**Decision:** when a fact is written, its embedding is compared with the nearest `is_current=TRUE` fact of the same `project_id`. If the cosine distance is at most `1 - COSINE_CONFLICT_THRESHOLD`, the old row gets `is_current=FALSE` and the new row is inserted with `version + 1` and `replaces_id` pointing at the old row. Both statements run in one transaction.

**Known limitation:** the check and the insert are separate statements at the default `READ COMMITTED` isolation. Two concurrent `memorize()` calls with near-identical content can both see no conflict and both insert an `is_current=TRUE` row. The transaction does not close that window.

**Rationale:** the transaction makes retire-and-insert atomic: a failure between the two statements cannot leave the old fact retired with no replacement.

**Threshold:** `COSINE_CONFLICT_THRESHOLD = 0.96` is a starting point that has not been calibrated against real data.

**Affected files:** `src/mlms/storage/facts.py` (`upsert_fact`), `src/mlms/config.py`

**See also:** Decision 10.

---

## Decision 9: `MCP_TRANSPORT` switch between stdio and HTTP

**Decided:** 2026-05-16

**Last updated:** 2026-08-29

**Decision:** `MCP_TRANSPORT` (default `stdio`) selects the transport in `server.py`. For `sse` and `streamable-http` the server also reads `MCP_HOST` / `MCP_PORT`.

**Rationale:** `stdio` is enough when the MCP client launches the server as a child process, and it opens no network port. HTTP is needed when the client cannot start the server itself, for example a server on another machine.

**Affected files:** `src/mlms/config.py`, `src/mlms/server.py`

**See also:** Decision 4 (DNS rebinding applies only to the HTTP transports).

---

## Decision 10: `revert_fact` follows the `replaces_id` chain and blocks cross-project reverts

**Decided:** 2026-05-20

**Last updated:** 2026-08-29

**Decision:** each fact version stores `replaces_id`, a reference to the version it replaced (column added in migration `007`, set on conflict as in Decision 6). `revert_fact()` retires the given fact and restores its predecessor in one transaction, one step back. It first checks that both versions have the same `project_id` and refuses otherwise.

**Rationale:** chains are built inside one project, so a mismatch can only come from inconsistent data (a manual edit or a bug). The check stops a revert from reactivating a fact in another project in that case. The tool takes only `fact_id`, so it does not check that the caller owns the fact.

**Affected files:** `alembic/versions/007_add_project_id_session_log_replaces_id_facts.py`, `src/mlms/tools/revert_fact.py`

**See also:** Decision 6.

---

## Decision 14: Keyset pagination on `(time, id)` in `get_events()`

**Decided:** 2026-05-13

**Last updated:** 2026-08-30

**Decision:** the cursor is the pair `(time, id)` with `WHERE time > %s OR (time = %s AND id > %s)`, ordered by `time, id`, instead of `time` alone.

**Rationale:** timestamps are not unique. With a `time`-only cursor, events sharing a timestamp across a page boundary are skipped or repeated. `id` gives a strict tie-break.

**Known limitation:** pagination is implemented and unit-tested in `storage/timeline.py`, but the `get_timeline` tool does not use it. It runs its own SQL without a cursor, so the tool cannot return more than `limit` events. This is a limit of the tool, not of the pagination code; it is recorded in `docs/known-issues.md` (#13).

**Affected files:** `src/mlms/storage/timeline.py` (`get_events`)

---

## Decision 15: `memorize(content)` length limit: hard 10,000 characters, soft 500

**Decided:** 2026-08-30

**Decision:** `content` above `MEMORY_CONTENT_MAX_LENGTH = 10_000` characters is rejected, not truncated. The check runs in the tool signature through a `BeforeValidator`, so the error reports the actual length and the limit; `Field(max_length=...)` stays alongside so the limit also appears in the tool's JSON Schema. The tool docstring recommends about 500 characters per fact (`MEMORY_CONTENT_RECOMMENDED_LENGTH`, guidance only), so composite facts are split across several calls.

**Rationale:** `content` had no limit, and for facts the whole text is sent to the embedding API. At an assumed 1.5 characters per token for Cyrillic (an estimate, not measured with the real tokenizer), 10,000 characters is about 6,700 tokens. That is far below the 32k context of both Qwen3-Embedding models ([8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B), [0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)) and within the `max_length = 8192` used in both model cards' usage examples. The margin is deliberate: the fallback `qwen3-embedding-0.6b` is intended for local serving (not implemented yet), where the usable context is set by the server.

**Alternatives:**
- Truncate silently: loses part of a fact without warning, while an error lets the agent shorten or split the text.
- Different limits per `memorize()` type: only `fact` content and the `session_log` label (already capped at 100 characters) reach the embedding API, so `event` and `skill` are not restricted in practice, and per-type limits would only add complexity.

**Affected files:** `src/mlms/server.py` (`_check_content_length`, `_MemoryContent`, `memorize` signature and docstring), `src/mlms/config.py`, `tests/unit/test_memorize_content_limit.py` (new)

---
