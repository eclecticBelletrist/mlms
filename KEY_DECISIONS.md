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

**Last updated:** 2026-08-25

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
