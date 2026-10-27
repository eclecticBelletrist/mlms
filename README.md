# MLMS: Multi-Layer Memory System

A self-hosted [MCP](https://modelcontextprotocol.io) server giving LLM agents persistent,
cross-session memory across four [CoALA](https://arxiv.org/abs/2309.02427)-style layers:
semantic facts, episodic timeline events, procedural skills, and session-scoped working memory.
Text to embed is sent to an OpenAI-compatible embeddings API, OpenRouter by default.

[![CI](https://github.com/eclecticBelletrist/mlms/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/eclecticBelletrist/mlms/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)

---

## What this is

With MLMS connected, the agent looks up what earlier chats stored before it answers, and stores
new values, decisions and procedures as they come up, without being told to remember (see the
[Walkthrough](#walkthrough)).

The agent uses MLMS only through MCP tool calls and never touches the database directly. There
are four memory layers:

- Semantic, `facts`: versioned, with embedding-based conflict detection and soft delete
  (`is_current`). Old versions stay in the database; the tools return the current one, and
  `revert_fact` restores the previous one.
- Episodic, `timeline_events` and `session_log`: events at project scale (a TimescaleDB
  hypertable) and entries at chat scale (in-session and cross-session retrieval).
- Procedural, `skills`: versioned, never deleted.
- Working, a per-chat session record in Redis with a 48-hour TTL, renewed on every access. It
  holds the project and the time of the last write; its `open_todos`, `active_entities` and
  `recent_decisions` fields are reserved and always empty.

Everything durable is stored in Postgres (pgvector + TimescaleDB). Redis holds the
session-scoped working memory and a 24-hour cache of computed embeddings.

MLMS is built for self-hosted, single-user use: one person with a few trusted clients running
their own instance. It is not a multi-tenant service. The default `stdio` transport has no
network port and no auth. The HTTP transports (`sse`, `streamable-http`) are protected by one
shared secret (see "Running over HTTP" below). Every holder gets equal, full read/write access,
so there is no per-user auth or isolation (see [`KEY_DECISIONS.md`](KEY_DECISIONS.md),
Decision 23). Adding them would be a redesign.

MLMS is experimental (version 0.1). Conflict detection compares embeddings by cosine
similarity, but its threshold is not independently calibrated yet, and there is no forgetting
or decay mechanism.
Known limitations are listed in [`docs/known-issues.md`](docs/known-issues.md); read it before
relying on this for anything beyond experimentation.

## Quick start

Requires Docker (for Postgres + Redis), Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/eclecticBelletrist/mlms.git
cd mlms

# Postgres + Redis; all stored memory is in two volumes (docker compose down -v deletes them)
docker compose up -d --wait

# app: installs the versions pinned in uv.lock into .venv
uv sync --locked --extra dev
source .venv/bin/activate
alembic upgrade head
```

Without uv, `pip install -e ".[dev]"` in a virtualenv of your own installs the same package, but
resolves dependency versions from PyPI instead of `uv.lock`.

Default transport is `stdio`: the MCP client launches the server as a subprocess and talks to it
over stdin/stdout. Started by hand in a terminal, it just waits for input and prints nothing
(apart from a warning if `EMBEDDING_API_KEY` is empty). The client starts it in its own working
directory, so a `.env` in the clone is not read: pass the key in the client config.

Claude Code:

```bash
claude mcp add mlms -e EMBEDDING_API_KEY=your-key-here -- /path/to/mlms/.venv/bin/python -m mlms.server
```

Claude Desktop (`claude_desktop_config.json`) and other clients that use the `mcpServers` format:

```json
{
  "mcpServers": {
    "mlms": {
      "command": "/path/to/mlms/.venv/bin/python",
      "args": ["-m", "mlms.server"],
      "env": {
        "EMBEDDING_API_KEY": "your-key-here",
        "DATABASE_URL": "postgresql+psycopg://mlms:mlms@localhost:5432/mlms",
        "REDIS_URL": "redis://localhost:6379/0"
      }
    }
  }
}
```

`command` must be the full path to the Python that has MLMS installed (`.venv/bin/python` in the
clone after `uv sync`). A plain `python` starts the system interpreter, which does not have the
package.

**What leaves your machine.** The embedding API receives the content of every fact and
`session_log` label and every semantic search query (`about`, `query`, `semantic_query`). Events
and skills are not embedded. Apart from this API, the server connects only to Postgres and Redis.

<details>
<summary>Environment variables</summary>

Read from the environment or from `.env` in the working directory (template:
[`.env.example`](.env.example)). Defaults are in `src/mlms/config.py`.

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_API_KEY` | empty | Key for the embedding API. Empty: facts, `session_log` and semantic search fail; events, skills and `get_timeline` work |
| `EMBEDDING_API_BASE` | `https://openrouter.ai/api/v1` | Embedding endpoint, called as `<base>/embeddings` |
| `DATABASE_URL` | `postgresql+psycopg://mlms:mlms@localhost:5432/mlms` | Postgres; must use the `postgresql+psycopg://` scheme |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis |
| `MCP_TRANSPORT` | `stdio` | `stdio`, `sse` or `streamable-http` |
| `MCP_HOST` | `0.0.0.0` | Bind address for the HTTP transports |
| `MCP_PORT` | `8000` | Port for the HTTP transports |
| `MCP_SHARED_SECRET` | unset | Bearer token for the HTTP transports; required when `MCP_HOST` is not loopback |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | `mlms` | Used by `docker-compose.yml` only |

</details>

## Walkthrough

Two prompts that show the main behavior. Send each in a new chat, so the agent has nothing to go
on but memory. Neither asks the agent to remember anything: the server's instructions
(`KEY_DECISIONS.md`, Decision 30) tell it when to read and write memory. Facts and searches need
a working `EMBEDDING_API_KEY`.

**Chat 1.** The agent stores a fact, a decision and a skill.

> I'm working on the billing-service project. Today we decided to move payment notifications
> from REST polling to webhooks. Failed webhook deliveries will be retried 5 times with
> exponential backoff. A trick I always use when debugging webhooks: replay the stored payload
> with `stripe events resend <id>` instead of waiting for a new event. Can you sketch the retry
> schedule for me, assuming a 30-second base delay?

**Chat 2.** The agent changes the fact, which retires the old version, and recalls the decision
and the skill from the first chat.

> billing-service update: failed webhook deliveries are now retried 8 times instead of 5. Remind
> me too: what did we decide about payment notifications, and how do I usually debug a webhook?

The old version is retired, not deleted:

```bash
docker compose exec postgres psql -U mlms -d mlms \
  -c "select version, is_current, content from facts order by created_at"
```

```text
 version | is_current | content   (other facts trimmed)
---------+------------+-----------------------------------------------------------------------
       1 | f          | Failed webhook deliveries are retried 5 times with exponential backoff.
       2 | t          | Failed webhook deliveries are retried 8 times with exponential backoff.
```

A new fact replaces an old one when its embedding is at least 0.96 similar to a current fact in
the same project. That threshold is not calibrated, so an update worded differently from the
original can leave both current; this is why the instructions ask the agent to reuse the old
wording. The tools return only the current version of a fact. If an update was a mistake, tell
the agent: it calls `revert_fact` and the previous version is current again.

A client that does not pass server instructions to the agent needs the same text in its own
system prompt, or an explicit "remember this".

## Architecture

```mermaid
graph TD
  LLM[LLM Client] -->|MCP JSON-RPC only| Server[MCP Server]
  Server --> EmbAPI[Embedding API\nQwen3-Embedding-8B]
  Server --> PG[(PostgreSQL\npgvector + TimescaleDB)]
  Server --> RD[(Redis)]

  PG --> F[facts]
  PG --> TE[timeline_events]
  PG --> SL[session_log]
  PG --> SK[skills]
  RD --> SC[session context]
```

## Tools

10 tools, registered over MCP `tools/list`:

| Tool | Kind | What it does |
|---|---|---|
| `create_project` | write | Get or create a project by name. Call this first, once per project |
| `memorize` | write | Store a fact, event, skill, or session_log entry |
| `get_facts` | read (embeds on `about`) | Retrieve current facts, optional semantic query |
| `get_timeline` | read | Query timeline events (phases, actions, decisions) |
| `get_session_log` | read (embeds on `semantic_query`) | Session log entries, in-session or cross-session (mutually exclusive modes) |
| `get_session_context` | read | Working memory for a chat session, from Redis. `open_todos`, `active_entities` and `recent_decisions` are always empty: no tool fills them |
| `get_skills` | read | Procedural skills, optional domain filter |
| `revert_fact` | write | Revert a fact to its previous version (soft, `is_current` flip) |
| `search_memory` | read (embeds unless `memory_type` scopes to event/skill) | Cross-layer search over facts + timeline + skills |
| `export_project` | read | Export all current facts, timeline events, and skills for a project as JSON |

## Measured

Latency, `scripts/measure_latency.py`, 100 calls per operation, Docker Desktop on Windows 11:

| Operation | p50 | p99 | Target |
|---|---|---|---|
| `get_facts`, query embedding already cached | 4.7 ms | 16.0 ms | < 200 ms, met |
| `get_session_context` (Redis) | 1.6 ms | 9.9 ms | ≤ 5 ms, missed by one slow call of 100 |
| embed through the API and store a fact | 1158 ms | 9237 ms | none set |

Retrieval on `synthetic_hard`: 93 scenarios over a shared corpus of 400 facts, each with 2 to 5
relevant facts and a query worded so that word matching fails. Recall@5 is the average share of
the relevant facts that land in the top 5:

| Retriever, in memory without Postgres | Recall@5 |
|---|---|
| Cosine search, Qwen3-Embedding-8B at 1536 dimensions | 0.320 |
| BM25 on the same queries | 0.040 |

The dataset is LLM-generated, so the numbers only compare the methods with each other and say
nothing about quality on real data. In particular it cannot show whether Session-Augmented RAG
(adding the chat's session log to the query) helps: the session logs were written to contain the
facts' words, and BM25 gains almost as much from them as the embedding model does. How the
numbers were measured, the per-type results and the limits are in
[`docs/mlms_benchmark_design.md`](docs/mlms_benchmark_design.md).

## Testing

```bash
pytest -m unit                              # ~4 s, no external services
MLMS_INTEGRATION=1 pytest -m integration    # Postgres + Redis from Quick start, migrated
```

32 test files, split into unit and integration suites. CI (`.github/workflows/ci.yml`) runs
on every push and PR to `main`: lint (ruff), types (strict mypy), the docs consistency check
(`scripts/check_docs.py`), the migrations, the unit suite with the coverage gate
(`fail_under = 80`), the integration suite against real Postgres + Redis `services:` containers,
and a Docker image build.

## Running over HTTP

`MCP_TRANSPORT=sse` or `streamable-http` serves MCP over HTTP instead of `stdio`.
`docker compose --profile app up -d --build` runs the server this way in a container and applies
the migrations on start. Over HTTP:

- with `MCP_SHARED_SECRET` set, every request must carry `Authorization: Bearer <secret>`, or
  it gets `401`;
- on a non-loopback `MCP_HOST`, including the `0.0.0.0` the container uses, the server refuses
  to start without the secret;
- the `Host` and `Origin` headers are checked against a loopback allowlist (`421`, `403`).

The secret is one static token that gives every holder the same full access. Firewall, tunnel
and reverse-proxy setup, `curl` checks for these responses, and backup and restore are in
[`docs/deployment.md`](docs/deployment.md).

## Design decisions & known limitations

- [`KEY_DECISIONS.md`](KEY_DECISIONS.md): non-obvious architectural decisions, alternatives
  rejected, trade-offs
- [`docs/known-issues.md`](docs/known-issues.md): known, accepted-but-unfixed limitations

## To do

- [ ] Install with one command, with setup for Claude Code and other clients (OpenCode, Codex
  and others)
- [ ] Support more embedding models: Qwen3-Embedding-0.6B and a few others
- [ ] Rework how session-augmented search (SAR) is used and re-run its tests on data that does
  not leak the answer
- [ ] Rethink the four memory layers, working memory in particular
- [ ] Run public memory benchmarks such as LoCoMo

## License

[Apache-2.0](LICENSE).
