# MLMS — Multi-Layer Memory System

A self-hosted [MCP](https://modelcontextprotocol.io) server giving LLM agents persistent,
cross-session memory across four CoALA-style layers: semantic facts, episodic timeline
events, procedural skills, and session-scoped working memory.

Python 3.11+ · Apache-2.0 · early-stage, pre-1.0

---

## What this is

MLMS is a memory backend an LLM agent talks to exclusively through MCP tool calls — the
agent never touches the database directly. It stores four kinds of memory:

- **facts** — versioned semantic memory, embedding-based conflict detection, soft-delete
  (`is_current`) so history is never lost
- **timeline_events** — episodic memory at project scale, backed by a TimescaleDB hypertable
- **session_log** — episodic memory at chat scale, in-session and cross-session retrieval
- **skills** — procedural memory, versioned, never deleted

Storage is Postgres (pgvector + TimescaleDB) for everything durable, Redis for
session-scoped working memory.

This is built for **self-hosted, single-/few-user use** — one person or a small team
running their own instance, not a multi-tenant SaaS. There's no auth layer on the HTTP
transport (see [`KEY_DECISIONS.md`](KEY_DECISIONS.md)) and no per-user isolation; if that
positioning ever changes, that's a real redesign, not a toggle.

It's early-stage. Cosine-similarity conflict detection exists but its threshold isn't
independently calibrated yet, and there's no forgetting/decay mechanism. Known limitations
are tracked honestly in [`docs/known-issues.md`](docs/known-issues.md) — read it before
relying on this for anything beyond experimentation.

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

## Quick start

Requires Docker (for Postgres + Redis) and Python 3.11+.

```bash
git clone <this-repo>
cd mlms

# infrastructure
docker compose up -d

# app
pip install -e ".[dev]"
alembic upgrade head

# embedding API key — see src/mlms/config.py for all settings
export EMBEDDING_API_KEY=your-key-here

python -m mlms.server
```

Default transport is `stdio` — the MCP client launches the server as a subprocess. To
connect from Claude Desktop, add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mlms": {
      "command": "python",
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

HTTP transports (`sse`, `streamable-http`) are supported via `MCP_TRANSPORT` but have no
auth layer — see the positioning note above before exposing one off `localhost`.

## Deployment status

The `docker compose up -d` above starts **development infrastructure only** — Postgres and
Redis with plaintext local credentials, ports open on the host. There is no production
deployment path yet: no Dockerfile for the MLMS server itself, no secrets management, no
hardened compose profile. That work hasn't started — it's tracked as upcoming, not published
prematurely. Don't run this setup as anything but a local dev instance.

## Tools

9 tools, registered over MCP `tools/list`:

| Tool | Kind | What it does |
|---|---|---|
| `memorize` | write | Store a fact, event, skill, or session_log entry |
| `get_facts` | read (embeds on `about`) | Retrieve current facts, optional semantic query |
| `get_timeline` | read | Query timeline events (phases, actions, decisions) |
| `get_session_log` | read (embeds on `semantic_query`) | Session log entries — in-session or cross-session, mutually exclusive modes |
| `get_session_context` | read | Working memory for a chat session, from Redis |
| `get_skills` | read | Procedural skills, optional domain filter |
| `revert_fact` | write | Revert a fact to its previous version (soft, `is_current` flip) |
| `search_memory` | read (embeds) | Cross-layer search over facts + timeline + skills |
| `export_project` | read | Export all current facts, timeline events, and skills for a project as JSON |

## Testing

```bash
pytest -m unit                 # fast, no external services
docker compose up -d && pytest -m integration   # needs Postgres + Redis
```

262 test functions across 24 files, unit + integration + benchmark suites. Coverage gate
(`fail_under = 80`) and strict mypy are configured in `pyproject.toml` but run locally only
— there's no CI wired up yet.

## Design decisions & known limitations

- [`KEY_DECISIONS.md`](KEY_DECISIONS.md) — non-obvious architectural decisions, alternatives
  rejected, trade-offs
- [`docs/known-issues.md`](docs/known-issues.md) — known, accepted-but-unfixed limitations

## License

[Apache-2.0](LICENSE).
