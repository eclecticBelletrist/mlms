# MLMS: Multi-Layer Memory System

A self-hosted [MCP](https://modelcontextprotocol.io) server giving LLM agents persistent,
cross-session memory across four CoALA-style layers: semantic facts, episodic timeline
events, procedural skills, and session-scoped working memory.

Python 3.11+ · Apache-2.0 · early-stage, pre-1.0

---

## What this is

An LLM agent uses MLMS only through MCP tool calls and never touches the database directly.
It has four memory layers:

- Semantic, `facts`: versioned, with embedding-based conflict detection and soft delete
  (`is_current`), so history is never lost.
- Episodic, `timeline_events` and `session_log`: events at project scale (a TimescaleDB
  hypertable) and entries at chat scale (in-session and cross-session retrieval).
- Procedural, `skills`: versioned, never deleted.
- Working, a per-chat session record in Redis.

Everything durable is stored in Postgres (pgvector + TimescaleDB); Redis holds only the
session-scoped working memory.

MLMS is built for self-hosted, single-user use: one person with a few trusted clients running
their own instance. It is not a multi-tenant service. The default `stdio` transport has no
network port and no auth. `sse` and `streamable-http` have a single shared-secret gate
(`MCP_SHARED_SECRET`), optional on loopback; on any other bind the server refuses to start
without it (see "Homelab/VPS hardening" below). Every holder gets equal, full read/write access,
so there is no per-user auth or isolation (see [`KEY_DECISIONS.md`](KEY_DECISIONS.md),
Decision 23). Adding them would be a redesign.

Conflict detection compares embeddings by cosine similarity, but its threshold is not
independently calibrated yet, and there is no forgetting or decay mechanism. Known limitations
are listed in [`docs/known-issues.md`](docs/known-issues.md); read it before relying on this
for anything beyond experimentation.

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
docker compose up -d --wait

# app
pip install -e ".[dev]"
alembic upgrade head

# embedding API key — see src/mlms/config.py for all settings
export EMBEDDING_API_KEY=your-key-here

python -m mlms.server
```

Default transport is `stdio`: the MCP client launches the server as a subprocess and talks to it
over stdin/stdout. Started by hand in a terminal, it just waits for input and prints nothing
(apart from a warning if `EMBEDDING_API_KEY` is empty). To connect from Claude Desktop, add to
`claude_desktop_config.json`:

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

`command` must be the Python that has MLMS installed. If you installed it into a virtualenv, give
the full path to that environment's interpreter (for example `/path/to/mlms/.venv/bin/python`);
a plain `python` starts the system interpreter, which does not have the package.

HTTP transports (`sse`, `streamable-http`) are selected with `MCP_TRANSPORT` and protected by a
shared-secret gate (optional on loopback). See "Homelab/VPS hardening" below before exposing one
off `localhost`.

## Walkthrough

Once the server is connected to your agent (Quick start above), these calls show what it does. Ask
the agent to make each call with exactly these arguments. IDs and timestamps in your output will
differ; outputs below are trimmed to the relevant fields. Steps 2 to 5 embed text, so they need
a working `EMBEDDING_API_KEY`; without one only events, skills and `get_timeline` work.

**1. Create a project.** Everything else is scoped to the returned `project_id`.

```json
create_project  {"name": "billing-service", "description": "Billing backend"}
→ {"project_id": "4b4af135-…", "name": "billing-service", …}
```

**2. Store a fact and an event** (use your `project_id`).

```json
memorize  {"content": "Failed webhook deliveries are retried 5 times with exponential backoff.",
           "type": "fact", "metadata": {"project_id": "<project_id>", "chat_id": "session-1"}}
→ {"ok": true, "memory_id": "953207f6-…"}

memorize  {"content": "Billing moves from REST polling to webhooks", "type": "event",
           "metadata": {"project_id": "<project_id>", "title": "Webhooks replace polling",
                        "event_type": "decision", "chat_id": "session-1"}}
→ {"ok": true, "memory_id": "2f9f2f14-…"}
```

**3. Find the fact by meaning, from a new session.** Start a fresh conversation (the agent has
no memory of step 2) and ask in different words:

```json
get_facts  {"about": "how many times do we retry failed notifications?", "project_id": "<project_id>"}
→ [{"content": "Failed webhook deliveries are retried 5 times with exponential backoff.",
    "version": 1, "is_current": true, …}]
```

**4. Change the fact.** Store the same statement with a new value:

```json
memorize  {"content": "Failed webhook deliveries are retried 8 times with exponential backoff.",
           "type": "fact", "metadata": {"project_id": "<project_id>", "chat_id": "session-2"}}
→ {"ok": true, "memory_id": "360f7a80-…"}

get_facts  {"about": "how many times do we retry failed notifications?", "project_id": "<project_id>"}
→ [{"content": "Failed webhook deliveries are retried 8 times with exponential backoff.",
    "version": 2, "is_current": true, …}]
```

Only the current version comes back. The old row is retired, not deleted; see it in the database:

```bash
docker compose exec postgres psql -U mlms -d mlms \
  -c "select version, is_current, content from facts order by created_at"
```

Which new facts count as a change to an old one is decided by embedding similarity (at least
0.96 to a current fact in the same project). That threshold is not calibrated, so two
differently worded statements about the same thing can both stay current. The MCP tools return
only the current version of a fact; they do not show history.

**5. Search every layer at once.**

```json
search_memory  {"query": "webhooks", "filters": {"project_id": "<project_id>"}}
→ [{"layer": "fact",  "content": "Failed webhook deliveries are retried 8 times with …", …},
   {"layer": "event", "content": "Billing moves from REST polling to webhooks", …}]
```

Session-Augmented RAG (enriching a query with the session log before embedding) is a research
experiment measured in [`docs/mlms_benchmark_design.md`](docs/mlms_benchmark_design.md); it is not
part of the served tools.

## Deployment status

`docker compose --profile app up -d --build` starts the full stack: Postgres, Redis, and the
MLMS server itself (the `mlms` service), connected over the compose network. A plain
`docker compose up -d` starts only Postgres and Redis, which is all the `stdio` setup in Quick
start needs. The `mlms` service refuses to start until `MCP_SHARED_SECRET` is set in `.env` (see
[`.env.example`](.env.example)); `docker compose logs mlms` shows why. The server
container runs `alembic upgrade head` before starting and serves `streamable-http` on
`:8000` (a detached container has no stdin to serve `stdio` on; see
[`KEY_DECISIONS.md`](KEY_DECISIONS.md), Decision 20). You build the image yourself
(`docker build .`); it isn't published to a registry.

This is a desktop/local setup, not a hardened production one. `postgres` credentials come from
`.env` (`POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB`, see `.env.example`), but the shipped
default is still `mlms`/`mlms`. `redis` has no authentication. Both are published on the
host's loopback (`127.0.0.1`) only, so they are not reachable from the network; the optional
`mlms` service publishes `:8000` on all interfaces. For a homelab or VPS, follow
"Homelab/VPS hardening" below.

## Backup and restore

Manual snapshot and restore for the two named volumes (`postgres_data`, `redis_data`).
Nothing runs on a schedule.

**Postgres.** Use the custom dump format and `pg_restore`, not a plain-text dump: Timescale's
`pre_restore`/`post_restore` pair needs it to put hypertable metadata back correctly:

```bash
# snapshot
docker compose exec -T postgres pg_dump -U mlms -d mlms -Fc > mlms.bak

# restore — into an empty database, not the live one: it already has this schema in it,
# and pg_restore will refuse to recreate tables/constraints that already exist.
# DROP DATABASE fails while any connection is open, so stop the app first.
docker compose stop mlms
docker compose exec -T postgres psql -U mlms -d postgres -c "DROP DATABASE mlms;"
docker compose exec -T postgres psql -U mlms -d postgres -c "CREATE DATABASE mlms OWNER mlms;"
docker compose exec -T postgres psql -U mlms -d mlms -c "SELECT timescaledb_pre_restore();"
docker compose exec -T postgres pg_restore -U mlms -d mlms --no-owner < mlms.bak
docker compose exec -T postgres psql -U mlms -d mlms -c "SELECT timescaledb_post_restore();"
docker compose start mlms   # skip if you run the server on the host
```

Stop every other client that holds a connection to the database too (a psql session, a
host-run `python -m mlms.server`), or `DROP DATABASE` fails with "database is being accessed by
other users". `docker compose stop mlms` is harmless when the `app` profile is not running.

Skip `-j` on `pg_restore`: parallel restore is known to cause foreign-key errors when
restoring TimescaleDB's catalog tables.

**Redis.** `SAVE` blocks until the snapshot is written, so the file you copy out is always
complete:

```bash
# snapshot
docker compose exec redis redis-cli SAVE
docker compose cp redis:/data/dump.rdb ./dump.rdb

# restore
docker compose stop redis
docker compose cp ./dump.rdb redis:/data/dump.rdb
docker compose start redis
```

## Homelab/VPS hardening

Everything below only matters if you set `MCP_TRANSPORT=sse` or `streamable-http`. The
default `stdio` transport doesn't listen on the network at all, so none of this applies to it.

**Layer 1: don't rely on the app alone to keep the port closed.** `MCP_HOST` defaults to
`0.0.0.0`, and the server's `allowed_hosts`/`allowed_origins` allowlist (see
[`KEY_DECISIONS.md`](KEY_DECISIONS.md), Decision 4) only rejects requests at the application
layer (`421` for a bad Host, `403` for a bad Origin); the TCP port itself is still open to the world underneath that check.
Set `MCP_HOST=127.0.0.1` explicitly and add a firewall rule (`ufw`/`iptables`) denying inbound
traffic on `MCP_PORT` from outside. The firewall still holds even if someone later changes
`MCP_HOST` and forgets why.

The compose deploy works differently. Inside the container `MCP_HOST` has to stay `0.0.0.0`,
because with `127.0.0.1` Docker's port forwarding cannot reach the server. The `mlms` service
publishes `"8000:8000"` on every host interface, and Docker's published ports bypass `ufw`, so a
`ufw` rule does not close it. On a VPS, change the mapping to `"127.0.0.1:8000:8000"` and reach
the server through a tunnel or a reverse proxy on the same host (Layers 2 and 3).

**Layer 2, recommended: don't open a public port at all.** [Tailscale](https://tailscale.com/kb/1080/cli)
or [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)
let a client reach the server over a private mesh or an outbound-only connection, with
`MCP_HOST` staying `127.0.0.1`. Nothing below replaces this. A shared secret narrows who
can use the server once a request arrives, not who can reach the port.

**Layer 1.5: shared-secret token, `MCP_SHARED_SECRET`.** Only checked on `sse`/
`streamable-http`; `stdio` has no network port, so it never applies there. Generate one with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Set it as `MCP_SHARED_SECRET` in `.env`, and have every client send it as
`Authorization: Bearer <value>` on every request, not as a URL query parameter (query strings
end up in proxy/access logs). A missing or wrong token gets `401`, checked before the request
reaches any tool.

If `MCP_HOST` is a loopback address (`127.0.0.1`/`localhost`/`::1`), the secret stays optional,
matching the project's single-user, local-by-default posture (see
[`KEY_DECISIONS.md`](KEY_DECISIONS.md), Decision 23).
If `MCP_HOST` is anything else (`0.0.0.0`, a LAN/VPN IP), which is exactly the case this
section is about, the server refuses to start without `MCP_SHARED_SECRET` rather than listening
unauthenticated.

This is a single static token: no rotation, no per-user scoping, no rate limiting. It closes the
gap where anyone who can reach the port gets full read/write access to every memory. Layers 1, 2
and 3 still do the real perimeter work; the token is an extra check on top of them.

It does not protect against three things:

- **`MCP_HOST=127.0.0.1` behind a reverse proxy that terminates publicly is network-reachable
  even though the process binds loopback.** The fail-fast above only looks at the `MCP_HOST`
  value the process itself sees, so it can't see a proxy in front of it. The `Host`/`Origin`
  check below has the same blind spot.
- **The shared secret gives every holder equal access.** All clients use the same token, so any
  holder can read or replay another holder's session. The secret walls the server off from
  outsiders and does nothing to separate clients from each other. Look elsewhere if you need
  that boundary.
- **`sse` with a browser's built-in `EventSource` API doesn't work.** `EventSource` has no way
  to set custom request headers, so it can't send `Authorization: Bearer`, and every connection
  attempt from such a client gets `401`. Moving the token into the URL doesn't fix it, because
  that is the query-string leak this section already tells you to avoid. If your `sse` client is
  a browser tab making a raw `EventSource` connection (not an MCP client library that controls
  its own headers), this gate isn't compatible with it.

`allowed_hosts`/`allowed_origins` validate the `Host`/`Origin` headers of the incoming
request, not which interface the connection came in on. This does not work out of the box with
a tunnel or mesh tool unless it forwards the original public `Host`/`Origin` through to your
local server, which depends entirely on its own config (e.g. Cloudflare Tunnel's
`originRequest.httpHostHeader`). This was not verified end-to-end against a live Cloudflare
Tunnel or Tailscale Serve setup. Check your tool's docs and test it yourself (see verification
step below) before trusting it; don't assume the public hostname is silently rewritten to
`localhost` on the way in.

**Layer 3: reverse proxy with a real domain and TLS**, if you need one instead of a tunnel.
Put nginx or caddy in front, have it terminate TLS and proxy to the MCP server listening on
`127.0.0.1`. This repo doesn't ship an nginx/caddy config, only the pattern. Whichever proxy
you use, you still have to add every hostname/IP/domain your clients will actually connect
through to `_LOOPBACK_HOSTS` in `src/mlms/server.py` (it feeds both `allowed_hosts` and
`allowed_origins`), or every request gets `421 Invalid Host Header`. See Decision 4 for why
that list exists and Decision 22 for why it isn't wired up to an env var.

**Verify it's actually working.** This was run against the `mlms` service started with
`docker compose --profile app up` (Docker 29, rootless and rootful). In a request the
shared-secret check runs first and the Host/Origin check after it, on the transport's own
route (`/mcp` for `streamable-http`, `/sse` for `sse`), not on `/`, so a plain `GET /` won't
exercise it. You're only checking the status code, not a full MCP handshake, so a
non-`421`/`403` code on a "good" request (`406` in local testing) is expected and fine. Host
and Origin fail with different codes.

The `docker-compose.yml` `mlms` service binds `0.0.0.0`, which makes `MCP_SHARED_SECRET`
mandatory (Decision 23). The shared-secret middleware runs before the Host/Origin check, so
every command here needs `Authorization: Bearer <your-secret>`; without it you get a blanket
`401` regardless of Host/Origin and learn nothing about the check you're testing. To test a
bare `MCP_HOST=127.0.0.1` run instead (no secret required there), drop that header:

```bash
TOKEN=your-mcp-shared-secret   # only needed if MCP_HOST is non-loopback (e.g. the compose deploy)

# 421 — Host not on the allowlist
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $TOKEN" -H "Host: evil.example.com" http://<your-host>:<mcp-port>/mcp

# not 421 (406 in local testing) — Host passed the check
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $TOKEN" -H "Host: 127.0.0.1" http://<your-host>:<mcp-port>/mcp

# not 403 — a legitimate loopback Origin (with scheme — Origin headers always carry one)
# should pass, same as sending no Origin at all
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $TOKEN" -H "Host: 127.0.0.1" -H "Origin: http://127.0.0.1" http://<your-host>:<mcp-port>/mcp

# 403 — Origin present and not allowlisted (a different failure code from Host, not 421)
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $TOKEN" -H "Host: 127.0.0.1" -H "Origin: https://evil.example.com" http://<your-host>:<mcp-port>/mcp
```

The allowlist match is exact and case-sensitive, with no host normalization: only the literal
forms above (`127.0.0.1`/`localhost`, lowercase, no trailing dot) are covered. A client that
sends `Host: [::1]`, `LOCALHOST` or `localhost.` gets `421` even though the request is
legitimate. The same applies to `MCP_HOST=::1`: it counts as loopback for the shared-secret
check, but `[::1]` is not on the Host allowlist. The check fails toward rejecting a request, so
this is safe, but it explains a `421` from a local client.

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

## Testing

```bash
pytest -m unit                                                    # fast, no external services
docker compose up -d --wait && alembic upgrade head && MLMS_INTEGRATION=1 pytest -m integration  # needs Postgres + Redis
```

32 test files, split into unit and integration suites. CI (`.github/workflows/ci.yml`) runs
on every push and PR to `main`: lint (ruff), types (strict mypy), the docs consistency check
(`scripts/check_docs.py`), the migrations, the unit suite with the coverage gate
(`fail_under = 80`), the integration suite against real Postgres + Redis `services:` containers,
and a Docker image build.

## Design decisions & known limitations

- [`KEY_DECISIONS.md`](KEY_DECISIONS.md): non-obvious architectural decisions, alternatives
  rejected, trade-offs
- [`docs/known-issues.md`](docs/known-issues.md): known, accepted-but-unfixed limitations

## License

[Apache-2.0](LICENSE).
