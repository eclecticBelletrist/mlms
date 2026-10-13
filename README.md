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
session-scoped working memory. A session record holds five fields (`project_id`, `last_activity`,
`open_todos`, `active_entities`, `recent_decisions`), but the tools only write the first two; the
three lists stay empty because no tool fills them yet.

This is built for **self-hosted, single-user use** — one person with a few trusted clients
running their own instance, not a multi-tenant SaaS. The default `stdio` transport has no
network port and no auth at all. `sse`/`streamable-http` have a single shared-secret gate
(`MCP_SHARED_SECRET`; optional on loopback, the server refuses to start without it on any other
bind, see "Homelab/VPS hardening" below) — every holder
gets equal, full read/write access, so this is not per-user auth or isolation in any real
sense (see [`KEY_DECISIONS.md`](KEY_DECISIONS.md), Decision 23). If that positioning ever
changes, that's a real redesign, not a toggle.

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
docker compose up -d --wait

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

HTTP transports (`sse`, `streamable-http`) are supported via `MCP_TRANSPORT` and are protected by
the optional shared-secret gate — see "Homelab/VPS hardening" below before exposing one off
`localhost`.

## Deployment status

`docker compose --profile app up -d --build` brings up the full stack — Postgres, Redis, and the
MLMS server itself (`mlms` service), talking to each other over the compose network. A plain
`docker compose up -d` starts only Postgres and Redis, which is all the `stdio` setup in Quick
start needs. The `mlms` service refuses to start until `MCP_SHARED_SECRET` is set in `.env` (see
[`.env.example`](.env.example)); `docker compose logs mlms` shows why. The server
container runs `alembic upgrade head` before starting, and speaks `streamable-http` on
`:8000` (a detached container has no stdin to serve `stdio` on — see
[`KEY_DECISIONS.md`](KEY_DECISIONS.md), Decision 20). You build the image yourself
(`docker build .`); it isn't published to a registry.

That said, this is still a desktop/local setup, not a hardened production one: `postgres`
credentials are now `.env`-driven (`POSTGRES_USER`/`POSTGRES_PASSWORD`/`POSTGRES_DB` in
`.env.example`), but the shipped default is still `mlms`/`mlms` — nothing changes out of the
box, only the ability to change it. `redis` has no authentication at all, hardcoded or
otherwise. Both are published on the host's loopback (`127.0.0.1`) only, so they are not
reachable from the network; the optional `mlms` service publishes `:8000` on all interfaces. The
defaults are for a local setup; for a homelab or VPS follow "Homelab/VPS hardening" below.

## Backup and restore

This shows you how to take a point-in-time snapshot of the two named volumes
(`postgres_data`, `redis_data`) and put one back — it's a manual save point you run
yourself, not a backup schedule.

**Postgres.** Use the custom dump format and `pg_restore`, not a plain-text dump — Timescale's
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

Skip `-j` on `pg_restore` — parallel restore is known to cause foreign-key errors when
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

Everything below only matters if you set `MCP_TRANSPORT=sse` or `streamable-http` — the
default `stdio` transport doesn't listen on the network at all, so none of this applies to it.

**Layer 1 — don't rely on the app alone to keep the port closed.** `MCP_HOST` defaults to
`0.0.0.0`, and the server's `allowed_hosts`/`allowed_origins` allowlist (see
[`KEY_DECISIONS.md`](KEY_DECISIONS.md), Decision 4) only rejects requests at the application
layer with a `421` — the TCP port itself is still open to the world underneath that check.
Set `MCP_HOST=127.0.0.1` explicitly and add a firewall rule (`ufw`/`iptables`) denying inbound
traffic on `MCP_PORT` from outside — a second layer that still holds even if someone later
changes `MCP_HOST` and forgets why.

**Layer 2 — recommended: don't open a public port at all.** [Tailscale](https://tailscale.com/kb/1080/cli)
or [Cloudflare Tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/)
let a client reach the server over a private mesh or an outbound-only connection, with
`MCP_HOST` staying `127.0.0.1`. Nothing below replaces this — a shared secret narrows *who*
can use the server once a request arrives, not who can reach the port at all.

**Layer 1.5 — shared-secret token, `MCP_SHARED_SECRET`.** Only checked on `sse`/
`streamable-http`; `stdio` has no network port, so it never applies there. Generate one with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Set it as `MCP_SHARED_SECRET` in `.env`, and have every client send it back as
`Authorization: Bearer <value>` on every request — not as a URL query parameter (query strings
end up in proxy/access logs). A missing or wrong token gets `401`, checked before the request
reaches any tool.

If `MCP_HOST` is a loopback address (`127.0.0.1`/`localhost`/`::1`), the secret stays optional —
this matches the project's original single-user, local-by-default posture (see
[`KEY_DECISIONS.md`](KEY_DECISIONS.md), Decision 23).
If `MCP_HOST` is anything else (`0.0.0.0`, a LAN/VPN IP) — exactly the case this section is
about — the server refuses to start at all without `MCP_SHARED_SECRET` set, rather than
listening unauthenticated.

This is a single static token: no rotation, no per-user scoping, no rate limiting. It closes the
gap where anyone who can reach the port gets full read/write access to every memory. Layers 1, 2
and 3 still do the real perimeter work; this only adds a check on top of them.

Three things it does **not** protect against, worth knowing rather than assuming away:

- **`MCP_HOST=127.0.0.1` behind a reverse proxy that terminates publicly is network-reachable
  even though the process binds loopback.** The fail-fast above only looks at the `MCP_HOST`
  value the process itself sees — it can't see a proxy in front of it. Same blind spot as the
  `Host`/`Origin` check two paragraphs below, just for the new gate instead of the old one.
- **The shared secret gives every holder equal access.** All clients use the same token, so any
  holder can read or replay another holder's session; the secret only walls off the server from
  outsiders, and does nothing to separate clients from each other on the inside. Look elsewhere
  if you need that boundary.
- **`sse` + a browser's built-in `EventSource` API don't mix with this.** `EventSource` has no
  way to set custom request headers, so it can't send `Authorization: Bearer`. Every connection
  attempt from such a client gets `401`. This isn't fixable by moving the token into the URL —
  that's the query-string leak this section already tells you to avoid. If your `sse` client is
  a browser tab making a raw `EventSource` connection (not an MCP client library that controls
  its own headers), this gate isn't compatible with it.

`allowed_hosts`/`allowed_origins` validate the `Host`/`Origin` **headers** of the incoming
request, not which interface the connection came in on. This does not work out of the box with
a tunnel or mesh tool unless it forwards the original public `Host`/`Origin` through to your
local server, which depends entirely on its own config (e.g. Cloudflare Tunnel's
`originRequest.httpHostHeader`). This was not verified end-to-end against a live Cloudflare
Tunnel or Tailscale Serve setup. Check your tool's docs and test it yourself (see verification
step below) before trusting it; don't assume the public hostname is silently rewritten to
`localhost` on the way in.

**Layer 3 — reverse proxy with a real domain and TLS**, if you need one instead of a tunnel:
put nginx or caddy in front, have it terminate TLS and proxy to the MCP server listening on
`127.0.0.1`. This repo doesn't ship an nginx/caddy config — only the pattern. Whichever proxy
you use, you still have to add every hostname/IP/domain your clients will actually connect
through to `_LOOPBACK_HOSTS` in `src/mlms/server.py` (it feeds both `allowed_hosts` and
`allowed_origins`), or every request gets `421 Invalid Host Header` — see Decision 4 for why
that list exists and Decision 22 for why it isn't wired up to an env var.

**Verify it's actually working** — this was run against the `mlms` service started with
`docker compose --profile app up` (Docker 29, rootless and rootful), not assumed from reading
the SDK source. In a request the shared-secret check runs first and the Host/Origin check
after it, on the transport's own route — `/mcp` for `streamable-http`, `/sse` for `sse` —
not on `/`, so a plain `GET /` won't exercise it. You're only checking the status code, not a
full MCP handshake, so a non-`421`/`403` code on a "good" request (`406` in local testing) is
expected and fine — Host and Origin fail with *different* codes.

The `docker-compose.yml` `mlms` service binds `0.0.0.0`, which makes `MCP_SHARED_SECRET`
mandatory (Decision 23) — the shared-secret middleware runs *before* the Host/Origin check
below it, so every command here needs `Authorization: Bearer <your-secret>` or you'll get a
blanket `401` regardless of Host/Origin and conclude nothing about the check you're actually
testing. Testing a bare `MCP_HOST=127.0.0.1` run instead (no secret required there) — drop
that header:

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

The allowlist match is exact and case-sensitive, with no host normalization — only the literal
forms above (`127.0.0.1`/`localhost`, lowercase, no trailing dot) are covered. A client that
happens to resolve `localhost` to the IPv6 loopback (`::1`), or sends `LOCALHOST`/`localhost.`,
will get rejected even though the request is legitimate. That's a fail-safe rejection, harmless
on its own, but worth knowing if a "why is my local client getting 421" mystery ever comes up.

## Tools

10 tools, registered over MCP `tools/list`:

| Tool | Kind | What it does |
|---|---|---|
| `create_project` | write | Get or create a project by name — call this first, once per project |
| `memorize` | write | Store a fact, event, skill, or session_log entry |
| `get_facts` | read (embeds on `about`) | Retrieve current facts, optional semantic query |
| `get_timeline` | read | Query timeline events (phases, actions, decisions) |
| `get_session_log` | read (embeds on `semantic_query`) | Session log entries — in-session or cross-session, mutually exclusive modes |
| `get_session_context` | read | Working memory for a chat session, from Redis |
| `get_skills` | read | Procedural skills, optional domain filter |
| `revert_fact` | write | Revert a fact to its previous version (soft, `is_current` flip) |
| `search_memory` | read (embeds unless `memory_type` scopes to event/skill) | Cross-layer search over facts + timeline + skills |
| `export_project` | read | Export all current facts, timeline events, and skills for a project as JSON |

## Testing

```bash
pytest -m unit                                                    # fast, no external services
docker compose up -d --wait && alembic upgrade head && MLMS_INTEGRATION=1 pytest -m integration  # needs Postgres + Redis
```

295 test functions across 32 files, unit + integration suites. CI
(`.github/workflows/ci.yml`) runs lint (ruff), types (mypy), and both the unit and
integration suites against real Postgres + Redis `services:` containers on every push and
PR to `main`. Coverage gate (`fail_under = 80`) and strict mypy are enforced there too.

## Design decisions & known limitations

- [`KEY_DECISIONS.md`](KEY_DECISIONS.md) — non-obvious architectural decisions, alternatives
  rejected, trade-offs
- [`docs/known-issues.md`](docs/known-issues.md) — known, accepted-but-unfixed limitations

## License

[Apache-2.0](LICENSE).
