from __future__ import annotations

import asyncio
import logging
import secrets
import sys
from collections.abc import AsyncIterator

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any, Literal, cast

import psycopg
import redis.asyncio as aioredis
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from psycopg_pool import AsyncConnectionPool, PoolTimeout
from pydantic import BeforeValidator, Field

from mlms.config import MEMORY_CONTENT_MAX_LENGTH, pg_dsn, settings
from mlms.tools.create_project import create_project as _create_project
from mlms.tools.export_project import export_project as _export_project
from mlms.tools.get_facts import get_facts as _get_facts
from mlms.tools.get_session_ctx import get_session_context as _get_session_ctx
from mlms.tools.get_session_log import get_session_log as _get_session_log
from mlms.tools.get_skills import get_skills as _get_skills
from mlms.tools.get_timeline import get_timeline as _get_timeline
from mlms.tools.memorize import memorize as _memorize
from mlms.tools.revert_fact import revert_fact as _revert_fact
from mlms.tools.search_memory import search_memory as _search_memory


@dataclass
class _AppCtx:
    pool: AsyncConnectionPool[Any]
    redis: Any


def _check_content_length(v: str) -> str:
    if len(v) > MEMORY_CONTENT_MAX_LENGTH:
        raise ValueError(
            f"content is {len(v)} characters, exceeds the {MEMORY_CONTENT_MAX_LENGTH}-character "
            "limit — split it into multiple memorize() calls instead of one large one."
        )
    return v


_MemoryContent = Annotated[
    str,
    BeforeValidator(_check_content_length),
    Field(max_length=MEMORY_CONTENT_MAX_LENGTH),
]


@asynccontextmanager
async def _lifespan(_server: FastMCP) -> AsyncIterator[_AppCtx]:
    pool: AsyncConnectionPool[Any] = AsyncConnectionPool(
        pg_dsn(settings.database_url),
        min_size=1,
        max_size=5,
        open=False,
        timeout=10,
        max_lifetime=3600,
        max_idle=300,
        kwargs={"autocommit": True},
    )
    await pool.open(wait=True, timeout=10)
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        yield _AppCtx(pool=pool, redis=r)
    finally:
        await pool.close()
        await r.aclose()


_log = logging.getLogger(__name__)


class PoolUnavailable(Exception):
    """Raised at the tool boundary when a DB connection can't be checked out in time."""


@asynccontextmanager
async def _acquire(app: _AppCtx) -> AsyncIterator[psycopg.AsyncConnection[Any]]:
    try:
        async with app.pool.connection() as conn:
            yield conn
    except (PoolTimeout, psycopg.OperationalError) as exc:
        _log.warning("db pool unavailable: %s stats=%s", exc, app.pool.get_stats())
        raise PoolUnavailable("database temporarily unavailable, try again shortly") from exc


# Sent to the client in the initialize response; clients that support server instructions
# add them to the agent's context.
_INSTRUCTIONS = """\
MLMS is the user's long-term memory across chats. Use it without being asked.

When the conversation is about a project, call `create_project` with the project's name first:
the same name returns the same project_id in every chat. Then look up what is already known with
`search_memory` or `get_facts` before answering from your own assumptions, and search memory
before telling the user you don't know something about the project.

Store new information with `memorize` as it comes up, without asking for permission:
- fact: the current value of something that may matter in a later chat: a setting, a number,
  a constraint ("Failed webhook deliveries are retried 5 times"). Values go into facts, not
  only into events. When a stored value changes, store the new statement as a fact again,
  worded like the old one with only the value changed: the server retires the old version
  only when the two are very similar.
- event: that something was decided, changed or done (event_type "decision", "phase" or
  "action"). A decision that sets a value is stored twice: the value as a fact, the decision
  as an event.
- skill: a reusable procedure or trick the user relies on.

Store one statement per call. Don't store small talk, your own guesses, or anything the user
asks you not to keep. If the user says a stored change was a mistake, call `revert_fact` on the
current version of the fact to restore the previous one.
"""

mcp: FastMCP = FastMCP("mlms", instructions=_INSTRUCTIONS, lifespan=_lifespan)

# CVE-2025-66416 (GHSA-9h52-p55h-vw2f): HTTP-based transports (sse/streamable-http) need
# explicit DNS rebinding protection — mcp_host defaults to 0.0.0.0, so the SDK's automatic
# localhost-only protection (added in 1.23.0) does not kick in here. Host/Origin allowlists
# are scoped to loopback for this single-/few-user, local-by-default deployment. Exposing
# MCP_HOST beyond loopback (LAN/VPS/domain) without adding that host to both lists below
# will make every request fail with 421 Invalid Host Header.
_LOOPBACK_HOSTS = ["localhost", "localhost:*", "127.0.0.1", "127.0.0.1:*"]
# Origin headers always carry a scheme (browsers/webapps send "http://127.0.0.1", never a
# bare "127.0.0.1"), unlike Host — reusing _LOOPBACK_HOSTS here would make _validate_origin
# reject every real Origin header, including legitimate loopback ones (found by testing a
# live server, not just reading the SDK source).
_LOOPBACK_ORIGINS = [
    f"{scheme}://{host}" for scheme in ("http", "https") for host in _LOOPBACK_HOSTS
]
mcp.settings.transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=_LOOPBACK_HOSTS,
    allowed_origins=_LOOPBACK_ORIGINS,
)

# Decision 23 — shared-secret gate for sse/streamable-http, layered on top of the
# host/Origin allowlist above, not instead of it. _LOOPBACK_HOSTS above carries "host:*"
# glob patterns for Origin/Host *header* matching; MCP_HOST is a plain bind address, so it
# needs its own plain set here.
_BIND_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class _SharedSecretMiddleware:
    """Bare ASGI middleware gating http scopes behind an Authorization: Bearer header.

    Deliberately not FastMCP's token_verifier/AuthSettings path: that requires
    issuer_url/resource_server_url and makes the server advertise OAuth discovery
    metadata even with no real authorization server behind it — MCP clients that see
    that metadata (including Claude Code, see anthropics/claude-code#59467) can ignore
    a configured static Authorization header and fail into a broken OAuth flow. This
    stays a flat token compare with no discovery surface at all.

    Only ever constructed with a non-empty secret (see __main__) — rejects the ValueError
    case as a coding-time invariant rather than a runtime "empty secret = allow all" path.
    """

    def __init__(self, app: Any, secret: str) -> None:
        if not secret:
            raise ValueError("shared secret must be non-empty")
        self._app = app
        self._secret = secret

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        # Only "http" carries an Authorization header to check here. Neither sse_app()
        # nor streamable_http_app() ever emits a "websocket" scope (grepped the installed
        # SDK — zero hits) — a prior version of this gated "websocket" too, but rejected it
        # by sending http.response.* events into a websocket scope, which is protocol-invalid
        # ASGI and would raise inside uvicorn rather than cleanly deny the connection. Simpler
        # and correct: gate only the scope type that can actually appear.
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        # First match, not dict()-over-the-list: a dict would silently keep the *last*
        # Authorization header if a client or proxy sent more than one — HTTP forbids
        # repeating it, but nothing on the wire enforces that.
        auth_header = next(
            (v for k, v in (scope.get("headers") or []) if k == b"authorization"), b""
        ).decode("latin-1")
        # RFC 7235 §2.1: auth-scheme is case-insensitive ("bearer"/"BEARER" are both
        # valid, and some HTTP client libraries normalize to one or the other).
        scheme, _, rest = auth_header.partition(" ")
        token = rest if scheme.lower() == "bearer" else ""
        # bytes, not str: compare_digest raises TypeError on a non-ASCII str, which would
        # turn a malformed header into a 500 instead of a 401.
        if not token or not secrets.compare_digest(token.encode("latin-1"), self._secret.encode()):
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [(b"content-type", b"text/plain")],
                }
            )
            await send({"type": "http.response.body", "body": b"unauthorized"})
            return
        await self._app(scope, receive, send)


_MemType = Literal["fact", "event", "skill", "session_log"]
_EventType = Literal["phase", "action", "decision"]

# readOnlyHint already tells clients "won't write/delete" — destructiveHint/idempotentHint
# add nothing on a read-only tool, so read-only annotations carry only the two fields that
# are actually informative: readOnlyHint and openWorldHint (does it call the external Qwen
# embedding API?).
_READ_ONLY_LOCAL = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
# get_facts(about=...) and get_session_log(semantic_query=...) call embed() conditionally,
# same as search_memory's fact branch — openWorldHint has to be True on all three or it lies
# about the two that were missed on the first pass.
_READ_ONLY_EXTERNAL = ToolAnnotations(readOnlyHint=True, openWorldHint=True)


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        # get-or-create by name: calling this again with the same name returns the
        # same project, doesn't create a duplicate.
        idempotentHint=True,
        openWorldHint=False,
    )
)
async def create_project(
    name: str,
    ctx: Context[Any, Any, Any],
    description: str | None = None,
) -> dict[str, Any]:
    """Get or create a project by name. Call this first, once per project, before
    using project_id anywhere else.

    name: stable human-chosen identifier, e.g. "acme-website-redesign". Calling this
    again later (new chat, same project) with the same name returns the same
    project_id — you don't need to remember the UUID across sessions, only the name.
    description: optional, only used on first creation.

    Returns: {"project_id": "<uuid>", "name": "...", "description": "...", "created_at": "..."}
    Use the returned project_id in every other tool's project_id argument.
    """
    app: _AppCtx = ctx.request_context.lifespan_context
    async with _acquire(app) as conn:
        p = await _create_project(name=name, description=description, conn=conn)
    return {
        "project_id": str(p.id),
        "name": p.name,
        "description": p.description,
        "created_at": p.created_at.isoformat(),
    }


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        # additive-only: creates a new fact/event/skill/log row, never deletes or
        # overwrites one (soft versioning via is_current, see storage/facts.py).
        destructiveHint=False,
        # every call creates a new row/version — repeating it is not a no-op.
        idempotentHint=False,
        # fact/session_log branches call the external Qwen embedding API.
        openWorldHint=True,
    )
)
async def memorize(
    content: _MemoryContent,
    type: str,
    metadata: dict[str, Any],
    ctx: Context[Any, Any, Any],
) -> dict[str, Any]:
    """Store a memory in the long-term memory system.

    content: max 10,000 characters (hard limit, rejected past this). Recommended
    ~500 characters per memory — a single atomic fact retrieves more precisely
    than a long composite one. If a fact bundles several unrelated details, split
    it across multiple memorize() calls rather than cramming it into one — don't
    drop details to fit, split them instead.

    type: "fact" | "event" | "skill" | "session_log"

    metadata by type — copy these templates exactly:

    fact:
      {"project_id": "<uuid from create_project>"}
      optional: "tags": ["list","of","strings"], "category": "string", "chat_id": "any-string-you-choose"

    event:
      {"project_id": "<uuid from create_project>",
       "title": "Short title under 100 chars",
       "event_type": "phase"}
      event_type values: "phase" | "action" | "decision"
      optional: "chat_id": "any-string"

    skill:
      {"name": "skill name"}
      optional: "domain": "string", "trigger_conditions": {"when": "short description"},
                "chat_id": "any-string", "project_id": "<uuid from create_project>"

    session_log:
      {"chat_id": "any-string-you-choose",
       "label": "Short label under 100 chars",
       "type": "topic"}
      type values: "topic"|"decision"|"problem"|"solution"|"insight"|"action"|"source"|"context"|"result"
      optional: "project_id": "<uuid from create_project>"

    NOTE: Never invent project_id UUIDs — call create_project first and use the returned id.
    NOTE: memory_id in the response is server-generated — do not pass it back anywhere.
    Returns: {"ok": true, "memory_id": "<server-generated-id>"}
    """
    app: _AppCtx = ctx.request_context.lifespan_context
    async with _acquire(app) as conn:
        return await _memorize(
            content=content,
            type=cast(_MemType, type),
            metadata=metadata,
            conn=conn,
            redis=app.redis,
        )


@mcp.tool(annotations=_READ_ONLY_EXTERNAL)
async def get_facts(
    ctx: Context[Any, Any, Any],
    about: str | None = None,
    project_id: str | None = None,
    tags: list[str] | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Retrieve current facts from semantic memory.

    about: natural-language query for vector search, e.g. "what is the project deadline"
    project_id: uuid from create_project. Omit to search all projects.
    tags: optional list to filter, e.g. ["backend", "auth"]
    limit: max results, default 10

    Returns list of fact objects: id, content, tags, category, version, created_at.
    Only returns is_current=TRUE facts (outdated versions are excluded automatically).
    """
    app: _AppCtx = ctx.request_context.lifespan_context
    async with _acquire(app) as conn:
        rows = await _get_facts(
            about=about,
            project_id=project_id,
            tags=tags,
            limit=limit,
            conn=conn,
            redis=app.redis,
        )
    return [r.model_dump(mode="json") for r in rows]


@mcp.tool(annotations=_READ_ONLY_LOCAL)
async def get_timeline(
    ctx: Context[Any, Any, Any],
    project_id: str | None = None,
    time_range: dict[str, str] | None = None,
    event_type: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Query timeline events (phases, actions, decisions).

    project_id: uuid from create_project
    time_range: {"start": "2025-01-01T00:00:00Z", "end": "2026-12-31T23:59:59Z"} — both optional
    event_type: "phase" | "action" | "decision" — optional filter
    limit: default 20
    """
    app: _AppCtx = ctx.request_context.lifespan_context
    async with _acquire(app) as conn:
        rows = await _get_timeline(
            project_id=project_id,
            time_range=time_range,
            event_type=cast(_EventType, event_type),
            limit=limit,
            conn=conn,
        )
    return [r.model_dump(mode="json") for r in rows]


@mcp.tool(annotations=_READ_ONLY_EXTERNAL)
async def get_session_log(
    ctx: Context[Any, Any, Any],
    chat_id: str | None = None,
    limit: int = 50,
    entry_type: str | None = None,
    semantic_query: str | None = None,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve session log entries. Two exclusive modes — do NOT combine chat_id and semantic_query.

    Mode 1 — current session: provide chat_id, omit semantic_query
      chat_id: the session string you used when calling memorize
      entry_type: optional filter string

    Mode 2 — cross-session search: provide semantic_query, omit chat_id
      semantic_query: natural-language search, e.g. "authentication decisions"
      project_id: optional scope — uuid from create_project
      limit: default 50
    """
    app: _AppCtx = ctx.request_context.lifespan_context
    async with _acquire(app) as conn:
        rows = await _get_session_log(
            chat_id=chat_id,
            limit=limit,
            entry_type=entry_type,
            semantic_query=semantic_query,
            project_id=project_id,
            conn=conn,
            redis=app.redis,
        )
    return [r.model_dump(mode="json") for r in rows]


@mcp.tool(annotations=_READ_ONLY_LOCAL)
async def get_session_context(
    chat_id: str,
    ctx: Context[Any, Any, Any],
) -> dict[str, Any] | None:
    """Read working memory for a chat session from Redis. Returns None if session missing."""
    app: _AppCtx = ctx.request_context.lifespan_context
    result = await _get_session_ctx(chat_id=chat_id, redis=app.redis)
    return result.model_dump(mode="json") if result is not None else None


@mcp.tool(annotations=_READ_ONLY_LOCAL)
async def get_skills(
    ctx: Context[Any, Any, Any],
    domain: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Get procedural skills, optionally filtered by domain."""
    app: _AppCtx = ctx.request_context.lifespan_context
    async with _acquire(app) as conn:
        rows = await _get_skills(domain=domain, limit=limit, conn=conn)
    return [r.model_dump(mode="json") for r in rows]


@mcp.tool(
    annotations=ToolAnnotations(
        readOnlyHint=False,
        # Soft-delete only: flips is_current, never DROPs a row (see storage/facts.py
        # anti-pattern note in CLAUDE.md). Judgment call, not an unambiguous protocol
        # fact — from the calling LLM's view, the reverted fact vanishes from every
        # default get_facts()/search_memory() listing, which reads like a deletion
        # even though the row is still physically present.
        destructiveHint=False,
        # Calling revert_fact(fact_id) twice in a row raises on the second call
        # (fact_id is no longer is_current after the first revert). False anyway —
        # it's a guard against a concurrent memorize() interleaving and mutating the
        # replaces_id chain between the two calls, not a claim that a single-caller
        # retry is unsafe.
        idempotentHint=False,
        openWorldHint=False,
    )
)
async def revert_fact(
    fact_id: str,
    ctx: Context[Any, Any, Any],
) -> dict[str, Any]:
    """Revert a fact to its previous version. fact_id must be the current version."""
    from uuid import UUID

    app: _AppCtx = ctx.request_context.lifespan_context
    async with _acquire(app) as conn:
        return await _revert_fact(fact_id=UUID(fact_id), conn=conn)


@mcp.tool(annotations=_READ_ONLY_EXTERNAL)
async def search_memory(
    query: str,
    ctx: Context[Any, Any, Any],
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search across ALL memory layers simultaneously (facts + timeline + skills).

    query: natural-language search, e.g. "what did we decide about the database"
    filters (optional dict):
      "project_id": uuid from create_project
      "memory_type": "fact" | "event" | "skill"
      "time_range": {"start": "ISO8601", "end": "ISO8601"}

    Returns ranked results from all layers with a "layer" field indicating source.
    """
    app: _AppCtx = ctx.request_context.lifespan_context
    async with _acquire(app) as conn:
        rows = await _search_memory(query=query, filters=filters, conn=conn, redis=app.redis)
    return [r.model_dump(mode="json") for r in rows]


@mcp.tool(annotations=_READ_ONLY_LOCAL)
async def export_project(
    project_id: str,
    ctx: Context[Any, Any, Any],
) -> str:
    """Export all current facts, timeline events, and skills for a project as JSON."""
    app: _AppCtx = ctx.request_context.lifespan_context
    async with _acquire(app) as conn:
        return await _export_project(project_id=project_id, conn=conn)


def _require_secret_on_non_loopback_bind(host: str, secret: str | None, transport: str) -> None:
    """Decision 23: a non-loopback bind is exactly the homelab/VPS scenario the gate covers —
    refuse to listen unauthenticated rather than pass through unnoticed."""
    if host.strip().lower() in _BIND_LOOPBACK_HOSTS or secret:
        return
    raise SystemExit(
        f"MCP_HOST={host!r} is not loopback: refusing to start {transport!r} without "
        f"MCP_SHARED_SECRET set. Loopback hosts ({sorted(_BIND_LOOPBACK_HOSTS)}) don't "
        "require it."
    )


def _warn_if_no_embedding_key() -> None:
    if not settings.embedding_api_key:
        _log.warning(
            "EMBEDDING_API_KEY is empty: storing facts and session_log entries and searching "
            "by meaning will fail until it is set"
        )


if __name__ == "__main__":
    _warn_if_no_embedding_key()
    transport = settings.mcp_transport
    if transport in ("sse", "streamable-http"):
        mcp.settings.host = settings.mcp_host
        mcp.settings.port = settings.mcp_port
        _require_secret_on_non_loopback_bind(
            settings.mcp_host, settings.mcp_shared_secret, transport
        )
        if settings.mcp_shared_secret:
            import uvicorn

            inner_app = mcp.sse_app() if transport == "sse" else mcp.streamable_http_app()
            gated_app = _SharedSecretMiddleware(inner_app, settings.mcp_shared_secret)
            uvicorn.run(
                gated_app,
                host=mcp.settings.host,
                port=mcp.settings.port,
                log_level=mcp.settings.log_level.lower(),
            )
        else:
            mcp.run(transport=transport)
    else:
        mcp.run()
