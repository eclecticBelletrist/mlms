from __future__ import annotations

import asyncio
import logging
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

from mlms.config import MEMORY_CONTENT_MAX_LENGTH, settings
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


def _pg_dsn(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


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
        _pg_dsn(settings.database_url),
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


mcp: FastMCP = FastMCP("mlms", lifespan=_lifespan)

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
      {"project_id": "a1b2c3d4-0000-0000-0000-000000000001"}
      optional: "tags": ["list","of","strings"], "category": "string", "chat_id": "any-string-you-choose"

    event:
      {"project_id": "a1b2c3d4-0000-0000-0000-000000000001",
       "title": "Short title under 100 chars",
       "event_type": "phase"}
      event_type values: "phase" | "action" | "decision"
      optional: "chat_id": "any-string"

    skill:
      {"name": "skill name"}
      optional: "domain": "string", "trigger_conditions": "string",
                "chat_id": "any-string", "project_id": "a1b2c3d4-0000-0000-0000-000000000001"

    session_log:
      {"chat_id": "any-string-you-choose",
       "label": "Short label under 100 chars",
       "type": "topic"}
      type values: "topic"|"decision"|"problem"|"solution"|"insight"|"action"|"source"|"context"|"result"
      optional: "project_id": "a1b2c3d4-0000-0000-0000-000000000001"

    NOTE: Never invent project_id UUIDs — always use the one above for demo.
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
    project_id: "a1b2c3d4-0000-0000-0000-000000000001" for demo. Omit to search all projects.
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

    project_id: "a1b2c3d4-0000-0000-0000-000000000001" for demo
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
      project_id: optional scope — "a1b2c3d4-0000-0000-0000-000000000001"
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
        # Calling revert_fact(fact_id) twice in a row with no concurrent memorize()
        # in between is a no-op the second time (the fetch has no is_current filter).
        # False anyway — it's a guard against a concurrent memorize() interleaving
        # and mutating the replaces_id chain between the two calls, not a claim that
        # a single-caller retry is unsafe.
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
      "project_id": "a1b2c3d4-0000-0000-0000-000000000001"
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


if __name__ == "__main__":
    transport = settings.mcp_transport
    if transport in ("sse", "streamable-http"):
        mcp.settings.host = settings.mcp_host
        mcp.settings.port = settings.mcp_port
        mcp.run(transport=transport)
    else:
        mcp.run()
