from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

import psycopg
import redis.asyncio as aioredis
from mcp.server.fastmcp import Context, FastMCP

from mlms.config import settings
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
    conn: psycopg.AsyncConnection[Any]
    redis: Any


def _pg_dsn(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


@asynccontextmanager
async def _lifespan(_server: FastMCP) -> AsyncIterator[_AppCtx]:
    conn = await psycopg.AsyncConnection.connect(_pg_dsn(settings.database_url), autocommit=True)
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        yield _AppCtx(conn=conn, redis=r)
    finally:
        await conn.close()
        await r.aclose()


mcp: FastMCP = FastMCP("mlms", lifespan=_lifespan)

from mcp.server.fastmcp.server import TransportSecuritySettings
mcp.settings.transport_security = TransportSecuritySettings(enable_dns_rebinding_protection=False)


_MemType = Literal["fact", "event", "skill", "session_log"]
_EventType = Literal["phase", "action", "decision"]


@mcp.tool()
async def memorize(
    content: str,
    type: str,
    metadata: dict[str, Any],
    ctx: Context[Any, Any, Any],
) -> dict[str, Any]:
    """Store a memory in the long-term memory system.

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
    return await _memorize(
        content=content,
        type=cast(_MemType, type),
        metadata=metadata,
        conn=app.conn,
        redis=app.redis,
    )


@mcp.tool()
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
    rows = await _get_facts(
        about=about,
        project_id=project_id,
        tags=tags,
        limit=limit,
        conn=app.conn,
        redis=app.redis,
    )
    return [r.model_dump(mode="json") for r in rows]


@mcp.tool()
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
    rows = await _get_timeline(
        project_id=project_id,
        time_range=time_range,
        event_type=cast(_EventType, event_type),
        limit=limit,
        conn=app.conn,
    )
    return [r.model_dump(mode="json") for r in rows]


@mcp.tool()
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
    rows = await _get_session_log(
        chat_id=chat_id,
        limit=limit,
        entry_type=entry_type,
        semantic_query=semantic_query,
        project_id=project_id,
        conn=app.conn,
        redis=app.redis,
    )
    return [r.model_dump(mode="json") for r in rows]


@mcp.tool()
async def get_session_context(
    chat_id: str,
    ctx: Context[Any, Any, Any],
) -> dict[str, Any] | None:
    """Read working memory for a chat session from Redis. Returns None if session missing."""
    app: _AppCtx = ctx.request_context.lifespan_context
    result = await _get_session_ctx(chat_id=chat_id, redis=app.redis)
    return result.model_dump(mode="json") if result is not None else None


@mcp.tool()
async def get_skills(
    ctx: Context[Any, Any, Any],
    domain: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Get procedural skills, optionally filtered by domain."""
    app: _AppCtx = ctx.request_context.lifespan_context
    rows = await _get_skills(domain=domain, limit=limit, conn=app.conn)
    return [r.model_dump(mode="json") for r in rows]


@mcp.tool()
async def revert_fact(
    fact_id: str,
    ctx: Context[Any, Any, Any],
) -> dict[str, Any]:
    """Revert a fact to its previous version. fact_id must be the current version."""
    from uuid import UUID
    app: _AppCtx = ctx.request_context.lifespan_context
    return await _revert_fact(fact_id=UUID(fact_id), conn=app.conn)


@mcp.tool()
async def search_memory(
    query: str,
    ctx: Context[Any, Any, Any],
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Search across ALL memory layers simultaneously (facts + timeline + session_log + skills).

    query: natural-language search, e.g. "what did we decide about the database"
    filters (optional dict):
      "project_id": "a1b2c3d4-0000-0000-0000-000000000001"
      "memory_type": "fact" | "event" | "skill" | "session_log"
      "time_range": {"start": "ISO8601", "end": "ISO8601"}

    Returns ranked results from all layers with a "layer" field indicating source.
    """
    app: _AppCtx = ctx.request_context.lifespan_context
    rows = await _search_memory(query=query, filters=filters, conn=app.conn, redis=app.redis)
    return [r.model_dump(mode="json") for r in rows]


@mcp.tool()
async def export_project(
    project_id: str,
    ctx: Context[Any, Any, Any],
) -> str:
    """Export all current facts, timeline events, and skills for a project as JSON."""
    app: _AppCtx = ctx.request_context.lifespan_context
    return await _export_project(project_id=project_id, conn=app.conn)


if __name__ == "__main__":
    transport = settings.mcp_transport
    if transport in ("sse", "streamable-http"):
        mcp.settings.host = settings.mcp_host
        mcp.settings.port = settings.mcp_port
        mcp.run(transport=transport)
    else:
        mcp.run()
