"""Smoke tests: one minimal valid call per MCP tool.

MLMS_INTEGRATION=1 pytest tests/integration/test_smoke.py -v
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import psycopg
import pytest
import redis.asyncio as aioredis

from mlms.config import settings
from mlms.storage.facts import upsert_fact
from mlms.storage.session_ctx import SessionContext, set_session_context
from mlms.storage.session_log import insert_session_entry
from mlms.storage.skills import insert_skill
from mlms.storage.timeline import insert_event
from mlms.tools.export_project import export_project
from mlms.tools.get_facts import get_facts
from mlms.tools.get_session_ctx import get_session_context
from mlms.tools.get_session_log import get_session_log
from mlms.tools.get_skills import get_skills
from mlms.tools.get_timeline import get_timeline
from mlms.tools.memorize import memorize
from mlms.tools.revert_fact import revert_fact
from mlms.tools.search_memory import search_memory

pytestmark = pytest.mark.skipif(
    os.getenv("MLMS_INTEGRATION") != "1",
    reason="Set MLMS_INTEGRATION=1 to run (requires docker compose up)",
)

_DIM = 1536
_VEC_A: list[float] = [1.0] + [0.0] * (_DIM - 1)
_VEC_LIT = "[" + ",".join(str(x) for x in _VEC_A) + "]"


@pytest.fixture
async def rd_text() -> AsyncGenerator[Any, None]:
    """Redis with decode_responses=True — needed by session_ctx hash deserialization."""
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    yield r
    await r.aclose()


async def test_smoke_memorize(
    pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID, rd: Any
) -> None:
    with patch("mlms.tools.memorize.embed", new=AsyncMock(return_value=_VEC_A)):
        result = await memorize(
            content="Claude uses transformer architecture",
            type="fact",
            metadata={"project_id": str(project_id)},
            conn=pg,
            redis=rd,
        )
    assert result["ok"] is True
    assert uuid.UUID(result["memory_id"])


async def test_smoke_get_facts(
    pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID, rd: Any
) -> None:
    await upsert_fact(
        content="get_facts smoke", project_id=project_id, embedding_vec=_VEC_A, conn=pg
    )
    facts = await get_facts(project_id=str(project_id), conn=pg, redis=rd)
    assert any(f.content == "get_facts smoke" for f in facts)


async def test_smoke_get_timeline(
    pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID
) -> None:
    await insert_event(
        project_id=project_id, event_type="action", title="smoke-event", conn=pg
    )
    events = await get_timeline(project_id=str(project_id), conn=pg)
    assert any(e.title == "smoke-event" for e in events)


async def test_smoke_get_session_log(pg: psycopg.AsyncConnection[Any], rd: Any) -> None:
    chat_id = f"smoke-log-{uuid.uuid4()}"
    await insert_session_entry(
        chat_id=chat_id, entry_type="topic", label="smoke-topic", conn=pg
    )
    entries = await get_session_log(chat_id=chat_id, conn=pg, redis=rd)
    assert len(entries) >= 1
    assert entries[0].label == "smoke-topic"


async def test_smoke_get_session_context(rd_text: Any) -> None:
    chat_id = f"smoke-ctx-{uuid.uuid4()}"
    ctx = SessionContext(project_id="proj-smoke", last_activity=datetime.now(UTC))
    await set_session_context(chat_id, ctx, redis=rd_text)
    result = await get_session_context(chat_id=chat_id, redis=rd_text)
    assert result is not None
    assert result.project_id == "proj-smoke"


async def test_smoke_get_skills(pg: psycopg.AsyncConnection[Any]) -> None:
    skill_name = f"smoke-skill-{uuid.uuid4()}"
    skill = await insert_skill(name=skill_name, content="do X to achieve Y", conn=pg)
    try:
        skills = await get_skills(conn=pg)
        assert any(s.id == skill.id for s in skills)
    finally:
        await pg.execute("DELETE FROM skills WHERE id = %s", (skill.id,))


async def test_smoke_search_memory(
    pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID, rd: Any
) -> None:
    await upsert_fact(
        content="semantic search target", project_id=project_id, embedding_vec=_VEC_A, conn=pg
    )
    with patch("mlms.tools.search_memory.embed", new=AsyncMock(return_value=_VEC_A)):
        results = await search_memory(
            query="semantic search target",
            filters={"project_id": str(project_id), "memory_type": "fact"},
            conn=pg,
            redis=rd,
        )
    assert len(results) >= 1
    assert results[0].layer == "fact"


async def test_smoke_revert_fact(
    pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID
) -> None:
    # Pre-built chain: v1 (retired) ← v2 (current)
    v1_id = uuid.uuid4()
    await pg.execute(
        "INSERT INTO facts (id, project_id, content, embedding, version, is_current)"
        " VALUES (%s, %s, 'v1', %s::vector, 1, FALSE)",
        (v1_id, project_id, _VEC_LIT),
    )
    v2_id = uuid.uuid4()
    await pg.execute(
        "INSERT INTO facts"
        " (id, project_id, content, embedding, version, is_current, replaces_id)"
        " VALUES (%s, %s, 'v2', %s::vector, 2, TRUE, %s)",
        (v2_id, project_id, _VEC_LIT, v1_id),
    )
    result = await revert_fact(fact_id=v2_id, conn=pg)
    assert result["ok"] is True
    assert result["reverted_to"] == str(v1_id)


async def test_smoke_export_project(
    pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID
) -> None:
    await upsert_fact(
        content="export smoke content", project_id=project_id, embedding_vec=_VEC_A, conn=pg
    )
    raw = await export_project(project_id=str(project_id), conn=pg)
    data = json.loads(raw)
    assert data["project_id"] == str(project_id)
    assert "exported_at" in data
    assert len(data["facts"]) >= 1
