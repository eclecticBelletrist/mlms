"""Working memory: Redis hash per chat_id, TTL reset on every read/write."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from mlms.config import REDIS_SESSION_TTL


def _key(chat_id: str) -> str:
    return f"session:{chat_id}"


@dataclass
class SessionContext:
    project_id: str
    last_activity: datetime
    open_todos: list[str] = field(default_factory=list)
    active_entities: list[str] = field(default_factory=list)
    recent_decisions: list[str] = field(default_factory=list)


def _serialize(ctx: SessionContext) -> dict[str, str]:
    return {
        "project_id": ctx.project_id,
        "last_activity": ctx.last_activity.isoformat(),
        "open_todos": json.dumps(ctx.open_todos),
        "active_entities": json.dumps(ctx.active_entities),
        "recent_decisions": json.dumps(ctx.recent_decisions),
    }


def _deserialize(raw: dict[str, str]) -> SessionContext:
    return SessionContext(
        project_id=raw["project_id"],
        last_activity=datetime.fromisoformat(raw["last_activity"]),
        open_todos=json.loads(raw["open_todos"]),
        active_entities=json.loads(raw["active_entities"]),
        recent_decisions=json.loads(raw["recent_decisions"]),
    )


async def get_session_context(
    chat_id: str,
    *,
    redis: Any,
) -> SessionContext | None:
    key = _key(chat_id)
    raw: dict[str, str] = await redis.hgetall(key)
    if not raw:
        return None
    await redis.expire(key, REDIS_SESSION_TTL)
    return _deserialize(raw)


async def set_session_context(
    chat_id: str,
    ctx: SessionContext,
    *,
    redis: Any,
) -> None:
    key = _key(chat_id)
    await redis.hset(key, mapping=_serialize(ctx))
    await redis.expire(key, REDIS_SESSION_TTL)
