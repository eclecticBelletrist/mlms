from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel

from mlms.storage.session_ctx import get_session_context as _get_ctx


class SessionContext(BaseModel):
    project_id: str
    last_activity: datetime
    open_todos: list[str]
    active_entities: list[str]
    recent_decisions: list[str]


async def get_session_context(
    *,
    chat_id: str,
    redis: Any,
) -> SessionContext | None:
    ctx = await _get_ctx(chat_id, redis=redis)
    if ctx is None:
        return None
    return SessionContext(
        project_id=ctx.project_id,
        last_activity=ctx.last_activity,
        open_todos=ctx.open_todos,
        active_entities=ctx.active_entities,
        recent_decisions=ctx.recent_decisions,
    )
