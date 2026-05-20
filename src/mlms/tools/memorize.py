from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

import psycopg

from mlms.embedding import embed
from mlms.storage.facts import upsert_fact
from mlms.storage.session_ctx import SessionContext, get_session_context, set_session_context
from mlms.storage.session_log import insert_session_entry
from mlms.storage.skills import insert_skill
from mlms.storage.timeline import insert_event


async def _refresh_redis(chat_id: str, project_id: str, *, redis: Any) -> None:
    now = datetime.now(UTC)
    ctx = await get_session_context(chat_id, redis=redis)
    if ctx is None:
        ctx = SessionContext(project_id=project_id, last_activity=now)
    else:
        ctx.last_activity = now
    await set_session_context(chat_id, ctx, redis=redis)


async def memorize(
    *,
    content: str,
    type: Literal["fact", "event", "skill", "session_log"],
    metadata: dict[str, Any],
    conn: psycopg.AsyncConnection[Any],
    redis: Any,
) -> dict[str, Any]:
    if type == "fact":
        project_id = UUID(metadata["project_id"])
        vec = await embed(content, redis)
        result = await upsert_fact(
            content=content,
            project_id=project_id,
            embedding_vec=vec,
            tags=metadata.get("tags"),
            category=metadata.get("category"),
            conn=conn,
        )
        chat_id: str | None = metadata.get("chat_id")
        if chat_id:
            await _refresh_redis(chat_id, str(project_id), redis=redis)
        return {"ok": True, "memory_id": str(result.id)}

    elif type == "event":
        project_id = UUID(metadata["project_id"])
        title: str = metadata["title"]
        chat_id = metadata.get("chat_id")
        event_result = await insert_event(
            project_id=project_id,
            event_type=metadata["event_type"],
            title=title,
            content=content,
            conn=conn,
        )
        if chat_id:
            await insert_session_entry(
                chat_id=chat_id,
                entry_type="action",
                label=title[:100],
                content=content,
                project_id=project_id,
                conn=conn,
            )
            await _refresh_redis(chat_id, str(project_id), redis=redis)
        return {"ok": True, "memory_id": str(event_result.id)}

    elif type == "skill":
        name: str = metadata["name"]
        chat_id = metadata.get("chat_id")
        skill_result = await insert_skill(
            name=name,
            content=content,
            domain=metadata.get("domain"),
            trigger_conditions=metadata.get("trigger_conditions"),
            conn=conn,
        )
        if chat_id:
            pid_str = metadata.get("project_id")
            await insert_session_entry(
                chat_id=chat_id,
                entry_type="action",
                label=f"skill:{name}"[:100],
                content=content,
                project_id=UUID(pid_str) if pid_str else None,
                conn=conn,
            )
            await _refresh_redis(chat_id, pid_str or "", redis=redis)
        return {"ok": True, "memory_id": str(skill_result.id)}

    else:  # type == "session_log"
        chat_id = metadata["chat_id"]
        label: str = metadata["label"][:100]
        pid_str = metadata.get("project_id")
        vec = await embed(label, redis)
        await insert_session_entry(
            chat_id=chat_id,
            entry_type=metadata["type"],
            label=label,
            content=content,
            embedding_vec=vec,
            project_id=UUID(pid_str) if pid_str else None,
            conn=conn,
        )
        await _refresh_redis(chat_id, pid_str or "", redis=redis)
        return {"ok": True, "memory_id": f"{chat_id}:{label}"}
