from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import psycopg
import psycopg.rows


async def export_project(
    project_id: str,
    conn: psycopg.AsyncConnection[Any],
) -> str:
    pid = UUID(project_id)
    cur = conn.cursor(row_factory=psycopg.rows.dict_row)

    await cur.execute(
        "SELECT id, content, tags, category, version, created_at"
        " FROM facts WHERE project_id = %s AND is_current = TRUE",
        (pid,),
    )
    facts: list[dict[str, Any]] = await cur.fetchall()

    await cur.execute(
        "SELECT id, time, event_type, title, content"
        " FROM timeline_events WHERE project_id = %s ORDER BY time",
        (pid,),
    )
    timeline: list[dict[str, Any]] = await cur.fetchall()

    await cur.execute(
        "SELECT id, name, content, domain, version, created_at FROM skills",
    )
    skills: list[dict[str, Any]] = await cur.fetchall()

    def _default(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, UUID):
            return str(obj)
        raise TypeError(type(obj))

    return json.dumps(
        {
            "project_id": project_id,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "facts": facts,
            "timeline": timeline,
            "skills": skills,
        },
        default=_default,
    )
