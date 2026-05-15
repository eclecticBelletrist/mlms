from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import psycopg
from pydantic import BaseModel

from mlms.storage.skills import get_skills as _storage_get_skills


class Skill(BaseModel):
    id: UUID
    name: str
    domain: str | None = None
    content: str
    trigger_conditions: dict[str, Any] | None = None
    version: int
    created_at: datetime


async def get_skills(
    *,
    domain: str | None = None,
    limit: int = 20,
    conn: psycopg.AsyncConnection[Any],
) -> list[Skill]:
    raw = await _storage_get_skills(limit=limit, conn=conn)
    results = [
        Skill(
            id=s.id,
            name=s.name,
            domain=s.domain,
            content=s.content,
            trigger_conditions=s.trigger_conditions,
            version=s.version,
            created_at=s.created_at,
        )
        for s in raw
    ]
    if domain is not None:
        results = [r for r in results if r.domain == domain]
    return results
