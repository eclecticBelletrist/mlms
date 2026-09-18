from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import psycopg
from pydantic import BaseModel

from mlms.storage.projects import get_or_create_project as _storage_get_or_create_project


class Project(BaseModel):
    id: UUID
    name: str
    description: str | None = None
    created_at: datetime


async def create_project(
    *,
    name: str,
    description: str | None = None,
    conn: psycopg.AsyncConnection[Any],
) -> Project:
    p = await _storage_get_or_create_project(name=name, description=description, conn=conn)
    return Project(id=p.id, name=p.name, description=p.description, created_at=p.created_at)
