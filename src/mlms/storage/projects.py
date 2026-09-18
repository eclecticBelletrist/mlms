from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psycopg
import psycopg.rows


@dataclass(frozen=True)
class Project:
    id: uuid.UUID
    name: str
    description: str | None
    created_at: datetime


_UPSERT = """
    INSERT INTO projects (name, description)
    VALUES (%s, %s)
    ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
    RETURNING id, name, description, created_at
"""


async def get_or_create_project(
    *,
    name: str,
    description: str | None = None,
    conn: psycopg.AsyncConnection[Any],
) -> Project:
    cur = conn.cursor(row_factory=psycopg.rows.dict_row)
    await cur.execute(_UPSERT, (name, description))
    row = await cur.fetchone()
    assert row is not None
    return Project(
        id=row["id"],
        name=row["name"],
        description=row["description"],
        created_at=row["created_at"],
    )
