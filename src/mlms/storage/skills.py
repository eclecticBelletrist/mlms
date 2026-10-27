from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psycopg
import psycopg.rows
from psycopg.types.json import Jsonb


@dataclass(frozen=True)
class SkillInsertResult:
    id: uuid.UUID
    name: str
    version: int
    created_at: datetime


@dataclass(frozen=True)
class Skill:
    id: uuid.UUID
    name: str
    domain: str | None
    content: str
    trigger_conditions: dict[str, Any] | None
    version: int
    created_at: datetime


_MAX_VERSION = "SELECT COALESCE(MAX(version), 0) FROM skills WHERE name = %s"

_INSERT = """
    INSERT INTO skills (id, name, domain, content, trigger_conditions, version, created_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

_SELECT_COLS = "id, name, domain, content, trigger_conditions, version, created_at"

# DISTINCT ON + ORDER BY name, version DESC → first row per name = highest version.
# Domain and text filters run on this latest-per-name set, so an outdated version
# that matched the filter is never returned in place of a newer one that does not.
_LATEST_PER_NAME = (
    f"(SELECT DISTINCT ON (name) {_SELECT_COLS} FROM skills ORDER BY name, version DESC) AS latest"
)

_GET_ALL_LATEST = f"SELECT {_SELECT_COLS} FROM {_LATEST_PER_NAME} ORDER BY name LIMIT %s"

_GET_ALL_LATEST_BY_DOMAIN = (
    f"SELECT {_SELECT_COLS} FROM {_LATEST_PER_NAME} WHERE domain = %s ORDER BY name LIMIT %s"
)

_GET_LATEST_BY_NAME = (
    f"SELECT {_SELECT_COLS} FROM skills WHERE name = %s ORDER BY version DESC LIMIT 1"
)

# Full-text match on the words of the query, as for events in search_memory, so a
# natural-language query ("how to debug a webhook") finds "Debugging webhooks: ...".
# ILIKE keeps the exact-substring match for text the english configuration does not stem.
_GET_ALL_LATEST_BY_SEARCH = (
    f"SELECT {_SELECT_COLS} FROM {_LATEST_PER_NAME}"
    " WHERE to_tsvector('english', name || ' ' || content) @@ plainto_tsquery('english', %s)"
    " OR content ILIKE %s OR name ILIKE %s ORDER BY name LIMIT %s"
)


def _row_to_skill(r: dict[str, Any]) -> Skill:
    return Skill(
        id=r["id"],
        name=r["name"],
        domain=r["domain"],
        content=r["content"],
        trigger_conditions=r["trigger_conditions"],
        version=r["version"],
        created_at=r["created_at"],
    )


async def insert_skill(
    *,
    name: str,
    content: str,
    domain: str | None = None,
    trigger_conditions: dict[str, Any] | None = None,
    conn: psycopg.AsyncConnection[Any],
) -> SkillInsertResult:
    new_id = uuid.uuid4()
    now = datetime.now(UTC)
    tc_val: Jsonb | None = Jsonb(trigger_conditions) if trigger_conditions is not None else None

    cur = conn.cursor()
    await cur.execute(_MAX_VERSION, (name,))
    row: tuple[int] | None = await cur.fetchone()
    next_version = (row[0] if row is not None else 0) + 1

    await cur.execute(
        _INSERT,
        (new_id, name, domain, content, tc_val, next_version, now),
    )
    return SkillInsertResult(id=new_id, name=name, version=next_version, created_at=now)


async def get_skills(
    *,
    name: str | None = None,
    domain: str | None = None,
    search: str | None = None,
    limit: int = 20,
    conn: psycopg.AsyncConnection[Any],
) -> list[Skill]:
    cur = conn.cursor(row_factory=psycopg.rows.dict_row)
    if name is not None:
        await cur.execute(_GET_LATEST_BY_NAME, (name,))
    elif domain is not None:
        await cur.execute(_GET_ALL_LATEST_BY_DOMAIN, (domain, limit))
    elif search is not None:
        pattern = f"%{search}%"
        await cur.execute(_GET_ALL_LATEST_BY_SEARCH, (search, pattern, pattern, limit))
    else:
        await cur.execute(_GET_ALL_LATEST, (limit,))
    rows: list[dict[str, Any]] = await cur.fetchall()
    return [_row_to_skill(r) for r in rows]
