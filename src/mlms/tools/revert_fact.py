from __future__ import annotations

from typing import Any
from uuid import UUID

import psycopg
import psycopg.rows

_FETCH = "SELECT project_id, replaces_id FROM facts WHERE id = %s"
_RETIRE = "UPDATE facts SET is_current = FALSE WHERE id = %s"
_RESTORE = "UPDATE facts SET is_current = TRUE WHERE id = %s"


async def revert_fact(
    *,
    fact_id: UUID,
    conn: psycopg.AsyncConnection[Any],
) -> dict[str, Any]:
    cur = conn.cursor(row_factory=psycopg.rows.dict_row)

    await cur.execute(_FETCH, (fact_id,))
    fact = await cur.fetchone()
    if fact is None:
        raise ValueError(f"fact {fact_id} not found")

    restored_id: UUID | None = fact["replaces_id"]
    if restored_id is None:
        raise ValueError(f"fact {fact_id} has no prior version to revert to")

    await cur.execute(_FETCH, (restored_id,))
    restored = await cur.fetchone()
    if restored is None:
        raise ValueError(f"target fact {restored_id} not found")

    if restored["project_id"] != fact["project_id"]:
        raise ValueError(
            f"cross-project revert blocked: "
            f"{fact_id} in {fact['project_id']}, "
            f"{restored_id} in {restored['project_id']}"
        )

    async with conn.transaction():
        await conn.execute(_RETIRE, (fact_id,))
        await conn.execute(_RESTORE, (restored_id,))

    return {"ok": True, "reverted_to": str(restored_id)}
