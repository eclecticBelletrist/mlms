"""Integration: one-sided time ranges and exact-identifier fact hits against real Postgres.

MLMS_INTEGRATION=1 pytest tests/integration/test_read_filters.py -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import psycopg
import pytest

from mlms.storage.facts import upsert_fact
from mlms.storage.timeline import insert_event
from mlms.tools.get_facts import get_facts
from mlms.tools.get_timeline import get_timeline
from mlms.tools.search_memory import _search_events

pytestmark = pytest.mark.skipif(
    os.getenv("MLMS_INTEGRATION") != "1",
    reason="Set MLMS_INTEGRATION=1 to run (requires docker compose up)",
)

_DIM = 1536


def _axis(*weights: float) -> list[float]:
    """Vector with the given leading components, zero elsewhere."""
    return [*weights, *([0.0] * (_DIM - len(weights)))]


class TestOneSidedTimeRange:
    async def test_get_timeline_start_only_and_end_only(
        self, pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID
    ) -> None:
        for day in ("2024-03-01", "2024-06-01"):
            await pg.execute(
                "INSERT INTO timeline_events (id, time, project_id, event_type, title, content)"
                " VALUES (%s, %s, %s, 'action', %s, 'launch notes')",
                (uuid.uuid4(), f"{day}T12:00:00+00:00", project_id, day),
            )
        pid = str(project_id)
        after = await get_timeline(
            project_id=pid, time_range={"start": "2024-04-01T00:00:00+00:00"}, conn=pg
        )
        before = await get_timeline(
            project_id=pid, time_range={"end": "2024-04-01T00:00:00+00:00"}, conn=pg
        )
        assert [e.title for e in after] == ["2024-06-01"]
        assert [e.title for e in before] == ["2024-03-01"]

    async def test_search_events_one_sided(
        self, pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID
    ) -> None:
        await insert_event(
            project_id=project_id, event_type="action", title="t", content="launch notes", conn=pg
        )
        hits = await _search_events(
            "launch", str(project_id), {"start": "2000-01-01T00:00:00+00:00"}, pg, 5
        )
        assert len(hits) == 1


class TestExactIdentifierHit:
    async def test_exact_hit_survives_a_full_vector_page(
        self, pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID
    ) -> None:
        # pairwise orthogonal vectors: no conflict between the facts, and a fixed ranking
        for i, text in enumerate(("alpha note", "beta note", "gamma note"), start=1):
            await upsert_fact(
                content=text,
                project_id=project_id,
                embedding_vec=_axis(*([0.0] * i), 1.0),
                conn=pg,
            )
        exact = await upsert_fact(
            content="release v1.2.3 shipped",
            project_id=project_id,
            embedding_vec=_axis(*([0.0] * 4), 1.0),
            conn=pg,
        )
        query_vec = _axis(0.0, 1.0, 0.5, 0.25)
        with patch("mlms.tools.get_facts.embed", new=AsyncMock(return_value=query_vec)):
            result = await get_facts(
                about="v1.2.3", project_id=str(project_id), limit=3, conn=pg, redis=MagicMock()
            )
        assert len(result) == 3
        assert exact.id in {f.id for f in result}
