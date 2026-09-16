"""Integration: Session-Augmented RAG two-step flow.

Step 1: get_session_log(chat_id) → retrieves session context terms
Step 2: get_facts(about=enriched_query) → uses session vocabulary for better retrieval

Run with: MLMS_INTEGRATION=1 pytest tests/integration/test_session_rag.py -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import psycopg
import pytest

from mlms.storage.session_log import get_session_log as _storage_get_session_log
from mlms.storage.session_log import insert_session_entry
from mlms.tools.get_facts import get_facts
from mlms.tools.get_session_log import get_session_log
from mlms.tools.memorize import memorize

pytestmark = pytest.mark.skipif(
    os.getenv("MLMS_INTEGRATION") != "1",
    reason="Set MLMS_INTEGRATION=1 to run (requires docker compose up)",
)

_VEC_PY = [1.0] + [0.0] * 1535  # "python" cluster
_VEC_SQL = [0.0, 1.0] + [0.0] * 1534  # "sql" cluster — orthogonal


class TestSessionLogRetrieval:
    async def test_get_session_log_returns_entries_for_chat(
        self, pg: psycopg.AsyncConnection[Any], rd: Any
    ) -> None:
        chat_id = f"chat-{uuid.uuid4()}"
        await insert_session_entry(
            chat_id=chat_id,
            entry_type="topic",
            label="Python garbage collection",
            content="Discussing Python GC mechanisms",
            conn=pg,
        )
        await pg.commit()

        rows = await get_session_log(chat_id=chat_id, conn=pg, redis=rd)
        assert len(rows) >= 1
        assert any("Python" in r.label for r in rows)

    async def test_get_session_log_excludes_other_chats(
        self, pg: psycopg.AsyncConnection[Any], rd: Any
    ) -> None:
        chat_a = f"chat-{uuid.uuid4()}"
        chat_b = f"chat-{uuid.uuid4()}"

        await insert_session_entry(chat_id=chat_a, entry_type="topic", label="A topic", conn=pg)
        await insert_session_entry(chat_id=chat_b, entry_type="topic", label="B topic", conn=pg)
        await pg.commit()

        rows = await get_session_log(chat_id=chat_a, conn=pg, redis=rd)
        assert all(r.chat_id == chat_a for r in rows)

    async def test_session_log_ordered_most_recent_first(
        self, pg: psycopg.AsyncConnection[Any], rd: Any
    ) -> None:
        chat_id = f"chat-{uuid.uuid4()}"
        for label in ("first", "second", "third"):
            await insert_session_entry(chat_id=chat_id, entry_type="action", label=label, conn=pg)
        await pg.commit()

        rows = await get_session_log(chat_id=chat_id, conn=pg, redis=rd)
        labels = [r.label for r in rows]
        assert labels.index("third") < labels.index("second") < labels.index("first")


class TestSessionAugmentedRAGFlow:
    """Two-step RAG: session context enriches the semantic query."""

    async def test_step1_get_session_log_step2_get_facts_no_error(
        self, pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID, rd: Any
    ) -> None:
        chat_id = f"chat-{uuid.uuid4()}"

        # Seed session context
        await insert_session_entry(
            chat_id=chat_id,
            entry_type="topic",
            label="Python memory and GC",
            content="Discussing Python garbage collection reference counting",
            conn=pg,
        )
        # Seed a fact
        with patch("mlms.tools.memorize.embed", new=AsyncMock(return_value=_VEC_PY)):
            await memorize(
                content="Python reference counting is the primary GC mechanism",
                type="fact",
                metadata={"project_id": str(project_id), "chat_id": chat_id},
                conn=pg,
                redis=rd,
            )
        await pg.commit()

        # Step 1: retrieve session terms
        session_entries = await get_session_log(chat_id=chat_id, conn=pg, redis=rd)
        assert len(session_entries) > 0
        session_terms = " ".join(e.label for e in session_entries)

        # Step 2: get_facts enriched with session vocabulary
        enriched_query = f"memory management {session_terms}"
        with patch("mlms.tools.get_facts.embed", new=AsyncMock(return_value=_VEC_PY)):
            facts = await get_facts(
                about=enriched_query,
                project_id=str(project_id),
                conn=pg,
                redis=rd,
            )

        # Result should not be empty (seeded fact matches)
        assert len(facts) >= 1

    async def test_session_rag_filters_irrelevant_domain(
        self, pg: psycopg.AsyncConnection[Any], project_id: uuid.UUID, rd: Any
    ) -> None:
        """Under limit=1, a fact aligned with the query embedding wins over one from
        an orthogonal cluster."""
        # Seed Python fact (VEC_PY cluster)
        with patch("mlms.tools.memorize.embed", new=AsyncMock(return_value=_VEC_PY)):
            await memorize(
                content="Python GC uses reference counting",
                type="fact",
                metadata={"project_id": str(project_id)},
                conn=pg,
                redis=rd,
            )

        # Seed SQL fact (VEC_SQL cluster — orthogonal)
        with patch("mlms.tools.memorize.embed", new=AsyncMock(return_value=_VEC_SQL)):
            await memorize(
                content="PostgreSQL uses MVCC for concurrency",
                type="fact",
                metadata={"project_id": str(project_id)},
                conn=pg,
                redis=rd,
            )
        await pg.commit()

        # Query aligned with Python cluster
        with patch("mlms.tools.get_facts.embed", new=AsyncMock(return_value=_VEC_PY)):
            facts = await get_facts(
                about="Python memory",
                project_id=str(project_id),
                limit=1,
                conn=pg,
                redis=rd,
            )

        assert len(facts) == 1
        assert "Python" in facts[0].content


class TestSessionLogSelectiveFilterWithHNSW:
    """Reproduces the pgvector recall pitfall behind 3.5: HNSW pulls its ef_search
    candidate window (default 40) BEFORE any WHERE filter runs, so a selective
    entry_type filter can lose a real match that isn't among the top ~40 nearest by
    raw distance. hnsw.iterative_scan (storage/session_log.py) is what keeps the
    index searching until the filtered LIMIT is actually filled.
    """

    async def test_selective_entry_type_survives_dense_decoy_field(
        self, pg: psycopg.AsyncConnection[Any]
    ) -> None:
        chat_id = f"chat-{uuid.uuid4()}"

        # 50 decoys at the exact query vector — all rank ahead of the target by
        # cosine distance, comfortably exceeding the default ef_search=40 window.
        for i in range(50):
            await insert_session_entry(
                chat_id=chat_id,
                entry_type="topic",
                label=f"decoy-{i}",
                embedding_vec=_VEC_PY,
                conn=pg,
            )

        # Target: the only "action" entry, embedding nudged off the query vector so
        # it ranks behind the decoy wall by raw distance.
        target_vec = [0.98, 0.02] + [0.0] * 1534
        await insert_session_entry(
            chat_id=chat_id,
            entry_type="action",
            label="the-target",
            embedding_vec=target_vec,
            conn=pg,
        )
        await pg.commit()

        rows = await _storage_get_session_log(
            semantic_query=_VEC_PY, entry_type="action", limit=5, conn=pg
        )

        assert any(r.label == "the-target" for r in rows)
