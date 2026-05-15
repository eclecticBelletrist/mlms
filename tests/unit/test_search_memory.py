from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlms.tools.search_memory import MemoryResult, search_memory

_VEC = [0.1] * 1536
_P_EMBED = "mlms.tools.search_memory.embed"
_P_FACTS = "mlms.tools.search_memory._search_facts"
_P_EVENTS = "mlms.tools.search_memory._search_events"
_P_SKILLS = "mlms.tools.search_memory._search_skills"


def _fact(score: float = 0.9) -> MemoryResult:
    return MemoryResult(layer="fact", id="f1", content="fact content", score=score)


def _event(score: float = 0.5) -> MemoryResult:
    return MemoryResult(layer="event", id="e1", content="event content", score=score)


def _skill(score: float = 0.5) -> MemoryResult:
    return MemoryResult(layer="skill", id="s1", content="skill content", score=score)


class TestSearchMemoryRouting:
    async def test_embed_called_with_query(self) -> None:
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)) as mock_embed:
            with patch(_P_FACTS, new=AsyncMock(return_value=[])):
                with patch(_P_EVENTS, new=AsyncMock(return_value=[])):
                    with patch(_P_SKILLS, new=AsyncMock(return_value=[])):
                        await search_memory(query="what is X", conn=MagicMock(), redis=MagicMock())
        mock_embed.assert_called_once()
        assert mock_embed.call_args[0][0] == "what is X"

    async def test_no_filter_calls_all_layers(self) -> None:
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)):
            with patch(_P_FACTS, new=AsyncMock(return_value=[])) as mf:
                with patch(_P_EVENTS, new=AsyncMock(return_value=[])) as me:
                    with patch(_P_SKILLS, new=AsyncMock(return_value=[])) as ms:
                        await search_memory(query="q", conn=MagicMock(), redis=MagicMock())
        mf.assert_called_once()
        me.assert_called_once()
        ms.assert_called_once()

    async def test_memory_type_fact_calls_only_facts(self) -> None:
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)):
            with patch(_P_FACTS, new=AsyncMock(return_value=[])) as mf:
                with patch(_P_EVENTS, new=AsyncMock(return_value=[])) as me:
                    with patch(_P_SKILLS, new=AsyncMock(return_value=[])) as ms:
                        await search_memory(
                            query="q",
                            filters={"memory_type": "fact"},
                            conn=MagicMock(),
                            redis=MagicMock(),
                        )
        mf.assert_called_once()
        me.assert_not_called()
        ms.assert_not_called()

    async def test_memory_type_event_calls_only_events(self) -> None:
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)):
            with patch(_P_FACTS, new=AsyncMock(return_value=[])) as mf:
                with patch(_P_EVENTS, new=AsyncMock(return_value=[])) as me:
                    with patch(_P_SKILLS, new=AsyncMock(return_value=[])) as ms:
                        await search_memory(
                            query="q",
                            filters={"memory_type": "event"},
                            conn=MagicMock(),
                            redis=MagicMock(),
                        )
        mf.assert_not_called()
        me.assert_called_once()
        ms.assert_not_called()

    async def test_memory_type_skill_calls_only_skills(self) -> None:
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)):
            with patch(_P_FACTS, new=AsyncMock(return_value=[])) as mf:
                with patch(_P_EVENTS, new=AsyncMock(return_value=[])) as me:
                    with patch(_P_SKILLS, new=AsyncMock(return_value=[])) as ms:
                        await search_memory(
                            query="q",
                            filters={"memory_type": "skill"},
                            conn=MagicMock(),
                            redis=MagicMock(),
                        )
        mf.assert_not_called()
        me.assert_not_called()
        ms.assert_called_once()


class TestSearchMemoryResults:
    async def test_results_sorted_by_score_descending(self) -> None:
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)):
            with patch(_P_FACTS, new=AsyncMock(return_value=[_fact(0.3)])):
                with patch(_P_EVENTS, new=AsyncMock(return_value=[_event(0.9)])):
                    with patch(_P_SKILLS, new=AsyncMock(return_value=[])):
                        result = await search_memory(
                            query="q", conn=MagicMock(), redis=MagicMock()
                        )
        assert result[0].score == 0.9
        assert result[1].score == 0.3

    async def test_returns_memory_result_pydantic(self) -> None:
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)):
            with patch(_P_FACTS, new=AsyncMock(return_value=[_fact()])):
                with patch(_P_EVENTS, new=AsyncMock(return_value=[])):
                    with patch(_P_SKILLS, new=AsyncMock(return_value=[])):
                        result = await search_memory(
                            query="q", conn=MagicMock(), redis=MagicMock()
                        )
        assert isinstance(result[0], MemoryResult)

    async def test_empty_all_layers_returns_empty(self) -> None:
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)):
            with patch(_P_FACTS, new=AsyncMock(return_value=[])):
                with patch(_P_EVENTS, new=AsyncMock(return_value=[])):
                    with patch(_P_SKILLS, new=AsyncMock(return_value=[])):
                        result = await search_memory(
                            query="q", conn=MagicMock(), redis=MagicMock()
                        )
        assert result == []

    async def test_project_id_filter_passed_to_facts(self) -> None:
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)):
            with patch(_P_FACTS, new=AsyncMock(return_value=[])) as mf:
                with patch(_P_EVENTS, new=AsyncMock(return_value=[])):
                    with patch(_P_SKILLS, new=AsyncMock(return_value=[])):
                        await search_memory(
                            query="q",
                            filters={"project_id": "pid-1"},
                            conn=MagicMock(),
                            redis=MagicMock(),
                        )
        assert mf.call_args[0][1] == "pid-1"  # project_id is 2nd positional arg

    async def test_filters_none_treated_as_empty(self) -> None:
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)):
            with patch(_P_FACTS, new=AsyncMock(return_value=[])):
                with patch(_P_EVENTS, new=AsyncMock(return_value=[])):
                    with patch(_P_SKILLS, new=AsyncMock(return_value=[])):
                        result = await search_memory(
                            query="q", filters=None, conn=MagicMock(), redis=MagicMock()
                        )
        assert result == []
