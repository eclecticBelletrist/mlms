from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mlms.storage.session_log import SessionEntry
from mlms.tools.get_session_log import SessionLogEntry, get_session_log

_NOW = datetime(2024, 3, 10, 9, 0, 0, tzinfo=UTC)
_VEC = [0.1] * 1536
_P_EMBED = "mlms.tools.get_session_log.embed"
_P_STORAGE = "mlms.tools.get_session_log._storage_get_session_log"


def _make_entry(**kw: Any) -> SessionEntry:
    defaults: dict[str, Any] = {
        "time": _NOW,
        "chat_id": "chat-1",
        "entry_type": "decision",
        "label": "decided to use HNSW",
        "content": "full rationale",
        "tags": None,
        "meta": None,
    }
    return SessionEntry(**{**defaults, **kw})


class TestGetSessionLogModes:
    async def test_raises_if_both_chat_id_and_semantic_query(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            await get_session_log(
                chat_id="c", semantic_query="q", conn=MagicMock(), redis=MagicMock()
            )

    async def test_returns_empty_if_neither_provided(self) -> None:
        result = await get_session_log(conn=MagicMock(), redis=MagicMock())
        assert result == []

    async def test_in_session_mode_no_embed(self) -> None:
        with patch(_P_STORAGE, new=AsyncMock(return_value=[])) as mock_storage:
            with patch(_P_EMBED, new=AsyncMock()) as mock_embed:
                await get_session_log(chat_id="c1", conn=MagicMock(), redis=MagicMock())
                mock_embed.assert_not_called()
                mock_storage.assert_called_once()

    async def test_in_session_passes_chat_id_to_storage(self) -> None:
        with patch(_P_STORAGE, new=AsyncMock(return_value=[])) as mock_storage:
            await get_session_log(chat_id="my-chat", conn=MagicMock(), redis=MagicMock())
        assert mock_storage.call_args.kwargs["chat_id"] == "my-chat"

    async def test_cross_session_embeds_query(self) -> None:
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)) as mock_embed:
            with patch(_P_STORAGE, new=AsyncMock(return_value=[])):
                await get_session_log(
                    semantic_query="find prior decisions", conn=MagicMock(), redis=MagicMock()
                )
        mock_embed.assert_called_once()

    async def test_cross_session_passes_vec_to_storage(self) -> None:
        with patch(_P_EMBED, new=AsyncMock(return_value=_VEC)):
            with patch(_P_STORAGE, new=AsyncMock(return_value=[])) as mock_storage:
                await get_session_log(
                    semantic_query="q", conn=MagicMock(), redis=MagicMock()
                )
        assert mock_storage.call_args.kwargs["semantic_query"] == _VEC


class TestGetSessionLogFiltering:
    async def test_entry_type_filter_applied(self) -> None:
        entries = [
            _make_entry(entry_type="decision"),
            _make_entry(entry_type="action"),
        ]
        with patch(_P_STORAGE, new=AsyncMock(return_value=entries)):
            result = await get_session_log(
                chat_id="c", entry_type="decision", conn=MagicMock(), redis=MagicMock()
            )
        assert len(result) == 1
        assert result[0].entry_type == "decision"

    async def test_no_filter_returns_all(self) -> None:
        entries = [_make_entry(entry_type=t) for t in ("decision", "action", "topic")]
        with patch(_P_STORAGE, new=AsyncMock(return_value=entries)):
            result = await get_session_log(
                chat_id="c", conn=MagicMock(), redis=MagicMock()
            )
        assert len(result) == 3

    async def test_returns_session_log_entry_pydantic(self) -> None:
        with patch(_P_STORAGE, new=AsyncMock(return_value=[_make_entry()])):
            result = await get_session_log(chat_id="c", conn=MagicMock(), redis=MagicMock())
        assert isinstance(result[0], SessionLogEntry)
        assert result[0].label == "decided to use HNSW"
