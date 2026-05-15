from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

from mlms.storage.session_ctx import SessionContext as StorageSessionContext
from mlms.tools.get_session_ctx import SessionContext, get_session_context

_NOW = datetime(2024, 3, 10, 9, 0, 0, tzinfo=UTC)
_P_GET = "mlms.tools.get_session_ctx._get_ctx"


def _make_storage_ctx(**kw: object) -> StorageSessionContext:
    defaults = {
        "project_id": "proj-1",
        "last_activity": _NOW,
        "open_todos": [],
        "active_entities": [],
        "recent_decisions": [],
    }
    return StorageSessionContext(**{**defaults, **kw})  # type: ignore[arg-type]


class TestGetSessionContext:
    async def test_returns_none_when_key_missing(self) -> None:
        with patch(_P_GET, new=AsyncMock(return_value=None)):
            result = await get_session_context(chat_id="c1", redis=MagicMock())
        assert result is None

    async def test_returns_session_context_pydantic(self) -> None:
        with patch(_P_GET, new=AsyncMock(return_value=_make_storage_ctx())):
            result = await get_session_context(chat_id="c1", redis=MagicMock())
        assert isinstance(result, SessionContext)

    async def test_project_id_passed_through(self) -> None:
        ctx = _make_storage_ctx(project_id="proj-xyz")
        with patch(_P_GET, new=AsyncMock(return_value=ctx)):
            result = await get_session_context(chat_id="c1", redis=MagicMock())
        assert result is not None
        assert result.project_id == "proj-xyz"

    async def test_last_activity_passed_through(self) -> None:
        ctx = _make_storage_ctx(last_activity=_NOW)
        with patch(_P_GET, new=AsyncMock(return_value=ctx)):
            result = await get_session_context(chat_id="c1", redis=MagicMock())
        assert result is not None
        assert result.last_activity == _NOW

    async def test_open_todos_passed_through(self) -> None:
        ctx = _make_storage_ctx(open_todos=["todo1", "todo2"])
        with patch(_P_GET, new=AsyncMock(return_value=ctx)):
            result = await get_session_context(chat_id="c1", redis=MagicMock())
        assert result is not None
        assert result.open_todos == ["todo1", "todo2"]

    async def test_active_entities_passed_through(self) -> None:
        ctx = _make_storage_ctx(active_entities=["file.py"])
        with patch(_P_GET, new=AsyncMock(return_value=ctx)):
            result = await get_session_context(chat_id="c1", redis=MagicMock())
        assert result is not None
        assert result.active_entities == ["file.py"]

    async def test_recent_decisions_passed_through(self) -> None:
        ctx = _make_storage_ctx(recent_decisions=["use HNSW"])
        with patch(_P_GET, new=AsyncMock(return_value=ctx)):
            result = await get_session_context(chat_id="c1", redis=MagicMock())
        assert result is not None
        assert result.recent_decisions == ["use HNSW"]
