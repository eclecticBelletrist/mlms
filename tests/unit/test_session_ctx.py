"""Unit tests for storage/session_ctx.py — no real Redis required."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from mlms.config import REDIS_SESSION_TTL
from mlms.storage.session_ctx import (
    SessionContext,
    get_session_context,
    set_session_context,
)

_CHAT = "chat_abc"
_NOW = datetime(2024, 3, 10, 9, 0, 0, tzinfo=UTC)


def _make_redis(hgetall_result: dict[str, str] | None = None) -> Any:
    r: Any = MagicMock()
    r.hgetall = AsyncMock(return_value=hgetall_result if hgetall_result is not None else {})
    r.hset = AsyncMock(return_value=1)
    r.expire = AsyncMock(return_value=True)
    return r


def _raw_data(*, project_id: str = "proj-1") -> dict[str, str]:
    return {
        "project_id": project_id,
        "last_activity": _NOW.isoformat(),
        "open_todos": '["fix bug"]',
        "active_entities": '["auth.py"]',
        "recent_decisions": '["use redis"]',
    }


def _make_ctx(*, project_id: str = "proj-1") -> SessionContext:
    return SessionContext(
        project_id=project_id,
        last_activity=_NOW,
        open_todos=["fix bug"],
        active_entities=["auth.py"],
        recent_decisions=["use redis"],
    )


# ── get_session_context ───────────────────────────────────────────────────────


class TestGetSessionContext:
    async def test_returns_none_when_key_missing(self) -> None:
        redis = _make_redis({})
        result = await get_session_context(_CHAT, redis=redis)
        assert result is None

    async def test_expire_called_on_read(self) -> None:
        redis = _make_redis(_raw_data())
        await get_session_context(_CHAT, redis=redis)
        redis.expire.assert_called_once_with(f"session:{_CHAT}", REDIS_SESSION_TTL)

    async def test_expire_not_called_when_key_missing(self) -> None:
        redis = _make_redis({})
        await get_session_context(_CHAT, redis=redis)
        redis.expire.assert_not_called()

    async def test_returns_session_context_dataclass(self) -> None:
        redis = _make_redis(_raw_data())
        result = await get_session_context(_CHAT, redis=redis)
        assert isinstance(result, SessionContext)

    async def test_fields_deserialized_correctly(self) -> None:
        redis = _make_redis(_raw_data(project_id="proj-xyz"))
        result = await get_session_context(_CHAT, redis=redis)
        assert result is not None
        assert result.project_id == "proj-xyz"
        assert result.last_activity == _NOW
        assert result.open_todos == ["fix bug"]
        assert result.active_entities == ["auth.py"]
        assert result.recent_decisions == ["use redis"]

    async def test_key_pattern_correct(self) -> None:
        redis = _make_redis(_raw_data())
        await get_session_context("my_chat", redis=redis)
        redis.hgetall.assert_called_once_with("session:my_chat")


# ── set_session_context ───────────────────────────────────────────────────────


class TestSetSessionContext:
    async def test_hset_called(self) -> None:
        redis = _make_redis()
        await set_session_context(_CHAT, _make_ctx(), redis=redis)
        redis.hset.assert_called_once()

    async def test_expire_called_on_write(self) -> None:
        redis = _make_redis()
        await set_session_context(_CHAT, _make_ctx(), redis=redis)
        redis.expire.assert_called_once_with(f"session:{_CHAT}", REDIS_SESSION_TTL)

    async def test_expire_ttl_is_48h_not_24h(self) -> None:
        redis = _make_redis()
        await set_session_context(_CHAT, _make_ctx(), redis=redis)
        assert redis.expire.call_args[0][1] == 172800

    async def test_key_pattern_correct(self) -> None:
        redis = _make_redis()
        await set_session_context("xyz_chat", _make_ctx(), redis=redis)
        assert redis.hset.call_args[0][0] == "session:xyz_chat"

    async def test_project_id_in_mapping(self) -> None:
        redis = _make_redis()
        await set_session_context(_CHAT, _make_ctx(project_id="proj-99"), redis=redis)
        mapping: dict[str, str] = redis.hset.call_args[1]["mapping"]
        assert mapping["project_id"] == "proj-99"

    async def test_last_activity_iso_format(self) -> None:
        redis = _make_redis()
        await set_session_context(_CHAT, _make_ctx(), redis=redis)
        mapping: dict[str, str] = redis.hset.call_args[1]["mapping"]
        assert mapping["last_activity"] == _NOW.isoformat()

    async def test_lists_serialized_as_json(self) -> None:
        redis = _make_redis()
        await set_session_context(_CHAT, _make_ctx(), redis=redis)
        mapping: dict[str, str] = redis.hset.call_args[1]["mapping"]
        assert mapping["open_todos"] == '["fix bug"]'
        assert mapping["active_entities"] == '["auth.py"]'
        assert mapping["recent_decisions"] == '["use redis"]'


# ── round-trip ────────────────────────────────────────────────────────────────


class TestRoundtrip:
    async def test_serialize_deserialize_identity(self) -> None:
        ctx = SessionContext(
            project_id="proj-rt",
            last_activity=_NOW,
            open_todos=["a", "b"],
            active_entities=["x.py"],
            recent_decisions=["use postgres"],
        )
        set_redis = _make_redis()
        await set_session_context(_CHAT, ctx, redis=set_redis)
        written: dict[str, str] = set_redis.hset.call_args[1]["mapping"]

        get_redis = _make_redis(written)
        result = await get_session_context(_CHAT, redis=get_redis)

        assert result is not None
        assert result.project_id == ctx.project_id
        assert result.last_activity == ctx.last_activity
        assert result.open_todos == ctx.open_todos
        assert result.active_entities == ctx.active_entities
        assert result.recent_decisions == ctx.recent_decisions
