"""Unit tests for tools/memorize.py — no real DB or Redis required."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import ANY, AsyncMock, MagicMock, patch

from mlms.storage.facts import FactResult
from mlms.storage.session_ctx import SessionContext
from mlms.storage.session_log import SessionInsertResult
from mlms.storage.skills import SkillInsertResult
from mlms.storage.timeline import EventResult
from mlms.tools.memorize import _refresh_redis, memorize

_CHAT = "chat_test"
_PROJECT = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
_NOW = datetime(2024, 3, 10, 9, 0, 0, tzinfo=UTC)
_VEC = [0.1] * 1536
_FACT_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
_EVENT_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")
_SKILL_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")

_P_EMBED = "mlms.tools.memorize.embed"
_P_UPSERT = "mlms.tools.memorize.upsert_fact"
_P_EVENT = "mlms.tools.memorize.insert_event"
_P_ENTRY = "mlms.tools.memorize.insert_session_entry"
_P_SKILL = "mlms.tools.memorize.insert_skill"
_P_GET = "mlms.tools.memorize.get_session_context"
_P_SET = "mlms.tools.memorize.set_session_context"


def _conn() -> Any:
    return MagicMock()


def _redis() -> Any:
    return MagicMock()


def _fact_result() -> FactResult:
    return FactResult(id=_FACT_ID, version=1, is_conflict=False, replaced_id=None)


def _event_result() -> EventResult:
    return EventResult(id=_EVENT_ID, time=_NOW)


def _skill_result() -> SkillInsertResult:
    return SkillInsertResult(id=_SKILL_ID, name="my_skill", version=1, created_at=_NOW)


def _session_result() -> SessionInsertResult:
    return SessionInsertResult(time=_NOW, chat_id=_CHAT)


def _ctx() -> SessionContext:
    return SessionContext(project_id=_PROJECT, last_activity=_NOW)


# ── fact ──────────────────────────────────────────────────────────────────────


class TestMemorizeFact:
    async def test_embed_called_with_content(self) -> None:
        with (
            patch(_P_EMBED, new_callable=AsyncMock) as mock_embed,
            patch(_P_UPSERT, new_callable=AsyncMock) as mock_upsert,
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock),
        ):
            mock_embed.return_value = _VEC
            mock_upsert.return_value = _fact_result()
            await memorize(
                content="the fact",
                type="fact",
                metadata={"project_id": _PROJECT, "chat_id": _CHAT},
                conn=_conn(),
                redis=_redis(),
            )
            mock_embed.assert_called_once_with("the fact", ANY)

    async def test_upsert_fact_called(self) -> None:
        with (
            patch(_P_EMBED, new_callable=AsyncMock, return_value=_VEC),
            patch(_P_UPSERT, new_callable=AsyncMock) as mock_upsert,
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock),
        ):
            mock_upsert.return_value = _fact_result()
            await memorize(
                content="c",
                type="fact",
                metadata={"project_id": _PROJECT},
                conn=_conn(),
                redis=_redis(),
            )
            mock_upsert.assert_called_once()

    async def test_redis_updated_when_chat_id(self) -> None:
        with (
            patch(_P_EMBED, new_callable=AsyncMock, return_value=_VEC),
            patch(_P_UPSERT, new_callable=AsyncMock, return_value=_fact_result()),
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock) as mock_set,
        ):
            await memorize(
                content="c",
                type="fact",
                metadata={"project_id": _PROJECT, "chat_id": _CHAT},
                conn=_conn(),
                redis=_redis(),
            )
            mock_set.assert_called_once()

    async def test_no_redis_without_chat_id(self) -> None:
        with (
            patch(_P_EMBED, new_callable=AsyncMock, return_value=_VEC),
            patch(_P_UPSERT, new_callable=AsyncMock, return_value=_fact_result()),
            patch(_P_SET, new_callable=AsyncMock) as mock_set,
        ):
            await memorize(
                content="c",
                type="fact",
                metadata={"project_id": _PROJECT},
                conn=_conn(),
                redis=_redis(),
            )
            mock_set.assert_not_called()

    async def test_returns_ok_true_and_memory_id(self) -> None:
        with (
            patch(_P_EMBED, new_callable=AsyncMock, return_value=_VEC),
            patch(_P_UPSERT, new_callable=AsyncMock, return_value=_fact_result()),
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock),
        ):
            result = await memorize(
                content="c",
                type="fact",
                metadata={"project_id": _PROJECT, "chat_id": _CHAT},
                conn=_conn(),
                redis=_redis(),
            )
            assert result["ok"] is True
            assert result["memory_id"] == str(_FACT_ID)


# ── event ─────────────────────────────────────────────────────────────────────


class TestMemorizeEvent:
    async def test_insert_event_called(self) -> None:
        with (
            patch(_P_EVENT, new_callable=AsyncMock) as mock_event,
            patch(_P_ENTRY, new_callable=AsyncMock, return_value=_session_result()),
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock),
        ):
            mock_event.return_value = _event_result()
            await memorize(
                content="event body",
                type="event",
                metadata={
                    "project_id": _PROJECT,
                    "event_type": "decision",
                    "title": "chose postgres",
                    "chat_id": _CHAT,
                },
                conn=_conn(),
                redis=_redis(),
            )
            mock_event.assert_called_once()

    async def test_session_entry_created_when_chat_id(self) -> None:
        with (
            patch(_P_EVENT, new_callable=AsyncMock, return_value=_event_result()),
            patch(_P_ENTRY, new_callable=AsyncMock) as mock_entry,
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock),
        ):
            mock_entry.return_value = _session_result()
            await memorize(
                content="c",
                type="event",
                metadata={
                    "project_id": _PROJECT,
                    "event_type": "action",
                    "title": "deployed",
                    "chat_id": _CHAT,
                },
                conn=_conn(),
                redis=_redis(),
            )
            mock_entry.assert_called_once()

    async def test_no_session_entry_without_chat_id(self) -> None:
        with (
            patch(_P_EVENT, new_callable=AsyncMock, return_value=_event_result()),
            patch(_P_ENTRY, new_callable=AsyncMock) as mock_entry,
        ):
            await memorize(
                content="c",
                type="event",
                metadata={"project_id": _PROJECT, "event_type": "phase", "title": "start"},
                conn=_conn(),
                redis=_redis(),
            )
            mock_entry.assert_not_called()

    async def test_redis_updated_when_chat_id(self) -> None:
        with (
            patch(_P_EVENT, new_callable=AsyncMock, return_value=_event_result()),
            patch(_P_ENTRY, new_callable=AsyncMock, return_value=_session_result()),
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock) as mock_set,
        ):
            await memorize(
                content="c",
                type="event",
                metadata={
                    "project_id": _PROJECT,
                    "event_type": "action",
                    "title": "t",
                    "chat_id": _CHAT,
                },
                conn=_conn(),
                redis=_redis(),
            )
            mock_set.assert_called_once()

    async def test_returns_ok_true_and_event_id(self) -> None:
        with (
            patch(_P_EVENT, new_callable=AsyncMock, return_value=_event_result()),
            patch(_P_ENTRY, new_callable=AsyncMock, return_value=_session_result()),
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock),
        ):
            result = await memorize(
                content="c",
                type="event",
                metadata={
                    "project_id": _PROJECT,
                    "event_type": "decision",
                    "title": "t",
                    "chat_id": _CHAT,
                },
                conn=_conn(),
                redis=_redis(),
            )
            assert result["ok"] is True
            assert result["memory_id"] == str(_EVENT_ID)


# ── skill ─────────────────────────────────────────────────────────────────────


class TestMemorizeSkill:
    async def test_insert_skill_called(self) -> None:
        with (
            patch(_P_SKILL, new_callable=AsyncMock) as mock_skill,
            patch(_P_ENTRY, new_callable=AsyncMock, return_value=_session_result()),
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock),
        ):
            mock_skill.return_value = _skill_result()
            await memorize(
                content="do this when X",
                type="skill",
                metadata={"name": "my_skill", "chat_id": _CHAT},
                conn=_conn(),
                redis=_redis(),
            )
            mock_skill.assert_called_once()

    async def test_session_entry_created_when_chat_id(self) -> None:
        with (
            patch(_P_SKILL, new_callable=AsyncMock, return_value=_skill_result()),
            patch(_P_ENTRY, new_callable=AsyncMock) as mock_entry,
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock),
        ):
            mock_entry.return_value = _session_result()
            await memorize(
                content="c",
                type="skill",
                metadata={"name": "s", "chat_id": _CHAT},
                conn=_conn(),
                redis=_redis(),
            )
            mock_entry.assert_called_once()

    async def test_redis_updated_when_chat_id(self) -> None:
        with (
            patch(_P_SKILL, new_callable=AsyncMock, return_value=_skill_result()),
            patch(_P_ENTRY, new_callable=AsyncMock, return_value=_session_result()),
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock) as mock_set,
        ):
            await memorize(
                content="c",
                type="skill",
                metadata={"name": "s", "chat_id": _CHAT},
                conn=_conn(),
                redis=_redis(),
            )
            mock_set.assert_called_once()

    async def test_no_redis_without_chat_id(self) -> None:
        with (
            patch(_P_SKILL, new_callable=AsyncMock, return_value=_skill_result()),
            patch(_P_SET, new_callable=AsyncMock) as mock_set,
        ):
            await memorize(
                content="c",
                type="skill",
                metadata={"name": "s"},
                conn=_conn(),
                redis=_redis(),
            )
            mock_set.assert_not_called()

    async def test_returns_ok_true_and_skill_id(self) -> None:
        with (
            patch(_P_SKILL, new_callable=AsyncMock, return_value=_skill_result()),
            patch(_P_ENTRY, new_callable=AsyncMock, return_value=_session_result()),
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock),
        ):
            result = await memorize(
                content="c",
                type="skill",
                metadata={"name": "my_skill", "chat_id": _CHAT},
                conn=_conn(),
                redis=_redis(),
            )
            assert result["ok"] is True
            assert result["memory_id"] == str(_SKILL_ID)


# ── session_log ───────────────────────────────────────────────────────────────


class TestMemorizeSessionLog:
    async def test_embed_called_with_label_not_content(self) -> None:
        with (
            patch(_P_EMBED, new_callable=AsyncMock) as mock_embed,
            patch(_P_ENTRY, new_callable=AsyncMock, return_value=_session_result()),
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock),
        ):
            mock_embed.return_value = _VEC
            await memorize(
                content="long full content",
                type="session_log",
                metadata={"chat_id": _CHAT, "type": "topic", "label": "short label"},
                conn=_conn(),
                redis=_redis(),
            )
            mock_embed.assert_called_once_with("short label", ANY)

    async def test_insert_session_entry_called(self) -> None:
        with (
            patch(_P_EMBED, new_callable=AsyncMock, return_value=_VEC),
            patch(_P_ENTRY, new_callable=AsyncMock) as mock_entry,
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock),
        ):
            mock_entry.return_value = _session_result()
            await memorize(
                content="c",
                type="session_log",
                metadata={"chat_id": _CHAT, "type": "decision", "label": "lbl"},
                conn=_conn(),
                redis=_redis(),
            )
            mock_entry.assert_called_once()

    async def test_redis_always_updated(self) -> None:
        with (
            patch(_P_EMBED, new_callable=AsyncMock, return_value=_VEC),
            patch(_P_ENTRY, new_callable=AsyncMock, return_value=_session_result()),
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock) as mock_set,
        ):
            await memorize(
                content="c",
                type="session_log",
                metadata={"chat_id": _CHAT, "type": "insight", "label": "lbl"},
                conn=_conn(),
                redis=_redis(),
            )
            mock_set.assert_called_once()

    async def test_label_truncated_to_100_chars(self) -> None:
        long_label = "x" * 150
        with (
            patch(_P_EMBED, new_callable=AsyncMock) as mock_embed,
            patch(_P_ENTRY, new_callable=AsyncMock, return_value=_session_result()),
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock),
        ):
            mock_embed.return_value = _VEC
            await memorize(
                content="c",
                type="session_log",
                metadata={"chat_id": _CHAT, "type": "context", "label": long_label},
                conn=_conn(),
                redis=_redis(),
            )
            embedded_label: str = mock_embed.call_args[0][0]
            assert len(embedded_label) == 100

    async def test_returns_ok_true(self) -> None:
        with (
            patch(_P_EMBED, new_callable=AsyncMock, return_value=_VEC),
            patch(_P_ENTRY, new_callable=AsyncMock, return_value=_session_result()),
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock),
        ):
            result = await memorize(
                content="c",
                type="session_log",
                metadata={"chat_id": _CHAT, "type": "result", "label": "lbl"},
                conn=_conn(),
                redis=_redis(),
            )
            assert result["ok"] is True


# ── _refresh_redis ────────────────────────────────────────────────────────────


class TestRefreshRedis:
    async def test_creates_new_context_when_none(self) -> None:
        with (
            patch(_P_GET, new_callable=AsyncMock, return_value=None),
            patch(_P_SET, new_callable=AsyncMock) as mock_set,
        ):
            await _refresh_redis(_CHAT, _PROJECT, redis=_redis())
            ctx: SessionContext = mock_set.call_args[0][1]
            assert ctx.project_id == _PROJECT
            assert isinstance(ctx.last_activity, datetime)

    async def test_updates_last_activity_when_ctx_exists(self) -> None:
        old_ctx = _ctx()
        with (
            patch(_P_GET, new_callable=AsyncMock, return_value=old_ctx),
            patch(_P_SET, new_callable=AsyncMock) as mock_set,
        ):
            await _refresh_redis(_CHAT, _PROJECT, redis=_redis())
            ctx: SessionContext = mock_set.call_args[0][1]
            assert ctx.last_activity > _NOW

    async def test_preserves_existing_todos_and_entities(self) -> None:
        old_ctx = SessionContext(
            project_id=_PROJECT,
            last_activity=_NOW,
            open_todos=["fix bug"],
            active_entities=["auth.py"],
            recent_decisions=["use redis"],
        )
        with (
            patch(_P_GET, new_callable=AsyncMock, return_value=old_ctx),
            patch(_P_SET, new_callable=AsyncMock) as mock_set,
        ):
            await _refresh_redis(_CHAT, _PROJECT, redis=_redis())
            ctx: SessionContext = mock_set.call_args[0][1]
            assert ctx.open_todos == ["fix bug"]
            assert ctx.active_entities == ["auth.py"]
            assert ctx.recent_decisions == ["use redis"]
