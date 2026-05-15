from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from mlms.storage.skills import Skill as StorageSkill
from mlms.tools.get_skills import Skill, get_skills

_NOW = datetime(2024, 3, 10, 9, 0, 0, tzinfo=UTC)
_P_STORAGE = "mlms.tools.get_skills._storage_get_skills"


def _make_storage_skill(**kw: Any) -> StorageSkill:
    defaults: dict[str, Any] = {
        "id": uuid.uuid4(),
        "name": "my_skill",
        "domain": "general",
        "content": "do the thing",
        "trigger_conditions": None,
        "version": 1,
        "created_at": _NOW,
    }
    return StorageSkill(**{**defaults, **kw})


class TestGetSkills:
    async def test_returns_list_of_skill(self) -> None:
        with patch(_P_STORAGE, new=AsyncMock(return_value=[_make_storage_skill()])):
            result = await get_skills(conn=MagicMock())
        assert len(result) == 1
        assert isinstance(result[0], Skill)

    async def test_empty_storage_returns_empty(self) -> None:
        with patch(_P_STORAGE, new=AsyncMock(return_value=[])):
            result = await get_skills(conn=MagicMock())
        assert result == []

    async def test_domain_filter_excludes_non_matching(self) -> None:
        skills = [
            _make_storage_skill(domain="sql"),
            _make_storage_skill(domain="python"),
        ]
        with patch(_P_STORAGE, new=AsyncMock(return_value=skills)):
            result = await get_skills(domain="sql", conn=MagicMock())
        assert len(result) == 1
        assert result[0].domain == "sql"

    async def test_domain_none_returns_all(self) -> None:
        skills = [_make_storage_skill(domain=d) for d in ("sql", "python", "general")]
        with patch(_P_STORAGE, new=AsyncMock(return_value=skills)):
            result = await get_skills(conn=MagicMock())
        assert len(result) == 3

    async def test_limit_passed_to_storage(self) -> None:
        with patch(_P_STORAGE, new=AsyncMock(return_value=[])) as mock_storage:
            await get_skills(limit=5, conn=MagicMock())
        assert mock_storage.call_args.kwargs["limit"] == 5

    async def test_fields_mapped_correctly(self) -> None:
        s = _make_storage_skill(
            name="sql_skill", domain="database", content="use indexes", version=3
        )
        with patch(_P_STORAGE, new=AsyncMock(return_value=[s])):
            result = await get_skills(conn=MagicMock())
        r = result[0]
        assert r.name == "sql_skill"
        assert r.domain == "database"
        assert r.content == "use indexes"
        assert r.version == 3

    async def test_domain_filter_on_none_domain_skill(self) -> None:
        skills = [
            _make_storage_skill(domain=None),
            _make_storage_skill(domain="sql"),
        ]
        with patch(_P_STORAGE, new=AsyncMock(return_value=skills)):
            result = await get_skills(domain="sql", conn=MagicMock())
        assert len(result) == 1
