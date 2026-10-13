"""Integration: skill filters see only the latest version of each skill.

MLMS_INTEGRATION=1 pytest tests/integration/test_skills.py -v
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import psycopg
import pytest

from mlms.storage.skills import get_skills, insert_skill

pytestmark = pytest.mark.skipif(
    os.getenv("MLMS_INTEGRATION") != "1",
    reason="Set MLMS_INTEGRATION=1 to run (requires docker compose up)",
)


async def _two_versions(pg: psycopg.AsyncConnection[Any]) -> tuple[str, str, str]:
    tag = uuid.uuid4().hex[:8]
    name, old_domain, new_domain = f"skill-{tag}", f"legacy-{tag}", f"modern-{tag}"
    await insert_skill(name=name, content=f"old recipe {tag}", domain=old_domain, conn=pg)
    await insert_skill(name=name, content=f"new recipe {tag}", domain=new_domain, conn=pg)
    return name, old_domain, new_domain


class TestLatestVersionFilters:
    async def test_domain_of_an_outdated_version_returns_nothing(
        self, pg: psycopg.AsyncConnection[Any]
    ) -> None:
        _, old_domain, _ = await _two_versions(pg)
        assert await get_skills(domain=old_domain, conn=pg) == []

    async def test_domain_of_the_latest_version_returns_it(
        self, pg: psycopg.AsyncConnection[Any]
    ) -> None:
        name, _, new_domain = await _two_versions(pg)
        result = await get_skills(domain=new_domain, conn=pg)
        assert [(s.name, s.version) for s in result] == [(name, 2)]

    async def test_search_text_of_an_outdated_version_returns_nothing(
        self, pg: psycopg.AsyncConnection[Any]
    ) -> None:
        name, _, _ = await _two_versions(pg)
        tag = name.removeprefix("skill-")
        assert await get_skills(search=f"old recipe {tag}", conn=pg) == []
