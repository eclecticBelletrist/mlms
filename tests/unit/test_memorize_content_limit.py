"""Unit tests for the memorize() content-length guard — no real DB or Redis required."""

from __future__ import annotations

import pytest

from mlms.config import MEMORY_CONTENT_MAX_LENGTH
from mlms.server import _check_content_length, mcp


def test_content_at_limit_passes() -> None:
    content = "x" * MEMORY_CONTENT_MAX_LENGTH
    assert _check_content_length(content) == content


def test_content_over_limit_rejected_with_actual_length_and_limit() -> None:
    content = "x" * (MEMORY_CONTENT_MAX_LENGTH + 1)
    with pytest.raises(ValueError, match=str(MEMORY_CONTENT_MAX_LENGTH + 1)) as exc_info:
        _check_content_length(content)
    assert str(MEMORY_CONTENT_MAX_LENGTH) in str(exc_info.value)


async def test_schema_exposes_max_length_to_clients() -> None:
    tools = await mcp.list_tools()
    memorize_tool = next(t for t in tools if t.name == "memorize")
    content_schema = memorize_tool.inputSchema["properties"]["content"]
    assert content_schema["maxLength"] == MEMORY_CONTENT_MAX_LENGTH
