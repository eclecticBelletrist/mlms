from __future__ import annotations

from mlms.server import mcp

# Tools that hit the external Qwen embedding API conditionally (get_facts/get_session_log)
# or unconditionally in some branch (memorize, search_memory) — openWorldHint must be True.
_OPEN_WORLD_TRUE = {"memorize", "get_facts", "get_session_log", "search_memory"}
_OPEN_WORLD_FALSE = {
    "get_timeline",
    "get_session_context",
    "get_skills",
    "revert_fact",
    "export_project",
}


async def test_open_world_hint_matches_external_api_usage() -> None:
    tools = {t.name: t for t in await mcp.list_tools()}
    for name in _OPEN_WORLD_TRUE:
        assert tools[name].annotations.openWorldHint is True, name
    for name in _OPEN_WORLD_FALSE:
        assert tools[name].annotations.openWorldHint is False, name


async def test_read_only_tools_have_no_effect_hints() -> None:
    read_only = {
        "get_facts",
        "get_timeline",
        "get_session_log",
        "get_session_context",
        "get_skills",
        "search_memory",
        "export_project",
    }
    tools = {t.name: t for t in await mcp.list_tools()}
    for name in read_only:
        ann = tools[name].annotations
        assert ann.readOnlyHint is True, name
        assert ann.destructiveHint is None, name
        assert ann.idempotentHint is None, name


async def test_write_tools_annotated_non_destructive_non_idempotent() -> None:
    tools = {t.name: t for t in await mcp.list_tools()}
    for name in ("memorize", "revert_fact"):
        ann = tools[name].annotations
        assert ann.readOnlyHint is False, name
        assert ann.destructiveHint is False, name
        assert ann.idempotentHint is False, name
