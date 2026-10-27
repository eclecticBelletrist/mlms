from __future__ import annotations

import re

from mlms.server import _INSTRUCTIONS, mcp


def test_instructions_are_sent_in_the_initialize_response() -> None:
    options = mcp._mcp_server.create_initialization_options()
    assert options.instructions == _INSTRUCTIONS


async def test_every_tool_named_in_the_instructions_is_registered() -> None:
    registered = {t.name for t in await mcp.list_tools()}
    named = set(re.findall(r"`(\w+)`", _INSTRUCTIONS))
    assert named
    assert named <= registered, named - registered
