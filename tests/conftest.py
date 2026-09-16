import asyncio
import sys

import pytest

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply unit/integration markers by directory so `pytest -m unit` etc. work.

    pyproject.toml declares these markers but no test file applies them — `pytest -m
    unit` silently selects 0 tests despite being the README's documented workflow.
    """
    for item in items:
        path = str(item.path)
        if "/tests/unit/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/tests/integration/" in path:
            item.add_marker(pytest.mark.integration)
