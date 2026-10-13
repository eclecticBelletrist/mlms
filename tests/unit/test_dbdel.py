from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "dbdel", Path(__file__).parents[2] / "scripts" / "dbdel.py"
)
assert _SPEC is not None and _SPEC.loader is not None
dbdel = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dbdel)


@pytest.mark.parametrize("argv", [["dbdel.py"], ["dbdel.py", "--yes"]])
def test_refuses_url_without_database_name(argv: list[str]) -> None:
    with (
        patch.object(dbdel.settings, "database_url", "postgresql+psycopg://u:p@localhost:5432"),
        patch.object(dbdel.sys, "argv", argv),
        patch.object(dbdel.psycopg, "connect") as connect,
        pytest.raises(SystemExit, match="no database name"),
    ):
        dbdel.main()
    connect.assert_not_called()
