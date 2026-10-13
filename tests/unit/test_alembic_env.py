from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parents[2]


def test_percent_encoded_password_in_database_url_is_accepted() -> None:
    # --sql runs env.py offline, so no database is needed; the failure this guards against
    # is ConfigParser rejecting the "%" before any migration runs.
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head", "--sql"],
        cwd=_ROOT,
        env={**os.environ, "DATABASE_URL": "postgresql+psycopg://u:p%40ss@localhost/db"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
