from __future__ import annotations

import pytest
from pydantic import ValidationError

from mlms.config import Settings


def test_invalid_database_url_error_does_not_echo_the_url() -> None:
    with pytest.raises(ValidationError) as exc:
        Settings(database_url="mysql://user:hunter2@db.example/mlms")
    assert "postgresql+psycopg://" in str(exc.value)
    assert "hunter2" not in str(exc.value)
