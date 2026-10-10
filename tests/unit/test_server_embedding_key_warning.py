from __future__ import annotations

import logging

import pytest

from mlms import server
from mlms.config import settings


def test_warns_when_embedding_key_is_empty(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "embedding_api_key", "")
    with caplog.at_level(logging.WARNING, logger=server._log.name):
        server._warn_if_no_embedding_key()
    assert "EMBEDDING_API_KEY is empty" in caplog.text


def test_silent_when_embedding_key_is_set(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "embedding_api_key", "k")
    with caplog.at_level(logging.WARNING, logger=server._log.name):
        server._warn_if_no_embedding_key()
    assert caplog.text == ""
