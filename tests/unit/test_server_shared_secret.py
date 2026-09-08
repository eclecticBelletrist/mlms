from __future__ import annotations

import anyio
import pytest
from starlette.testclient import TestClient

from mlms.server import _require_secret_on_non_loopback_bind, _SharedSecretMiddleware


async def _ok_app(scope: object, receive: object, send: object) -> None:
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": b"ok"})


def test_empty_secret_rejected_at_construction() -> None:
    # Invariant from Decision 23: an empty/falsy secret must never mean "allow everything" —
    # enforced by refusing to even build the middleware, not by a runtime branch.
    with pytest.raises(ValueError):
        _SharedSecretMiddleware(_ok_app, "")


def test_missing_token_rejected() -> None:
    # No `with TestClient(...)` block: that would run the ASGI lifespan protocol, which
    # this bare mock app doesn't implement — plain requests skip it entirely.
    client = TestClient(_SharedSecretMiddleware(_ok_app, "s3cr3t"))
    resp = client.get("/anything")
    assert resp.status_code == 401


def test_wrong_token_rejected() -> None:
    client = TestClient(_SharedSecretMiddleware(_ok_app, "s3cr3t"))
    resp = client.get("/anything", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_correct_token_passes_through_to_inner_app() -> None:
    client = TestClient(_SharedSecretMiddleware(_ok_app, "s3cr3t"))
    resp = client.get("/anything", headers={"Authorization": "Bearer s3cr3t"})
    assert resp.status_code == 200
    assert resp.text == "ok"


def test_non_http_scope_passes_through_untouched() -> None:
    # Lifespan events (pool/redis startup in _lifespan) must never be gated — the
    # middleware only inspects headers on http/websocket scopes.
    seen: list[str] = []

    async def inner_app(scope: object, receive: object, send: object) -> None:
        seen.append("called")

    app = _SharedSecretMiddleware(inner_app, "s3cr3t")

    async def run() -> None:
        async def receive() -> dict[str, str]:
            return {"type": "lifespan.startup"}

        async def send(_message: dict[str, str]) -> None:
            return None

        await app({"type": "lifespan"}, receive, send)

    anyio.run(run)
    assert seen == ["called"]


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_loopback_bind_never_requires_secret(host: str) -> None:
    # The original stdio/desktop posture (and loopback HTTP) is untouched by Decision 23.
    _require_secret_on_non_loopback_bind(host, None, "streamable-http")


def test_non_loopback_bind_without_secret_fails_fast() -> None:
    with pytest.raises(SystemExit):
        _require_secret_on_non_loopback_bind("0.0.0.0", None, "streamable-http")


def test_non_loopback_bind_with_secret_is_allowed() -> None:
    _require_secret_on_non_loopback_bind("0.0.0.0", "s3cr3t", "streamable-http")
