from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import FastMCP
from starlette.testclient import TestClient

from mlms.server import mcp


def test_dns_rebinding_protection_enabled() -> None:
    security = mcp.settings.transport_security
    assert security.enable_dns_rebinding_protection is True
    assert "localhost" in security.allowed_hosts
    assert "127.0.0.1" in security.allowed_hosts
    # Origin headers always carry a scheme ("http://127.0.0.1"), unlike Host — a bare
    # "127.0.0.1" entry here would silently reject every real Origin, loopback included.
    assert "http://127.0.0.1" in security.allowed_origins
    assert "http://localhost" in security.allowed_origins


@asynccontextmanager
async def _noop_lifespan(_server: FastMCP) -> Any:
    yield {}


def _security_test_app() -> Any:
    """A FastMCP app carrying the real server's transport_security settings, but a
    no-op lifespan instead of mlms.server's real one.

    The real lifespan opens a live AsyncConnectionPool to Postgres — irrelevant to
    what this test checks (the Host/Origin middleware) and the cause of a real hang:
    TestClient's `with` block runs the app's actual ASGI lifespan, and psycopg_pool's
    background reconnect worker doesn't reliably die on pool.close() during shutdown
    (reproduces with Postgres up and reachable, not just down — see KEY_DECISIONS.md
    Decision 27). Building a separate app here sidesteps that bug entirely rather than
    working around it, since this test has no business touching the DB pool at all.
    """
    test_mcp = FastMCP("test-transport-security", lifespan=_noop_lifespan)
    test_mcp.settings.transport_security = mcp.settings.transport_security
    return test_mcp.streamable_http_app()


def test_legitimate_loopback_origin_is_not_rejected() -> None:
    # Regression test: allowed_origins used to reuse the scheme-less Host allowlist, so
    # the SDK's Origin check never matched a real Origin header (which always includes a
    # scheme) — every request carrying an Origin, including a legitimate loopback one, was
    # rejected with 403. Caught by testing against a live server, not by reading the SDK
    # source. Goes through a real HTTP request against the actual Starlette app (not the
    # SDK's private _validate_origin method directly) so it survives an SDK version bump
    # that renames or removes that internal method. TestClient's `with` block runs the
    # app's real ASGI lifespan (starts the session manager's task group) — hitting the
    # app directly over an ASGITransport without it raises "Task group is not initialized".
    # `.stream()` instead of `.get()`: a request that passes the security check opens an
    # SSE stream that stays open waiting for events — reading the body (what `.get()` does)
    # hangs forever. We only need the status code the security check produced, so we read
    # the headers and close without consuming the body.
    with TestClient(_security_test_app()) as client:
        with client.stream("GET", "/mcp", headers={"Host": "evil.example.com"}) as bad_host:
            assert bad_host.status_code == 421

        with client.stream("GET", "/mcp", headers={"Host": "127.0.0.1"}) as good_host:
            assert good_host.status_code != 421

        with client.stream(
            "GET", "/mcp", headers={"Host": "127.0.0.1", "Origin": "http://127.0.0.1"}
        ) as legit_origin:
            assert legit_origin.status_code != 403

        with client.stream(
            "GET",
            "/mcp",
            headers={"Host": "127.0.0.1", "Origin": "http://127.0.0.1:8000"},
        ) as legit_origin_with_port:
            assert legit_origin_with_port.status_code != 403

        with client.stream(
            "GET",
            "/mcp",
            headers={"Host": "127.0.0.1", "Origin": "https://evil.example.com"},
        ) as evil_origin:
            assert evil_origin.status_code == 403
