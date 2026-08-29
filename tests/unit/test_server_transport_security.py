from __future__ import annotations

from mlms.server import mcp


def test_dns_rebinding_protection_enabled() -> None:
    security = mcp.settings.transport_security
    assert security.enable_dns_rebinding_protection is True
    assert "localhost" in security.allowed_hosts
    assert "127.0.0.1" in security.allowed_hosts
    assert "localhost" in security.allowed_origins
    assert "127.0.0.1" in security.allowed_origins
