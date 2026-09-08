from __future__ import annotations

from pathlib import Path

from mlms.server import _BIND_LOOPBACK_HOSTS

_COMPOSE_PATH = Path(__file__).resolve().parents[2] / "docker-compose.yml"


def _mlms_service_environment_block() -> str:
    lines = _COMPOSE_PATH.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "mlms:")
    # Next top-level (2-space-indented) service key, or end of file, ends the block.
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i] and not lines[i].startswith(" ")),
        len(lines),
    )
    return "\n".join(lines[start:end])


def test_compose_requires_shared_secret_for_its_non_loopback_bind() -> None:
    # Regression guard: this exact drift shipped once already — docker-compose.yml bound
    # the mlms service to MCP_HOST=0.0.0.0 without ever wiring MCP_SHARED_SECRET through,
    # so `docker compose up` hit Decision 23's own fail-fast on every start. Neither file's own
    # unit tests caught it — nothing read both files together. No YAML parser here on
    # purpose — pyyaml isn't a declared project dependency.
    block = _mlms_service_environment_block()
    host_line = next(line for line in block.splitlines() if "MCP_HOST:" in line)
    host = host_line.split("MCP_HOST:", 1)[1].strip()
    if host not in _BIND_LOOPBACK_HOSTS:
        assert "MCP_SHARED_SECRET" in block, (
            f"docker-compose.yml binds mlms to non-loopback MCP_HOST={host!r} but never "
            "sets MCP_SHARED_SECRET — the container will refuse to start (Decision 23)"
        )
