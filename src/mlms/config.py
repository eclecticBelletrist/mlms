from __future__ import annotations

from typing import Final, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EMBEDDING_DIM: Final[int] = 1536  # Matryoshka truncation of Qwen3-8B; < 2000 HNSW limit
EMBEDDING_MODEL: Final[str] = "qwen3-embedding-8b"
EMBEDDING_FALLBACK: Final[str] = "qwen3-embedding-0.6b"

# 0.96 is a calibration starting point, not a fixed production value
COSINE_CONFLICT_THRESHOLD: Final[float] = 0.96

REDIS_SESSION_TTL: Final[int] = 172800  # 48 h
REDIS_KEY_PATTERN: Final[str] = "session:{chat_id}"

LATENCY_P99_TARGET_MS: Final[int] = 200
LATENCY_SESSION_CTX_MS: Final[int] = 5
PRECISION_AT_5_TARGET: Final[float] = 0.80

# Hard reject limit for memorize(content). Sized off the *weaker* fallback model
# (qwen3-embedding-0.6b, 8192-token limit) with a pessimistic Cyrillic char/token
# ratio (~1.5, not the ~3 that holds for English) so the cap holds under either
# embedding model: 8192 * 1.5 ≈ 12288, rounded down to 10000 for margin.
MEMORY_CONTENT_MAX_LENGTH: Final[int] = 10_000
# Soft guidance only (not enforced) — atomic facts retrieve more precisely than
# long composite ones; see memorize()'s docstring.
MEMORY_CONTENT_RECOMMENDED_LENGTH: Final[int] = 500


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://mlms:mlms@localhost:5432/mlms"
    redis_url: str = "redis://localhost:6379/0"

    @field_validator("database_url")
    @classmethod
    def _check_database_url_scheme(cls, v: str) -> str:
        if not v.startswith("postgresql+psycopg://"):
            raise ValueError(f"DATABASE_URL must start with 'postgresql+psycopg://', got: {v!r}")
        return v

    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_api_base: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="EMBEDDING_API_BASE",
    )

    mcp_transport: Literal["stdio", "sse", "streamable-http"] = Field(
        default="stdio", alias="MCP_TRANSPORT"
    )
    mcp_host: str = Field(default="0.0.0.0", alias="MCP_HOST")
    mcp_port: int = Field(default=8000, alias="MCP_PORT")

    # Shared-secret gate for sse/streamable-http only — stdio has no network port, so this
    # is never checked there. None on a loopback host keeps the stdio/desktop and loopback
    # HTTP behavior; None on a non-loopback host is a startup fail-fast, not a silent
    # pass-through. See Decision 23.
    mcp_shared_secret: str | None = Field(default=None, alias="MCP_SHARED_SECRET")


def pg_dsn(url: str) -> str:
    return url.replace("postgresql+psycopg://", "postgresql://", 1)


settings = Settings()
