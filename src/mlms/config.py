from __future__ import annotations

from typing import Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

EMBEDDING_DIM: Final[int] = 1536          # Matryoshka truncation of Qwen3-8B; < 2000 HNSW limit
EMBEDDING_MODEL: Final[str] = "qwen3-embedding-8b"
EMBEDDING_FALLBACK: Final[str] = "qwen3-embedding-0.6b"

# 0.96 is a calibration starting point, not a fixed production value
COSINE_CONFLICT_THRESHOLD: Final[float] = 0.96

REDIS_SESSION_TTL: Final[int] = 172800    # 48 h
REDIS_KEY_PATTERN: Final[str] = "session:{chat_id}"

LATENCY_P99_TARGET_MS: Final[int] = 200
LATENCY_SESSION_CTX_MS: Final[int] = 5
PRECISION_AT_5_TARGET: Final[float] = 0.80


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://mlms:mlms@localhost:5432/mlms"
    redis_url: str = "redis://localhost:6379/0"

    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_api_base: str = Field(
        default="https://openrouter.ai/api/v1",
        alias="EMBEDDING_API_BASE",
    )

    mcp_transport: str = Field(default="stdio", alias="MCP_TRANSPORT")
    mcp_host: str = Field(default="0.0.0.0", alias="MCP_HOST")
    mcp_port: int = Field(default=8000, alias="MCP_PORT")


settings = Settings()
