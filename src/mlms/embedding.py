"""Qwen3 embedding API with Redis semantic cache and model fallback."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import math
from typing import Final, cast

import httpx
from redis.asyncio import Redis

from mlms.config import EMBEDDING_DIM, EMBEDDING_FALLBACK, EMBEDDING_MODEL, settings

logger = logging.getLogger(__name__)

_CACHE_PREFIX: Final = "emb:"
_CACHE_TTL: Final = 86400  # 24 h
_TIMEOUT: Final = httpx.Timeout(10.0, connect=5.0)


def _cache_key(text: str, model: str) -> str:
    digest = hashlib.sha256(f"{model}:{text}".encode()).hexdigest()
    return f"{_CACHE_PREFIX}{digest}"


def _truncate_normalize(vec: list[float]) -> list[float]:
    """Truncate to EMBEDDING_DIM and L2-normalize."""
    v = vec[:EMBEDDING_DIM]
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else v


def _retry_wait(response: httpx.Response, attempt: int) -> float:
    """Retry-After header if present, else exponential backoff."""
    header = response.headers.get("retry-after")
    if header is not None:
        with contextlib.suppress(ValueError):
            return float(header)
    return 2.0 ** attempt


async def _call_api(text: str, model: str, *, max_retries: int = 3) -> list[float]:
    """POST to /embeddings with retry on 429 and connection errors."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for attempt in range(max_retries):
            try:
                r = await client.post(
                    f"{settings.embedding_api_base}/embeddings",
                    headers={
                        "Authorization": f"Bearer {settings.embedding_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "input": text,
                        "dimensions": EMBEDDING_DIM,
                        "encoding_format": "float",
                    },
                )
            except (httpx.ConnectError, httpx.TransportError) as exc:
                if attempt == max_retries - 1:
                    raise
                wait = 2.0 ** (attempt + 1)
                logger.warning("connect error (attempt %d/%d) in %.1fs: %s", attempt + 1, max_retries, wait, exc)
                await asyncio.sleep(wait)
                continue
            if r.status_code != 429 or attempt == max_retries - 1:
                r.raise_for_status()
                return cast(list[float], r.json()["data"][0]["embedding"])
            wait = _retry_wait(r, attempt)
            logger.warning("rate limited, retry %d/%d in %.1fs", attempt + 1, max_retries, wait)
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")


async def embed(text: str, redis: Redis[bytes]) -> list[float]:
    """Return cached embedding or call API; falls back to smaller model on failure."""
    last_exc: Exception | None = None

    for model, label in (
        (EMBEDDING_MODEL, "primary"),
        (EMBEDDING_FALLBACK, "fallback"),
    ):
        key = _cache_key(text, model)

        cached = await redis.get(key)
        if cached is not None:
            logger.debug("embedding cache hit (%s)", label)
            return cast(list[float], json.loads(cached))

        try:
            vec = _truncate_normalize(await _call_api(text, model))
            await redis.setex(key, _CACHE_TTL, json.dumps(vec))
            logger.info("embedding via %s model", label)
            return vec
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.ConnectError) as exc:
            logger.warning("embedding %s failed: %s", label, exc)
            last_exc = exc

    raise RuntimeError("Both embedding models are unavailable") from last_exc
