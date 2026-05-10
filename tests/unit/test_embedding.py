"""Unit tests for embedding module — no DB/Redis/network required."""

from __future__ import annotations

import json
import math
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mlms.config import EMBEDDING_DIM, EMBEDDING_FALLBACK, EMBEDDING_MODEL
from mlms.embedding import _cache_key, _call_api, _retry_wait, _truncate_normalize, embed


@pytest.fixture
def redis() -> AsyncMock:
    r = AsyncMock()
    r.get = AsyncMock(return_value=None)  # cache miss by default
    r.setex = AsyncMock()
    return r


# ── _truncate_normalize ────────────────────────────────────────────────────


class TestTruncateNormalize:
    def test_truncates_to_embedding_dim(self) -> None:
        result = _truncate_normalize([1.0] * 4096)
        assert len(result) == EMBEDDING_DIM

    def test_l2_norm_is_one(self) -> None:
        vec = list(range(1, EMBEDDING_DIM + 100))
        result = _truncate_normalize(vec)
        norm = math.sqrt(sum(x * x for x in result))
        assert abs(norm - 1.0) < 1e-6

    def test_zero_vector_no_division_error(self) -> None:
        vec = [0.0] * EMBEDDING_DIM
        result = _truncate_normalize(vec)
        assert result == vec

    def test_already_correct_dim(self) -> None:
        result = _truncate_normalize([1.0] * EMBEDDING_DIM)
        assert len(result) == EMBEDDING_DIM


# ── _cache_key ─────────────────────────────────────────────────────────────


class TestCacheKey:
    def test_deterministic(self) -> None:
        assert _cache_key("hello", EMBEDDING_MODEL) == _cache_key("hello", EMBEDDING_MODEL)

    def test_model_differentiates(self) -> None:
        assert _cache_key("hello", EMBEDDING_MODEL) != _cache_key("hello", EMBEDDING_FALLBACK)

    def test_text_differentiates(self) -> None:
        assert _cache_key("hello", EMBEDDING_MODEL) != _cache_key("world", EMBEDDING_MODEL)

    def test_has_emb_prefix(self) -> None:
        assert _cache_key("x", EMBEDDING_MODEL).startswith("emb:")


# ── embed ──────────────────────────────────────────────────────────────────


class TestEmbed:
    async def test_cache_hit_primary(self, redis: AsyncMock) -> None:
        cached_vec = [0.1] * EMBEDDING_DIM
        redis.get = AsyncMock(return_value=json.dumps(cached_vec).encode())

        result = await embed("hello", redis)

        assert result == cached_vec
        redis.setex.assert_not_called()

    async def test_primary_api_success_caches_result(self, redis: AsyncMock) -> None:
        raw_vec = [1.0] * 4096

        with patch("mlms.embedding._call_api", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = raw_vec
            result = await embed("hello", redis)

        assert len(result) == EMBEDDING_DIM
        redis.setex.assert_called_once()
        cache_key_used = redis.setex.call_args[0][0]
        assert cache_key_used == _cache_key("hello", EMBEDDING_MODEL)

    async def test_fallback_called_on_primary_timeout(self, redis: AsyncMock) -> None:
        fallback_raw = [0.5] * EMBEDDING_DIM
        call_log: list[str] = []

        async def side_effect(text: str, model: str) -> list[float]:
            call_log.append(model)
            if model == EMBEDDING_MODEL:
                raise httpx.TimeoutException("timeout")
            return fallback_raw

        with patch("mlms.embedding._call_api", side_effect=side_effect):
            result = await embed("hello", redis)

        assert call_log == [EMBEDDING_MODEL, EMBEDDING_FALLBACK]
        assert len(result) == EMBEDDING_DIM
        cache_key_used = redis.setex.call_args[0][0]
        assert cache_key_used == _cache_key("hello", EMBEDDING_FALLBACK)

    async def test_fallback_called_on_primary_http_error(self, redis: AsyncMock) -> None:
        fallback_raw = [0.5] * EMBEDDING_DIM

        async def side_effect(text: str, model: str) -> list[float]:
            if model == EMBEDDING_MODEL:
                raise httpx.HTTPStatusError(
                    "500", request=httpx.Request("POST", "http://x"), response=httpx.Response(500)
                )
            return fallback_raw

        with patch("mlms.embedding._call_api", side_effect=side_effect):
            result = await embed("hello", redis)

        assert len(result) == EMBEDDING_DIM

    async def test_fallback_cache_hit_skips_api(self, redis: AsyncMock) -> None:
        """Primary miss + primary API fail → fallback cache hit → no API call for fallback."""
        fallback_vec = [0.3] * EMBEDDING_DIM
        fallback_key = _cache_key("hello", EMBEDDING_FALLBACK)

        async def get_side_effect(key: str) -> bytes | None:
            return json.dumps(fallback_vec).encode() if key == fallback_key else None

        redis.get = AsyncMock(side_effect=get_side_effect)

        async def fail_always(text: str, model: str) -> list[float]:
            raise httpx.TimeoutException("timeout")

        with patch("mlms.embedding._call_api", side_effect=fail_always):
            result = await embed("hello", redis)

        assert result == fallback_vec
        redis.setex.assert_not_called()

    async def test_both_models_fail_raises(self, redis: AsyncMock) -> None:
        async def fail_always(text: str, model: str) -> list[float]:
            raise httpx.TimeoutException("timeout")

        with patch("mlms.embedding._call_api", side_effect=fail_always):
            with pytest.raises(RuntimeError, match="Both embedding models"):
                await embed("hello", redis)

    async def test_result_is_l2_normalised(self, redis: AsyncMock) -> None:
        raw_vec = list(range(1, 4097))  # 4096-dim, un-normalized

        with patch("mlms.embedding._call_api", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = raw_vec
            result = await embed("hello", redis)

        norm = math.sqrt(sum(x * x for x in result))
        assert abs(norm - 1.0) < 1e-6


# ── _retry_wait ────────────────────────────────────────────────────────────


class TestRetryWait:
    def _make_response(self, status: int, retry_after: str | None = None) -> httpx.Response:
        headers = {"retry-after": retry_after} if retry_after is not None else {}
        return httpx.Response(status, headers=headers)

    def test_uses_retry_after_header(self) -> None:
        r = self._make_response(429, retry_after="5")
        assert _retry_wait(r, attempt=0) == 5.0

    def test_exponential_fallback_when_no_header(self) -> None:
        r = self._make_response(429)
        assert _retry_wait(r, attempt=0) == 1.0
        assert _retry_wait(r, attempt=1) == 2.0
        assert _retry_wait(r, attempt=2) == 4.0

    def test_exponential_fallback_on_unparseable_header(self) -> None:
        r = self._make_response(429, retry_after="not-a-number")
        assert _retry_wait(r, attempt=1) == 2.0


# ── _call_api ──────────────────────────────────────────────────────────────


_DUMMY_REQUEST = httpx.Request("POST", "http://test.invalid")


def _resp(status: int, **kwargs: object) -> httpx.Response:
    """Build an httpx.Response with a dummy request attached (required for raise_for_status)."""
    return httpx.Response(status, request=_DUMMY_REQUEST, **kwargs)


def _make_mock_client(responses: list[httpx.Response]) -> tuple[AsyncMock, MagicMock]:
    """Return (mock_client, MockClass) for patching httpx.AsyncClient."""
    call_iter = iter(responses)

    async def mock_post(*args: object, **kwargs: object) -> httpx.Response:
        return next(call_iter)

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=mock_post)
    mock_class = MagicMock()
    mock_class.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_class.return_value.__aexit__ = AsyncMock(return_value=False)
    return mock_client, mock_class


class TestCallApi:
    async def test_success_on_first_attempt(self) -> None:
        data = {"data": [{"embedding": [0.1] * 4096}]}
        _, mock_class = _make_mock_client([_resp(200, json=data)])

        with patch("mlms.embedding.httpx.AsyncClient", mock_class):
            result = await _call_api("hello", EMBEDDING_MODEL)

        assert len(result) == 4096

    async def test_retries_on_429_then_succeeds(self) -> None:
        data = {"data": [{"embedding": [0.1] * 4096}]}
        responses = [
            _resp(429, headers={"retry-after": "0"}),
            _resp(200, json=data),
        ]
        mock_client, mock_class = _make_mock_client(responses)

        with (
            patch("mlms.embedding.httpx.AsyncClient", mock_class),
            patch("mlms.embedding.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await _call_api("hello", EMBEDDING_MODEL)

        assert len(result) == 4096
        assert mock_client.post.call_count == 2
        mock_sleep.assert_called_once_with(0.0)

    async def test_raises_after_exhausting_retries(self) -> None:
        responses = [_resp(429, headers={"retry-after": "0"})] * 3
        _, mock_class = _make_mock_client(responses)

        with (
            patch("mlms.embedding.httpx.AsyncClient", mock_class),
            patch("mlms.embedding.asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await _call_api("hello", EMBEDDING_MODEL, max_retries=3)

        assert exc_info.value.response.status_code == 429

    async def test_no_retry_on_5xx(self) -> None:
        _, mock_class = _make_mock_client([_resp(500)])

        with patch("mlms.embedding.httpx.AsyncClient", mock_class):
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                await _call_api("hello", EMBEDDING_MODEL)

        assert exc_info.value.response.status_code == 500
