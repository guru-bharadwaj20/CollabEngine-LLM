"""The Gemini judge's wire format and rate-limit behavior.

Exercised against a stub transport rather than the live API: the free tier
allows five requests a minute, so a test suite that called it for real would be
both slow and quota-destroying. What matters here is that the request shape is
right, that a 429 is obeyed rather than fought, and that a failure degrades the
message instead of taking down a multi-hour coding run.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from collabengine.backends.base import ChatMessage, GenRequest
from collabengine.backends.gemini_judge import (
    GeminiJudgeBackend,
    _RateLimiter,
    _retry_delay,
)


def _backend(handler, **kw) -> GeminiJudgeBackend:
    backend = GeminiJudgeBackend(api_key="test-key", requests_per_minute=6000, **kw)
    backend._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://stub/v1beta",
        headers={"x-goog-api-key": "test-key"},
    )
    return backend


def _request() -> GenRequest:
    return GenRequest(
        messages=[
            ChatMessage(role="system", content="Choose one label."),
            ChatMessage(role="user", content="I checked the totals."),
        ],
        max_tokens=8,
        temperature=0.0,
    )


def _ok(text: str = "verify") -> dict:
    return {
        "candidates": [
            {"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 200, "candidatesTokenCount": 1},
    }


async def test_request_shape_matches_the_api() -> None:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.read() and __import__("json").loads(request.read()))
        return httpx.Response(200, json=_ok())

    backend = _backend(handler)
    response = await backend.generate(_request())

    assert response.ok and response.text == "verify"
    # The system prompt is its own field, not a message.
    assert seen["systemInstruction"]["parts"][0]["text"] == "Choose one label."
    assert [c["role"] for c in seen["contents"]] == ["user"]
    # Thinking off: 2.5 spends thinking tokens against maxOutputTokens, which at
    # an 8-token cap can consume the whole reply.
    assert seen["generationConfig"]["thinkingConfig"]["thinkingBudget"] == 0


async def test_a_429_waits_for_the_delay_the_server_asked_for() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "quota",
                        "details": [
                            {
                                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                                "retryDelay": "0.05s",
                            }
                        ],
                    }
                },
            )
        return httpx.Response(200, json=_ok("compute"))

    backend = _backend(handler)
    response = await backend.generate(_request())

    assert response.ok and response.text == "compute"
    assert len(calls) == 2


def test_retry_delay_is_parsed_from_retry_info() -> None:
    response = httpx.Response(
        429,
        json={
            "error": {
                "details": [
                    {
                        "@type": "type.googleapis.com/google.rpc.RetryInfo",
                        "retryDelay": "37s",
                    }
                ]
            }
        },
    )
    assert _retry_delay(response) == pytest.approx(38.0)


def test_retry_delay_absent_is_not_an_error() -> None:
    assert _retry_delay(httpx.Response(429, json={"error": {}})) is None


async def test_a_bad_key_fails_fast_rather_than_retrying() -> None:
    """A 401 will not fix itself; retrying spends the run's time on a wall."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    response = await _backend(handler).generate(_request())

    assert not response.ok
    assert "401" in (response.error or "")
    assert len(calls) == 1


async def test_a_blocked_prompt_degrades_the_message_not_the_run() -> None:
    """Safety filtering returns 200 with no candidate at all."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

    response = await _backend(handler).generate(_request())

    assert not response.ok
    assert "SAFETY" in (response.error or "")


async def test_the_limiter_paces_requests() -> None:
    """Bursting spends a per-minute allowance in the first second, then 429s for
    the rest of the window -- pacing is what keeps the quota usable."""
    limiter = _RateLimiter(per_minute=600)  # 0.1s apart
    start = asyncio.get_running_loop().time()
    for _ in range(3):
        await limiter.acquire()
    elapsed = asyncio.get_running_loop().time() - start

    assert elapsed >= 0.15


def test_config_builds_the_gemini_judge() -> None:
    from collabengine.config import ExperimentConfig

    config = ExperimentConfig.from_dict(
        {"backend": {"kind": "gemini", "model": "gemini-2.5-flash", "api_key": "k"}}
    )
    backend = config.backend.build()
    assert isinstance(backend, GeminiJudgeBackend)
    assert backend.model == "gemini-2.5-flash"
