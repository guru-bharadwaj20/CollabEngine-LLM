"""Gemini judge backend, for behavioral coding only.

The same role `anthropic_judge` fills, against Google's API: it reads finished
transcripts and never runs an episode. Using a frontier model to *produce* agent
turns would destroy the one-model control the design rests on; using one to
*code* text that is already written introduces no confound.

Talks to the REST endpoint through `httpx` rather than adding an SDK. httpx is
already a core dependency, the surface used here is one POST, and the error body
carries the quota detail this backend has to act on.

**Rate limiting is the design constraint, not an afterthought.** The free tier
allows five requests per minute per model, measured against this key. That is
~300 messages an hour, so coding a corpus is a background job running for hours
rather than a step in a pipeline -- and a client that simply fires requests gets
429s for its whole quota window instead of the five calls it was entitled to. So
requests are paced by a token bucket, and a 429 is obeyed rather than retried
blindly: the response carries a `RetryInfo.retryDelay`, and sleeping exactly
that long is the difference between resuming in 37 seconds and hammering a
closed door.

Two models on this key answer: `gemini-2.5-flash` and `gemini-3-flash-preview`.
They have separate quotas, so the two-judge reliability run costs no extra
wall clock. They are not two *families* though, which is what PLAN.md asks for
-- a kappa between two Google models is a weaker check than one across vendors,
because correlated errors are likelier. Report it as such.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from collabengine.backends.base import GenRequest, GenResponse, LLMBackend

DEFAULT_JUDGE_MODEL = "gemini-2.5-flash"
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"


class _RateLimiter:
    """Token bucket. Paces requests to a sustainable rate.

    The alternative -- fire and retry on 429 -- wastes the quota window it just
    tripped, because the limit is per minute and a burst spends the whole
    allowance in the first second.
    """

    def __init__(self, per_minute: float) -> None:
        self.interval = 60.0 / max(per_minute, 0.1)
        self._next = 0.0
        self._lock: asyncio.Lock | None = None
        self._loop: Any = None

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._loop, self._lock = loop, asyncio.Lock()
        async with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if wait:
            await asyncio.sleep(wait)


@dataclass
class GeminiJudgeBackend(LLMBackend):
    """Chat client for Google's generative-language API, shaped as a backend."""

    model: str = DEFAULT_JUDGE_MODEL
    api_key: str | None = None
    requests_per_minute: float = 5.0
    """Free-tier allowance per model, measured against this key.

    Raise it only with evidence: exceeding it does not queue, it 429s."""
    max_retries: int = 6
    timeout_s: float = 120.0
    thinking_budget: int = 0
    """Gemini 2.5 spends thinking tokens by default and they count against
    `maxOutputTokens`. For an eight-label classification that is budget spent to
    produce the same word, and at a small token cap it can consume the reply
    entirely -- the same trap as Qwen3's default `<think>` block."""
    name: str = "gemini_judge"

    _client: Any = field(default=None, init=False, repr=False)
    _limiter: Any = field(default=None, init=False, repr=False)

    def _ensure(self) -> tuple[httpx.AsyncClient, _RateLimiter]:
        if self._client is None:
            key = self.api_key or os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError(
                    "no GEMINI_API_KEY; set it or pass api_key= to run the "
                    "Gemini judge"
                )
            self._client = httpx.AsyncClient(
                base_url=API_ROOT,
                headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                timeout=httpx.Timeout(self.timeout_s),
            )
        if self._limiter is None:
            self._limiter = _RateLimiter(self.requests_per_minute)
        return self._client, self._limiter

    def _payload(self, request: GenRequest) -> dict[str, Any]:
        system = "\n\n".join(
            m.content for m in request.messages if m.role == "system"
        )
        contents = [
            {
                "role": "model" if m.role == "assistant" else "user",
                "parts": [{"text": m.content}],
            }
            for m in request.messages
            if m.role != "system"
        ]
        generation: dict[str, Any] = {
            "temperature": request.temperature,
            "maxOutputTokens": request.max_tokens,
            "topP": request.top_p,
        }
        if self.thinking_budget is not None:
            generation["thinkingConfig"] = {"thinkingBudget": self.thinking_budget}

        payload: dict[str, Any] = {"contents": contents, "generationConfig": generation}
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        return payload

    async def generate(self, request: GenRequest) -> GenResponse:
        client, limiter = self._ensure()
        payload = self._payload(request)
        path = f"/models/{self.model}:generateContent"
        last_error = "unknown error"

        for attempt in range(self.max_retries + 1):
            await limiter.acquire()
            try:
                response = await client.post(path, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code == 200:
                    return _parse(response.json())
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if response.status_code == 429:
                    if _is_daily_quota(response):
                        # A per-day quota does not refill in this run. Retrying
                        # burns the retry budget and then returns an error
                        # anyway, which the coding layer would turn into an
                        # "other" label -- a fabricated observation that looks
                        # exactly like a real one in the output file.
                        return GenResponse(
                            text="",
                            finish_reason="error",
                            error=f"daily quota exhausted for {self.model}",
                        )
                    # Per-minute: the server states how long to wait, and
                    # guessing shorter just spends another attempt on a closed
                    # window.
                    await asyncio.sleep(_retry_delay(response) or 30.0)
                    continue
                if response.status_code < 500:
                    break  # bad key, bad model id, malformed request

            if attempt < self.max_retries:
                await asyncio.sleep(min(30.0, 2.0 * (attempt + 1)))

        return GenResponse(text="", finish_reason="error", error=last_error)

    async def health(self) -> bool:
        try:
            client, _ = self._ensure()
        except RuntimeError:
            return False
        try:
            response = await client.get("/models")
            return response.status_code == 200
        except (httpx.TimeoutException, httpx.TransportError):
            return False

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _is_daily_quota(response: httpx.Response) -> bool:
    """Whether the 429 is a per-day cap rather than a per-minute one.

    The distinction decides whether waiting helps at all. Free tier reports
    `GenerateRequestsPerDayPerProjectPerModel-FreeTier` when the daily
    allowance is gone, and no amount of backoff inside one run recovers it.
    """
    try:
        details = response.json().get("error", {}).get("details", [])
    except ValueError:
        return False
    for detail in details:
        for violation in detail.get("violations", []):
            if "PerDay" in str(violation.get("quotaId", "")):
                return True
    return False


def _retry_delay(response: httpx.Response) -> float | None:
    """Seconds the server asked us to wait, from RetryInfo or the message text."""
    try:
        details = response.json().get("error", {}).get("details", [])
    except ValueError:
        return None
    for detail in details:
        raw = detail.get("retryDelay")
        if isinstance(raw, str):
            match = re.match(r"([\d.]+)s", raw)
            if match:
                return float(match.group(1)) + 1.0
    return None


def _parse(body: dict[str, Any]) -> GenResponse:
    candidates = body.get("candidates") or []
    if not candidates:
        # A prompt blocked by safety filtering returns no candidate at all.
        reason = (body.get("promptFeedback") or {}).get("blockReason", "no candidates")
        return GenResponse(text="", finish_reason="error", error=str(reason))

    candidate = candidates[0]
    parts = (candidate.get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts)
    usage = body.get("usageMetadata") or {}
    return GenResponse(
        text=text,
        prompt_tokens=int(usage.get("promptTokenCount", 0) or 0),
        completion_tokens=int(usage.get("candidatesTokenCount", 0) or 0),
        finish_reason=candidate.get("finishReason") or "stop",
    )
