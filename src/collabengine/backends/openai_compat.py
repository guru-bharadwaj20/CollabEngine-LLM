"""OpenAI-compatible backend, for vLLM or a llama.cpp server.

Untested against real hardware at time of writing -- the development machine has
no CUDA device. It is exercised end to end against a stub server in
`tests/test_openai_backend.py`, so the wire format, retry behavior, and error
handling are covered; only throughput characteristics are unknown.

Design notes for the real run:

* A failed turn returns a GenResponse carrying `error` rather than raising. One
  bad turn in a twelve-turn episode should degrade that episode, not abort a
  batch of hundreds that has been running for an hour.
* Concurrency is bounded here rather than at the runner, because the limit is a
  property of the server (its max_num_seqs) rather than of the experiment.
* `seed` is forwarded. vLLM honors it per request, which is what makes an
  episode reproducible from config alone.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

import httpx

from collabengine.backends.base import GenRequest, GenResponse, LLMBackend

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


@dataclass
class OpenAICompatBackend(LLMBackend):
    """Chat-completions client for a locally served model."""

    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen3-8B"
    api_key: str | None = None
    max_concurrency: int = 64
    """Cap on in-flight requests.

    Set at or slightly above the server's --max-num-seqs. Higher just queues in
    the client where you cannot see it; lower leaves the GPU idle."""
    timeout_s: float = 600.0
    max_retries: int = 4
    name: str = "openai_compat"

    _client: httpx.AsyncClient | None = field(default=None, init=False, repr=False)
    _sem: asyncio.Semaphore | None = field(default=None, init=False, repr=False)

    def _ensure(self) -> tuple[httpx.AsyncClient, asyncio.Semaphore]:
        # Built lazily so the backend can be constructed outside a running loop
        # (config parsing, CLI wiring) without binding a semaphore to the wrong one.
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url.rstrip("/"),
                headers=headers,
                timeout=httpx.Timeout(self.timeout_s),
                limits=httpx.Limits(
                    max_connections=self.max_concurrency + 8,
                    max_keepalive_connections=self.max_concurrency,
                ),
            )
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.max_concurrency)
        return self._client, self._sem

    def _payload(self, request: GenRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in request.messages],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.stop:
            payload["stop"] = request.stop
        return payload

    async def generate(self, request: GenRequest) -> GenResponse:
        client, sem = self._ensure()
        payload = self._payload(request)
        last_error = "unknown error"

        async with sem:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post("/chat/completions", json=payload)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                else:
                    if response.status_code == 200:
                        return _parse(response.json())
                    last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                    if response.status_code not in RETRYABLE_STATUS:
                        break

                if attempt < self.max_retries:
                    await asyncio.sleep(_backoff(attempt))

        # Degrade the turn, do not kill the episode.
        return GenResponse(text="", finish_reason="error", error=last_error)

    async def health(self) -> bool:
        client, _ = self._ensure()
        try:
            response = await client.get("/models")
            return response.status_code == 200
        except (httpx.TimeoutException, httpx.TransportError):
            return False

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _parse(body: dict[str, Any]) -> GenResponse:
    choices = body.get("choices") or []
    if not choices:
        return GenResponse(text="", finish_reason="error", error="no choices returned")

    message = choices[0].get("message") or {}
    usage = body.get("usage") or {}
    return GenResponse(
        text=message.get("content") or "",
        prompt_tokens=int(usage.get("prompt_tokens", 0)),
        completion_tokens=int(usage.get("completion_tokens", 0)),
        finish_reason=choices[0].get("finish_reason") or "stop",
    )


def _backoff(attempt: int, *, base: float = 0.5, cap: float = 20.0) -> float:
    """Exponential backoff with jitter.

    Jitter matters here specifically: dozens of episodes hit the same server, so
    un-jittered retries would resynchronize into a thundering herd after any
    server hiccup.
    """
    return min(cap, base * (2**attempt)) * (0.5 + random.random() / 2)
