"""Frontier-model judge backend, for behavioral coding only.

docs/PLAN.md 4, Phase 2: *"Use a frontier API model as the judge -- a local 7-8B is
too weak to code reliably. Two judges from different families, blind to
condition."* This backend is the first of those two judges.

It deliberately implements the same `LLMBackend` interface as the local and mock
backends even though it never runs an episode. Coding is a generation task with
a different cost profile and a different failure mode, and reusing the interface
means `analysis.coding` does not care where its labels come from -- the same
`code_episode` call runs against the mock in tests, the local 8B in a smoke run,
and Claude in the real one.

**The judge is not the subject.** Nothing here touches the agents under study.
Using a frontier model to *code* transcripts produced by a local 8B introduces
no confound, because the judge never participates in an episode; it reads
finished text. Using one to *produce* those transcripts would destroy the
one-model control that the whole design rests on.

Cost, for sizing: ~14k messages at a few hundred input tokens each is roughly
$5 on Claude Haiku 4.5 ($1/$5 per MTok in/out) with the 8-token replies this
prompt asks for. The Batch API would halve that for a corpus this size, at the
cost of a polling loop -- worth doing if the corpus grows past one run.
"""

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass, field
from typing import Any

from collabengine.backends.base import GenRequest, GenResponse, LLMBackend

DEFAULT_JUDGE_MODEL = "claude-haiku-4-5"
"""Cheapest current model that codes reliably at this task's difficulty.

The taxonomy is eight labels over one short message -- classification, not
reasoning. Spending Opus rates on 14k such calls buys accuracy this task does
not need. Swap to a second family for the inter-judge kappa run."""


@dataclass
class AnthropicJudgeBackend(LLMBackend):
    """Async Claude client shaped as an `LLMBackend`."""

    model: str = DEFAULT_JUDGE_MODEL
    api_key: str | None = None
    max_concurrency: int = 8
    max_retries: int = 4
    timeout_s: float = 120.0
    send_temperature: bool = True
    """Whether to forward `temperature`.

    Haiku 4.5 accepts it, and temperature 0 is what makes a *classification*
    reproducible rather than sampled. Newer frontier models reject sampling
    parameters outright, so set this False when pointing the judge at one --
    the coding call already asks for a single label, which is most of what
    temperature 0 was buying."""
    name: str = "anthropic_judge"

    _client: Any = field(default=None, init=False, repr=False)
    _sem: Any = field(default=None, init=False, repr=False)
    _loop: Any = field(default=None, init=False, repr=False)

    def _ensure(self):
        import anthropic

        if self._client is None:
            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError(
                    "no ANTHROPIC_API_KEY; set it or pass api_key= to run the "
                    "frontier judge (the local judge needs neither)"
                )
            self._client = anthropic.AsyncAnthropic(api_key=key, timeout=self.timeout_s)

        loop = asyncio.get_running_loop()
        if self._sem is None or self._loop is not loop:
            self._loop = loop
            self._sem = asyncio.Semaphore(self.max_concurrency)
        return self._client, self._sem

    async def generate(self, request: GenRequest) -> GenResponse:
        import anthropic

        client, sem = self._ensure()
        system, messages = _split_system(request)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if self.send_temperature:
            kwargs["temperature"] = request.temperature

        last_error = "unknown error"
        async with sem:
            for attempt in range(self.max_retries + 1):
                try:
                    message = await client.messages.create(**kwargs)
                except anthropic.RateLimitError as exc:
                    last_error = f"RateLimitError: {exc}"
                except anthropic.APIConnectionError as exc:
                    last_error = f"APIConnectionError: {exc}"
                except anthropic.APIStatusError as exc:
                    last_error = f"HTTP {exc.status_code}: {exc}"
                    if exc.status_code < 500:
                        # 400/401/404 will not fix themselves; a bad model id or
                        # a revoked key should surface now, not after 14k retries.
                        break
                else:
                    return _parse(message)

                if attempt < self.max_retries:
                    await asyncio.sleep(_backoff(attempt))

        return GenResponse(text="", finish_reason="error", error=last_error)

    async def health(self) -> bool:
        try:
            self._ensure()
        except (RuntimeError, ImportError):
            return False
        return True

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None


def _split_system(request: GenRequest) -> tuple[str, list[dict[str, str]]]:
    """Claude takes the system prompt as its own parameter, not a message."""
    system_parts = [m.content for m in request.messages if m.role == "system"]
    messages = [m.to_dict() for m in request.messages if m.role != "system"]
    if not messages:
        # The API rejects an empty turn list; a judge call always has content.
        messages = [{"role": "user", "content": ""}]
    return "\n\n".join(system_parts), messages


def _parse(message: Any) -> GenResponse:
    text = "".join(b.text for b in message.content if getattr(b, "type", "") == "text")
    usage = getattr(message, "usage", None)
    return GenResponse(
        text=text,
        prompt_tokens=getattr(usage, "input_tokens", 0) or 0,
        completion_tokens=getattr(usage, "output_tokens", 0) or 0,
        finish_reason=getattr(message, "stop_reason", None) or "stop",
    )


def _backoff(attempt: int, *, base: float = 1.0, cap: float = 30.0) -> float:
    """Jittered exponential backoff.

    Jitter matters at this concurrency: a rate-limit response hits every
    in-flight coding call at once, and un-jittered retries would resynchronize
    into the same wall a second later."""
    return min(cap, base * (2**attempt)) * (0.5 + random.random() / 2)
