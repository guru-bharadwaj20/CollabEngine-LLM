"""Backend interface.

Generation is async and batch-first. The whole experiment is throughput-bound
rather than latency-bound -- roughly 14M output tokens across the condition grid
-- so the interface is shaped to keep a batching server saturated rather than to
minimize any single call's round trip.
"""

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """A chat-template turn. Distinct from `protocol.Message`.

    `protocol.Message` is the experiment's record of who said what;
    `ChatMessage` is what actually goes over the wire to the model. The mapping
    between them is the orchestrator's job, and it differs per agent because each
    agent sees its own contributions as `assistant` and everyone else's as `user`.
    """

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class GenRequest:
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int | None = None
    stop: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    """Out-of-band context. Real backends ignore this; MockBackend uses it to
    simulate task-aware behavior without parsing the rendered prompt back."""


@dataclass(slots=True)
class GenResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = "stop"
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class LLMBackend(abc.ABC):
    """Abstract model server.

    Implementations must be safe to share across concurrent episodes; the runner
    fans many episodes at one backend instance to keep the server busy.
    """

    name: str = "backend"

    honors_request_seed: bool = True
    """Whether `GenRequest.seed` actually controls that request's sampling.

    A batching engine that seeds per sequence (vLLM) honors it; one that draws
    the whole batch from a single global RNG does not. This is not a performance
    detail -- `SymmetryBreaking.NAME_ONLY` and `NAME_SEED` differ *only* by
    per-agent seed, so on a backend that ignores seeds they are the same
    condition, and running both spends the card to produce a duplicate.
    """

    @abc.abstractmethod
    async def generate(self, request: GenRequest) -> GenResponse:
        """Complete a single request."""

    async def generate_batch(self, requests: list[GenRequest]) -> list[GenResponse]:
        """Complete many requests concurrently.

        The default gathers `generate` calls, which is correct for any backend
        whose own client multiplexes. Override only when the server exposes a
        genuine batch endpoint that beats concurrent singles.
        """
        if not requests:
            return []
        return list(await asyncio.gather(*(self.generate(r) for r in requests)))

    async def health(self) -> bool:
        """Whether the backend is reachable. Checked once before a run starts."""
        return True

    async def aclose(self) -> None:
        """Release connections. Idempotent."""

    async def __aenter__(self) -> LLMBackend:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
