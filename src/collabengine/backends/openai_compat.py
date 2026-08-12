"""OpenAI-compatible backend, for vLLM or a llama.cpp server.

Exercised end to end against an httpx stub in `tests/test_backend_and_runner.py`
and `tests/test_openai_preflight.py`, so the wire format, retry behavior, and
error handling are covered; only throughput characteristics are unknown until it
meets a card.

Design notes for the real run:

* A failed turn returns a GenResponse carrying `error` rather than raising. One
  bad turn in a twelve-turn episode should degrade that episode, not abort a
  batch of hundreds that has been running for an hour.
* Concurrency is bounded here rather than at the runner, because the limit is a
  property of the server (its --max-num-seqs or --parallel) rather than of the
  experiment.
* `seed` is forwarded. vLLM honors it per request, which is what makes an
  episode reproducible from config alone. llama.cpp honors it per slot; see
  docs/LLAMACPP-SETUP.md for what continuous batching does to that guarantee.

Two failures specific to a served model are handled here rather than left to the
analysis, because both have already cost this project a corpus in their
in-process form (RESEARCH-LOG 3.10, 3.11):

* **Context overflow.** A prompt longer than the slot's context is a 400 with a
  server-specific message. It is not retried -- the next attempt is the same
  length -- and it is labelled `context_overflow:` so it cannot be read as a
  transient server hiccup. This is the served analogue of the OOM that killed
  the first `xhard` corpus, and unlike an OOM it is deterministic, so
  regenerating the episode cannot fix it. Only a larger slot can.
* **Serving the wrong weights.** The identical-weights control is what rules out
  model heterogeneity as the source of any differentiation observed, and it is
  enforced by nothing at all if the config's model id is never compared against
  what the server actually loaded. `preflight` compares them.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from typing import Any

import httpx

from collabengine.backends.base import GenRequest, GenResponse, LLMBackend

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})

CONTEXT_OVERFLOW_MARKERS = (
    # llama.cpp server
    "exceeds the available context size",
    "context shift is disabled",
    "context size exceeded",
    # vLLM
    "maximum context length",
    "longer than the maximum",
    # generic
    "context_length_exceeded",
)
"""Substrings that identify a prompt-too-long rejection.

Phrases rather than tokens, and that is the whole design of this list. A bare
`n_ctx` would match it too -- and would also match half of llama.cpp's routine
diagnostics, turning a transient error into one that is never retried. The
asymmetry is small either way (an unretried turn is regenerated under the
preregistration, not dropped), so the list is drawn where it can be read: each
entry is a sentence a server writes only when a prompt did not fit."""

CONTEXT_OVERFLOW = "context_overflow"
"""Prefix on `GenResponse.error` when the prompt did not fit the slot.

The integrity audit already treats any errored turn as an instrument failure, so
this changes no analysis. It exists so the operator reading the error tally can
tell "the slot is too small" -- which regenerating will never fix -- from "the
server hiccuped", which regenerating does fix."""


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """What the server says about itself, checked against what the run needs.

    Cheap enough to run before every stage. Its whole purpose is to convert an
    eight-hour night that produces a corpus of errored turns into a refusal that
    takes two seconds, which is a trade this project has already paid for three
    times in the other direction (RESEARCH-LOG 4.1d)."""

    reachable: bool
    served_models: tuple[str, ...] = ()
    ctx_per_slot: int | None = None
    """Context window one request may use, as reported by llama.cpp `/props`.

    Recent llama.cpp divides `-c` across `--parallel` slots and reports the
    per-slot figure here; a request is bounded by that, not by the total. Older
    builds and vLLM do not serve `/props` at all, in which case this is None and
    the check is skipped rather than guessed at."""
    total_slots: int | None = None
    required_ctx: int | None = None
    problems: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.reachable and not self.problems

    def lines(self) -> list[str]:
        rows = [f"reachable        {self.reachable}"]
        if self.served_models:
            rows.append(f"served model     {', '.join(self.served_models)}")
        if self.ctx_per_slot is not None:
            need = "" if self.required_ctx is None else f" (need {self.required_ctx})"
            rows.append(f"ctx per slot     {self.ctx_per_slot}{need}")
        if self.total_slots is not None:
            rows.append(f"slots            {self.total_slots}")
        rows.extend(f"PROBLEM          {p}" for p in self.problems)
        return rows


@dataclass
class OpenAICompatBackend(LLMBackend):
    """Chat-completions client for a locally served model."""

    base_url: str = "http://localhost:8000/v1"
    model: str = "Qwen/Qwen3-8B"
    api_key: str | None = None
    max_concurrency: int = 64
    """Cap on in-flight requests.

    Set at or slightly above the server's --max-num-seqs. Higher just queues in
    the client where you cannot see it; lower leaves the GPU idle.

    Under llama.cpp the ceiling is `--parallel`, and exceeding it is worse than
    merely queueing: the context is divided across slots, so a run that raises
    concurrency without raising `-c` shrinks every slot's window until long
    prompts stop fitting."""
    timeout_s: float = 600.0
    max_retries: int = 4
    verify_model: bool = True
    """Whether `preflight` fails when the served model id is not `model`.

    On by default. The served id is a free, exact check on the one control that
    the whole design rests on -- every agent drawing from identical weights --
    and it is the kind of mismatch that produces a perfectly plausible corpus."""
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
                    if is_context_overflow(response.text):
                        # Deterministic in the prompt, so every retry fails
                        # identically and the backoff only wastes wall clock.
                        # Checked before the status class because llama.cpp has
                        # reported this as a 500 as well as a 400.
                        last_error = f"{CONTEXT_OVERFLOW}: {last_error}"
                        break
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

    async def preflight(self, required_ctx: int | None = None) -> PreflightReport:
        """Check the server is the one this config describes, before spending a night.

        `required_ctx` is the longest single request the run will make, in
        tokens: the worst-case rendered context plus `max_tokens`. Pass it and
        the slot size is checked; omit it and only reachability and model
        identity are.

        Every check degrades to "skipped" rather than "failed" when the server
        does not serve the endpoint, because vLLM has no `/props` and refusing
        to run against vLLM would be a regression.
        """
        client, _ = self._ensure()
        problems: list[str] = []

        try:
            models_response = await client.get("/models")
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            return PreflightReport(
                reachable=False,
                required_ctx=required_ctx,
                problems=(
                    f"cannot reach {self.base_url}: {type(exc).__name__}: {exc}",
                ),
            )
        if models_response.status_code != 200:
            return PreflightReport(
                reachable=False,
                required_ctx=required_ctx,
                problems=(f"GET /models returned HTTP {models_response.status_code}",),
            )

        served = _served_model_ids(models_response.json())
        if self.verify_model and served and not _model_matches(self.model, served):
            problems.append(
                f"config asks for model {self.model!r} but the server is serving "
                f"{', '.join(served)!r}; the identical-weights control is only "
                "as good as this line"
            )

        ctx_per_slot, total_slots = await self._props()
        if required_ctx is not None and ctx_per_slot is not None:
            if ctx_per_slot < required_ctx:
                problems.append(
                    f"slot context is {ctx_per_slot} tokens but the worst-case "
                    f"request needs {required_ctx}; raise -c to at least "
                    f"{required_ctx * (total_slots or 1)} "
                    f"(-c is divided across --parallel slots) or lower --parallel"
                )

        return PreflightReport(
            reachable=True,
            served_models=served,
            ctx_per_slot=ctx_per_slot,
            total_slots=total_slots,
            required_ctx=required_ctx,
            problems=tuple(problems),
        )

    async def _props(self) -> tuple[int | None, int | None]:
        """llama.cpp `/props`. Absent on vLLM, so absence is not a failure.

        `/props` is served at the server **root**, not under the OpenAI-compatible
        `/v1` prefix, so it is requested by absolute URL. Asking the client for
        the relative `/props` resolves it against `base_url` and hits
        `/v1/props`, which llama-server answers 404 -- indistinguishable here
        from a server that has no `/props` at all. That is what happened: this
        check silently degraded to "unverified" against a healthy llama-server
        for every run before 2026-08-12, and said so while blaming the server
        (RESEARCH-LOG 3.12). A guard that explains its own silence with a
        plausible wrong reason is worse than one that is merely absent.
        """
        client, _ = self._ensure()
        try:
            response = await client.get(_root_url(self.base_url) + "/props")
        except (httpx.TimeoutException, httpx.TransportError):
            return None, None
        if response.status_code != 200:
            return None, None
        try:
            body = response.json()
        except ValueError:
            return None, None
        if not isinstance(body, dict):
            return None, None

        settings = body.get("default_generation_settings")
        n_ctx = settings.get("n_ctx") if isinstance(settings, dict) else None
        slots = body.get("total_slots")
        return (
            int(n_ctx) if isinstance(n_ctx, (int, float)) else None,
            int(slots) if isinstance(slots, (int, float)) else None,
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _root_url(base_url: str) -> str:
    """Server root for a base URL that may carry an OpenAI-compatible prefix.

    llama-server serves `/props`, `/health` and `/tokenize` at the root while
    the chat endpoints live under `/v1`. Only a trailing `/v1` is stripped: a
    server mounted at `/llama/v1` keeps its `/llama`.
    """
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    return root


def is_context_overflow(body_text: str) -> bool:
    """Whether a server's rejection means the prompt did not fit the slot."""
    lowered = body_text.lower()
    return any(marker in lowered for marker in CONTEXT_OVERFLOW_MARKERS)


def _served_model_ids(body: Any) -> tuple[str, ...]:
    if not isinstance(body, dict):
        return ()
    data = body.get("data")
    if not isinstance(data, list):
        return ()
    return tuple(
        str(entry["id"]) for entry in data if isinstance(entry, dict) and "id" in entry
    )


def _model_matches(configured: str, served: tuple[str, ...]) -> bool:
    """Whether the configured id names one of the served ones.

    Loose on both sides on purpose. llama.cpp reports the GGUF's file stem or
    path while the config carries a readable name, so an exact comparison would
    fail on every correct setup. Matching on the basename without extension,
    case-folded, catches the failure that matters -- a different model entirely
    -- without crying wolf over `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` versus
    the path it was loaded from.
    """
    want = _model_key(configured)
    return any(
        want == _model_key(s) or want in _model_key(s) or _model_key(s) in want
        for s in served
    )


def _model_key(model_id: str) -> str:
    stem = model_id.replace("\\", "/").rsplit("/", 1)[-1]
    if stem.lower().endswith(".gguf"):
        stem = stem[: -len(".gguf")]
    return stem.strip().lower()


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
