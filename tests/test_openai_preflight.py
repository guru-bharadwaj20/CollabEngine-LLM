"""Preflight and context-overflow handling for the served backend.

These cover the two failures that a served model can produce silently, both of
which have in-process ancestors that cost this project a corpus: a prompt that
does not fit the slot (RESEARCH-LOG 3.10) and a server quietly holding weights
other than the ones the config names.
"""

from __future__ import annotations

import asyncio
import itertools

import httpx

from collabengine.backends.base import ChatMessage, GenRequest
from collabengine.backends.openai_compat import (
    CONTEXT_OVERFLOW,
    OpenAICompatBackend,
    _model_matches,
    is_context_overflow,
)

MODEL = "Meta-Llama-3.1-8B-Instruct-Q4_K_M"

LLAMACPP_OVERFLOW = (
    '{"error":{"code":400,"message":"the request exceeds the available '
    "context size. try increasing the context size or enable context "
    'shift","type":"server_error"}}'
)


def _backend(handler, base_url: str = "http://stub/v1", **kwargs) -> OpenAICompatBackend:
    backend = OpenAICompatBackend(
        model=MODEL, max_retries=2, base_url=base_url, **kwargs
    )
    backend._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=base_url
    )
    backend._sem = asyncio.Semaphore(4)
    return backend


def _server(
    *,
    model: str = MODEL,
    ctx_per_slot: int | None = 16384,
    slots: int = 4,
    models_status: int = 200,
):
    """A stub llama-server. `ctx_per_slot=None` serves no /props, like vLLM.

    Paths are matched exactly, and that is the whole point of this stub rather
    than a looser one. It previously matched `endswith("/props")`, which answers
    `/v1/props` as readily as `/props` -- so the suite passed for every build
    while the real check was requesting the wrong URL against a real server and
    reading the 404 as "this server has no /props" (RESEARCH-LOG 3.12). A stub
    laxer than the server it stands in for cannot fail on the difference.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        # Measured against llama-server b10369: `models` answers at both paths,
        # `props` only at the root. The asymmetry is the bug's whole habitat.
        if request.url.path in ("/v1/models", "/models"):
            return httpx.Response(
                models_status, json={"data": [{"id": model, "object": "model"}]}
            )
        if request.url.path == "/props":
            if ctx_per_slot is None:
                return httpx.Response(404, text="not found")
            return httpx.Response(
                200,
                json={
                    "default_generation_settings": {"n_ctx": ctx_per_slot},
                    "total_slots": slots,
                },
            )
        return httpx.Response(404, text="unexpected path")

    return handler


def _request() -> GenRequest:
    return GenRequest(messages=[ChatMessage("user", "hi")], max_tokens=32)


# ------------------------------------------------------------- preflight ----


def test_preflight_passes_when_the_server_matches_the_config() -> None:
    report = asyncio.run(_backend(_server()).preflight(required_ctx=16384))

    assert report.ok
    assert report.served_models == (MODEL,)
    assert report.ctx_per_slot == 16384
    assert report.total_slots == 4


def test_preflight_fails_when_the_slot_is_too_small() -> None:
    """The --parallel trap: -c is divided across slots, so 4 slots of 65536
    total leave 16384 each, and raising --parallel silently shrinks them."""
    report = asyncio.run(_backend(_server(ctx_per_slot=2730)).preflight(16384))

    assert not report.ok
    problem = " ".join(report.problems)
    assert "2730" in problem and "16384" in problem
    # The remedy names the total to pass to -c, not the per-slot figure, because
    # the per-slot figure is not a flag anyone can set.
    assert str(16384 * 4) in problem


def test_preflight_fails_when_the_server_holds_different_weights() -> None:
    """Identical weights across agents is the control the design rests on."""
    report = asyncio.run(_backend(_server(model="Qwen3-8B")).preflight(16384))

    assert not report.ok
    assert "Qwen3-8B" in " ".join(report.problems)


def test_model_check_can_be_disabled_for_an_unnamed_server() -> None:
    backend = _backend(_server(model="local-model"), verify_model=False)
    assert asyncio.run(backend.preflight(16384)).ok


def test_a_server_without_props_is_unverified_not_failed() -> None:
    """vLLM has no /props. Refusing to run against it would be a regression."""
    report = asyncio.run(_backend(_server(ctx_per_slot=None)).preflight(16384))

    assert report.ok
    assert report.ctx_per_slot is None


def test_props_is_requested_at_the_root_not_under_the_v1_prefix() -> None:
    """The slot check is only as good as the URL it asks.

    llama-server serves `/props` at the root and 404s `/v1/props`. Resolving it
    relative to a `/v1` base_url therefore produced a 404 that the check read as
    "no /props on this server" -- so the --parallel trap went unguarded against
    every real llama-server, while the suite stayed green (RESEARCH-LOG 3.12).
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return _server()(request)

    report = asyncio.run(_backend(handler).preflight(required_ctx=16384))

    assert "/props" in seen and "/v1/props" not in seen
    assert report.ctx_per_slot == 16384, "slot size must come back verified"


def test_a_base_url_without_a_v1_prefix_still_finds_props() -> None:
    report = asyncio.run(
        _backend(_server(), base_url="http://stub").preflight(required_ctx=16384)
    )

    assert report.ctx_per_slot == 16384


def test_preflight_reports_an_unreachable_server_rather_than_raising() -> None:
    def dead(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    report = asyncio.run(_backend(dead).preflight(16384))

    assert not report.ok and not report.reachable
    assert "cannot reach" in " ".join(report.problems)


def test_preflight_without_a_required_ctx_checks_only_identity() -> None:
    report = asyncio.run(_backend(_server(ctx_per_slot=1024)).preflight())
    assert report.ok, "no requirement was stated, so no size can be wrong"


def test_report_lines_name_the_shortfall() -> None:
    report = asyncio.run(_backend(_server(ctx_per_slot=2730)).preflight(16384))
    text = "\n".join(report.lines())
    assert "ctx per slot" in text and "PROBLEM" in text


# ------------------------------------------------------ context overflow ----


def test_context_overflow_is_labelled_and_not_retried() -> None:
    """Retrying is pure waste: the next attempt sends the same token count."""
    calls = itertools.count()

    def handler(request: httpx.Request) -> httpx.Response:
        next(calls)
        return httpx.Response(400, text=LLAMACPP_OVERFLOW)

    out = asyncio.run(_backend(handler).generate(_request()))

    assert not out.ok
    assert (out.error or "").startswith(CONTEXT_OVERFLOW)
    assert out.finish_reason == "error"
    assert next(calls) == 1


def test_overflow_reported_as_a_500_is_still_not_retried() -> None:
    """Checked by body before status class, because llama.cpp has used both."""
    calls = itertools.count()

    def handler(request: httpx.Request) -> httpx.Response:
        next(calls)
        return httpx.Response(500, text=LLAMACPP_OVERFLOW)

    out = asyncio.run(_backend(handler).generate(_request()))

    assert (out.error or "").startswith(CONTEXT_OVERFLOW)
    assert next(calls) == 1, "a 500 that is really an overflow must not retry"


def test_a_genuine_server_hiccup_still_retries() -> None:
    """The overflow check must not swallow the transient errors it sits next to."""
    counter = itertools.count()

    def handler(request: httpx.Request) -> httpx.Response:
        return (
            httpx.Response(503, text="loading model")
            if next(counter) < 1
            else httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1},
                },
            )
        )

    assert asyncio.run(_backend(handler).generate(_request())).text == "ok"


def test_overflow_markers_cover_both_servers() -> None:
    assert is_context_overflow(LLAMACPP_OVERFLOW)
    assert is_context_overflow(
        "This model's maximum context length is 8192 tokens, however you "
        "requested 15400"
    )
    assert not is_context_overflow("CUDA out of memory")
    assert not is_context_overflow("connection reset by peer")


# ------------------------------------------------------- model identity ----


def test_model_match_tolerates_the_forms_a_gguf_server_reports() -> None:
    served = (
        "models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "C:\\models\\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "meta-llama-3.1-8b-instruct",
    )
    for name in served:
        assert _model_matches(MODEL, (name,)), name


def test_model_match_still_catches_a_different_model() -> None:
    assert not _model_matches(MODEL, ("Qwen3-8B",))
    assert not _model_matches(MODEL, ("Meta-Llama-3.1-70B-Instruct-Q4_K_M",))
