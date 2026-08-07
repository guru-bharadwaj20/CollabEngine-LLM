"""OpenAI-compatible backend and the parallel runner.

The backend cannot be tested against real hardware here, so it is driven against
an httpx stub transport. That covers wire format, retry policy, and failure
degradation -- everything except throughput.
"""

from __future__ import annotations

import asyncio
import itertools

import httpx
import pytest

from collabengine.backends.base import ChatMessage, GenRequest
from collabengine.backends.openai_compat import OpenAICompatBackend, _backoff, _parse
from collabengine.runner import RunPlan, completed_episode_ids, run_plan
from collabengine.transcripts.store import EpisodeRecord, TranscriptReader
from collabengine.tasks.grader import GradeResult
from collabengine.tasks.schema import ALL_COMPONENTS, Solution


def _request() -> GenRequest:
    return GenRequest(
        messages=[ChatMessage("system", "sys"), ChatMessage("user", "hi")],
        max_tokens=32,
        seed=7,
    )


def _ok_body(text: str = "hello") -> dict:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 3},
    }


def _backend_with(handler) -> OpenAICompatBackend:
    backend = OpenAICompatBackend(max_retries=2)
    backend._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://stub/v1"
    )
    backend._sem = asyncio.Semaphore(4)
    return backend


def test_successful_generation_parses_text_and_usage() -> None:
    backend = _backend_with(lambda req: httpx.Response(200, json=_ok_body("answer")))
    out = asyncio.run(backend.generate(_request()))

    assert out.ok
    assert out.text == "answer"
    assert out.prompt_tokens == 11
    assert out.completion_tokens == 3


def test_seed_and_sampling_are_forwarded() -> None:
    """Reproducibility depends on the seed actually reaching the server."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_ok_body())

    asyncio.run(_backend_with(handler).generate(_request()))
    assert seen["seed"] == 7
    assert seen["max_tokens"] == 32
    assert "temperature" in seen and "top_p" in seen


def test_retries_then_succeeds_on_transient_error() -> None:
    counter = itertools.count()

    def handler(request: httpx.Request) -> httpx.Response:
        return (
            httpx.Response(503, text="overloaded")
            if next(counter) < 2
            else httpx.Response(200, json=_ok_body("recovered"))
        )

    backend = _backend_with(handler)
    backend.max_retries = 3
    out = asyncio.run(backend.generate(_request()))
    assert out.ok and out.text == "recovered"


def test_non_retryable_status_fails_fast_without_raising() -> None:
    calls = itertools.count()

    def handler(request: httpx.Request) -> httpx.Response:
        next(calls)
        return httpx.Response(400, text="bad request")

    out = asyncio.run(_backend_with(handler).generate(_request()))
    assert not out.ok
    assert "400" in (out.error or "")
    assert next(calls) == 1, "a 400 must not be retried"


def test_exhausted_retries_degrade_the_turn_rather_than_raising() -> None:
    """One bad turn must not abort an episode -- or a batch of hundreds."""
    backend = _backend_with(lambda req: httpx.Response(500, text="boom"))
    out = asyncio.run(backend.generate(_request()))
    assert not out.ok
    assert out.text == ""
    assert out.finish_reason == "error"


def test_empty_choices_is_an_error_not_a_crash() -> None:
    assert not _parse({"choices": []}).ok


def test_backoff_is_bounded_and_jittered() -> None:
    values = [_backoff(a) for a in range(6) for _ in range(20)]
    assert all(0 < v <= 20.0 for v in values)
    assert len(set(values)) > 1, "un-jittered retries would resynchronize"


# ---------------------------------------------------------------- runner ----


def _record(episode_id: str) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=episode_id,
        condition="baseline",
        instance_seed=0,
        difficulty="tiny",
        agents=["A1"],
        messages=[],
        solution=Solution(),
        grade=GradeResult(
            per_component={c: 1.0 for c in ALL_COMPONENTS}, overall=1.0, satisfied={}
        ),
    )


def test_runner_writes_every_episode(tmp_path) -> None:
    out = tmp_path / "shard.jsonl"

    async def factory(i: int) -> EpisodeRecord:
        await asyncio.sleep(0)
        return _record(f"ep-{i}")

    plans = [RunPlan(f"ep-{i}", lambda i=i: factory(i)) for i in range(20)]
    stats = asyncio.run(run_plan(plans, out_path=out, max_concurrency=8))

    assert stats.completed == 20
    assert len(TranscriptReader(out).load()) == 20


def test_runner_resumes_and_skips_finished_work(tmp_path) -> None:
    """A dropped connection at hour five must not cost hours one to four."""
    out = tmp_path / "shard.jsonl"

    async def factory(i: int) -> EpisodeRecord:
        return _record(f"ep-{i}")

    first = [RunPlan(f"ep-{i}", lambda i=i: factory(i)) for i in range(5)]
    asyncio.run(run_plan(first, out_path=out, max_concurrency=4))

    second = [RunPlan(f"ep-{i}", lambda i=i: factory(i)) for i in range(8)]
    stats = asyncio.run(run_plan(second, out_path=out, max_concurrency=4, resume=True))

    assert stats.skipped == 5
    assert stats.completed == 3
    assert len(completed_episode_ids(out)) == 8


def test_one_failing_episode_does_not_abort_the_batch(tmp_path) -> None:
    out = tmp_path / "shard.jsonl"

    async def good(i: int) -> EpisodeRecord:
        return _record(f"ep-{i}")

    async def bad() -> EpisodeRecord:
        raise RuntimeError("model server exploded")

    plans = [RunPlan(f"ep-{i}", lambda i=i: good(i)) for i in range(6)]
    plans.insert(3, RunPlan("ep-bad", bad))

    stats = asyncio.run(run_plan(plans, out_path=out, max_concurrency=3))
    assert stats.completed == 6
    assert stats.failed == 1
    # The failed id is absent, so a resume retries exactly it.
    assert "ep-bad" not in completed_episode_ids(out)


def test_runner_respects_concurrency_ceiling(tmp_path) -> None:
    out = tmp_path / "shard.jsonl"
    live = 0
    peak = 0

    async def factory(i: int) -> EpisodeRecord:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return _record(f"ep-{i}")

    plans = [RunPlan(f"ep-{i}", lambda i=i: factory(i)) for i in range(30)]
    asyncio.run(run_plan(plans, out_path=out, max_concurrency=5))
    assert peak <= 5, f"concurrency ceiling breached: peak={peak}"


def test_reader_tolerates_a_truncated_final_line(tmp_path) -> None:
    """The expected artifact of a killed run. Skipping beats refusing to read."""
    out = tmp_path / "shard.jsonl"
    asyncio.run(
        run_plan(
            [RunPlan("ep-0", lambda: _wrap(_record("ep-0")))],
            out_path=out,
            max_concurrency=1,
        )
    )
    with out.open("a", encoding="utf-8") as fh:
        fh.write('{"episode_id": "ep-1", "condition": ')

    reader = TranscriptReader(out)
    assert len(reader.load()) == 1
    assert reader.skipped == 1

    with pytest.raises(ValueError):
        TranscriptReader(out, strict=True).load()


async def _wrap(record: EpisodeRecord) -> EpisodeRecord:
    return record
