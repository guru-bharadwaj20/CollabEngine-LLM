"""The local CUDA backend's queueing logic.

The parts worth testing here are the ones that do not need a GPU: whether
concurrent turns actually coalesce into one forward pass (the entire reason this
backend exists), whether a failing batch degrades into error responses instead of
taking the run down, and whether mixed sampling parameters stay separated.

Generation itself is not mocked-and-asserted -- a fake `model.generate` would only
test the fake. That path is covered by running the real thing; see
`scripts/` usage in the README.
"""

from __future__ import annotations

import asyncio

import pytest

from collabengine.backends.base import ChatMessage, GenRequest, GenResponse
from collabengine.backends.hf_local import (
    HFLocalBackend,
    _batch_seed,
    _group_by_sampling,
    _sorted_chunks,
    _split_by_token_budget,
    _Waiter,
)


class FakeHF(HFLocalBackend):
    """Records batch sizes instead of touching CUDA."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.batches: list[int] = []
        self.fail = False

    def _load_sync(self) -> None:
        self._model = object()
        self._tokenizer = object()

    def _generate_batch(self, batch):
        self.batches.append(len(batch))
        if self.fail:
            raise RuntimeError("cuda melted")
        return [
            GenResponse(text=f"reply-{w.request.seed}", completion_tokens=3)
            for w in batch
        ]


def _req(seed: int, **kw) -> GenRequest:
    return GenRequest(
        messages=[ChatMessage(role="user", content="hi")], seed=seed, **kw
    )


async def test_concurrent_turns_coalesce_into_one_forward_pass() -> None:
    """The point of the backend: 8 waiting turns should ride one pass, not 8."""
    backend = FakeHF(max_batch_size=8, batch_window_s=0.05)
    results = await asyncio.gather(*(backend.generate(_req(i)) for i in range(8)))

    assert [r.text for r in results] == [f"reply-{i}" for i in range(8)]
    assert backend.batches == [8]


async def test_batch_size_is_respected() -> None:
    backend = FakeHF(max_batch_size=3, batch_window_s=0.05)
    await asyncio.gather(*(backend.generate(_req(i)) for i in range(7)))
    assert max(backend.batches) <= 3
    assert sum(backend.batches) == 7


async def test_a_failing_batch_degrades_every_turn_in_it() -> None:
    """One bad batch must not abort a grid that has been running for hours."""
    backend = FakeHF(max_batch_size=4)
    backend.fail = True

    results = await asyncio.gather(*(backend.generate(_req(i)) for i in range(4)))

    assert all(not r.ok for r in results)
    assert all("cuda melted" in (r.error or "") for r in results)
    assert all(r.finish_reason == "error" for r in results)


async def test_the_worker_survives_a_failed_batch() -> None:
    """A crash in one batch must not wedge the queue for everything after it."""
    backend = FakeHF(max_batch_size=2)
    backend.fail = True
    await asyncio.gather(*(backend.generate(_req(i)) for i in range(2)))

    backend.fail = False
    ok = await backend.generate(_req(99))
    assert ok.ok and ok.text == "reply-99"


def test_the_backend_survives_being_reused_across_event_loops() -> None:
    """`calibrate` builds the backend once and calls asyncio.run per difficulty.

    Queues and locks bind to a loop at first use, so a backend that cached them
    would fail on the second difficulty -- after the weights were loaded and the
    first sweep had already been paid for.
    """
    backend = FakeHF(max_batch_size=4)

    first = asyncio.run(backend.generate(_req(1)))
    second = asyncio.run(backend.generate(_req(2)))

    assert first.text == "reply-1"
    assert second.text == "reply-2"
    assert backend.batches == [1, 1]


def test_mixed_sampling_parameters_are_never_batched_together() -> None:
    """`generate` takes one temperature for the whole call, so a mixed batch
    would silently apply one episode's sampling to another's turn."""
    waiters = [
        _Waiter(_req(1, temperature=0.8), None),
        _Waiter(_req(2, temperature=0.2), None),
        _Waiter(_req(3, temperature=0.8), None),
        _Waiter(_req(4, max_tokens=64), None),
    ]
    groups = _group_by_sampling(waiters)

    assert sorted(len(g) for g in groups) == [1, 1, 2]
    for group in groups:
        params = {
            (w.request.max_tokens, w.request.temperature, w.request.top_p)
            for w in group
        }
        assert len(params) == 1


def test_token_budget_splits_long_contexts_and_packs_short_ones() -> None:
    """The batch that OOMs is the one carrying full transcripts.

    Cost is `count * (longest + max_new)`, so a sequence-count cap sized for
    round-one turns is far too large by round three. Short turns must still pack
    in, or the card idles on the cheap half of every episode."""
    waiters = [_Waiter(_req(i), None) for i in range(4)]

    short = _split_by_token_budget(
        waiters, [100] * 4, budget=8192, max_new_tokens=512
    )
    assert len(short) == 1  # 4 * (100 + 512) = 2448, fits

    long = _split_by_token_budget(
        waiters, [6000] * 4, budget=8192, max_new_tokens=512
    )
    assert [len(c) for c in long] == [1, 1, 1, 1]  # 6512 each, one at a time


def test_token_budget_preserves_order() -> None:
    """Results map back positionally; packing tighter by reordering would hand
    one episode another episode's turn."""
    waiters = [_Waiter(_req(i), None) for i in range(6)]
    chunks = _split_by_token_budget(
        waiters, [100, 5000, 100, 5000, 100, 100], budget=6000, max_new_tokens=256
    )
    flat = [w.request.seed for c in chunks for w in c]
    assert flat == [0, 1, 2, 3, 4, 5]


def test_length_sorting_groups_similar_contexts() -> None:
    """Left-padding computes every sequence at the longest one's width, so a
    round-one turn batched with a round-three turn is mostly padding."""
    waiters = [_Waiter(_req(i), None) for i in range(6)]
    lengths = [5000, 100, 4800, 120, 4900, 90]

    chunks, _ = _sorted_chunks(waiters, lengths, budget=11000, max_new_tokens=256)

    # All three short turns ride one pass; the long ones pack two then one.
    # Unsorted, each short turn would have been padded up to a ~5000-token
    # neighbour and computed at that width.
    assert [len(c) for c in chunks] == [3, 2, 1]
    assert {w.request.seed for w in chunks[0]} == {1, 3, 5}


def test_sorting_permutation_inverts_to_the_original_order() -> None:
    """Callers map responses back positionally. If the inverse is wrong, one
    episode receives another episode's turn -- a corruption that yields a
    plausible transcript rather than an error."""
    waiters = [_Waiter(_req(i), None) for i in range(7)]
    lengths = [900, 100, 700, 50, 800, 60, 400]

    chunks, order = _sorted_chunks(waiters, lengths, budget=2000, max_new_tokens=100)

    produced = [w.request.seed for chunk in chunks for w in chunk]
    restored = [None] * len(waiters)
    for position, original in enumerate(order):
        restored[original] = produced[position]

    assert restored == list(range(7))


def test_a_single_oversized_request_is_not_dropped() -> None:
    """Splitting cannot help one request, and the card can usually still run it."""
    waiters = [_Waiter(_req(0), None)]
    chunks = _split_by_token_budget(waiters, [99999], budget=1024, max_new_tokens=512)
    assert chunks == [waiters]


def test_batch_seed_is_deterministic_and_composition_sensitive() -> None:
    a = [_Waiter(_req(1), None), _Waiter(_req(2), None)]
    b = [_Waiter(_req(1), None), _Waiter(_req(2), None)]
    c = [_Waiter(_req(1), None), _Waiter(_req(3), None)]

    assert _batch_seed(a) == _batch_seed(b)
    assert _batch_seed(a) != _batch_seed(c)


def test_batch_seed_fits_the_set_seed_range() -> None:
    """`transformers.set_seed` rejects >= 2**32; a 64-bit hash would crash the
    very first batch of a run."""
    for seeds in ([0], [2**31], list(range(32))):
        waiters = [_Waiter(_req(s), None) for s in seeds]
        assert 0 <= _batch_seed(waiters) < 2**32


def test_config_builds_the_backend_without_importing_torch() -> None:
    from collabengine.config import ExperimentConfig

    config = ExperimentConfig.from_dict(
        {"backend": {"kind": "hf", "model": "Qwen/Qwen3-8B", "max_batch_size": 24}}
    )
    assert config.backend.to_dict()["max_batch_size"] == 24

    pytest.importorskip("torch")
    backend = config.backend.build()
    assert isinstance(backend, HFLocalBackend)
    assert backend.model_id == "Qwen/Qwen3-8B"
    assert backend.max_batch_size == 24
    # Off by default: a Qwen3 <think> block would eat the whole token budget
    # before the agent said anything to its team.
    assert backend.enable_thinking is False


def test_heartbeat_reports_between_episode_writes(capsys) -> None:
    """The only signal a stage emits before its episodes land in a clump.

    Exercised directly rather than through `generate`: `FakeHF` replaces
    `_generate_batch`, which is where the heartbeat is called from, so a test
    driving the queue would pass without ever running this code.
    """
    backend = FakeHF(model_id="m", heartbeat_s=0.0001)
    backend._passes, backend._sequences = 4, 48
    backend._generated, backend._busy_s = 900, 9.0

    backend._heartbeat(prompt_len=2510)

    line = capsys.readouterr().err
    assert "4 passes" in line
    assert "mean batch 12.0" in line   # the number that explains a slow run
    assert "~2510 tok" in line
    assert "100 tok/s" in line


def test_heartbeat_stays_quiet_until_the_interval_elapses() -> None:
    """A line per forward pass would bury the run it is meant to describe."""
    backend = FakeHF(model_id="m", heartbeat_s=3600)
    backend._passes, backend._sequences = 1, 1
    backend._generated, backend._busy_s = 10, 1.0

    backend._heartbeat(prompt_len=100)
    first = backend._last_heartbeat
    backend._heartbeat(prompt_len=100)

    assert backend._last_heartbeat == first


def test_heartbeat_can_be_switched_off(capsys) -> None:
    backend = FakeHF(model_id="m", heartbeat_s=0)
    backend._passes, backend._sequences = 1, 1
    backend._generated, backend._busy_s = 10, 1.0

    backend._heartbeat(prompt_len=100)

    assert capsys.readouterr().err == ""


def test_one_failing_chunk_does_not_blank_the_others() -> None:
    """The bug that cost a 12-episode corpus.

    A batch is split into chunks by token budget and each runs as its own
    forward pass. When one chunk raised, the exception escaped `_generate_batch`
    and the queue worker failed the *whole* group -- so a single round-three
    context too long to fit blanked every turn batched alongside it. Measured
    consequence: 96 of 144 agent turns came back empty, each recorded as a 0.0
    indistinguishable from a team that answered badly.
    """
    calls = {"n": 0}

    class Flaky(FakeHF):
        def _generate_batch(self, batch):
            # Only the innermost per-chunk calls reach here in the fake; fail
            # the second one and leave the rest alone.
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("CUDA OOM on a single sequence")
            return [
                GenResponse(text=f"ok-{w.request.seed}", completion_tokens=3)
                for w in batch
            ]

    backend = Flaky(model_id="m")
    chunks = [[_Waiter(request=_req(i), future=None)] for i in range(3)]

    produced = []
    for chunk in chunks:
        try:
            produced.extend(backend._generate_batch(chunk))
        except Exception as exc:  # mirrors the guard in the real chunk loop
            produced.extend(
                GenResponse(text="", finish_reason="error", error=str(exc))
                for _ in chunk
            )

    assert len(produced) == 3
    assert [r.finish_reason for r in produced] == ["stop", "error", "stop"]
    assert produced[0].text == "ok-0"
    assert produced[2].text == "ok-2"


def test_release_reclaims_between_chunks() -> None:
    """Reserved memory, not live memory, is what the fraction cap counts."""
    from collabengine.backends.hf_local import _release

    class FakeTorch:
        def __init__(self):
            self.emptied = 0
            self.cuda = self

        def empty_cache(self):
            self.emptied += 1

    t = FakeTorch()
    _release(t)
    assert t.emptied == 1


def test_long_prompts_lose_their_oldest_messages_not_their_instructions() -> None:
    """Right truncation would delete the answer contract and keep the history.

    A transcript outgrows max_model_len in the final round of a large instance.
    The tokenizer's default truncation_side is "right", which for a chat prompt
    removes the newest teammate turns, the final-round banner and the
    answer-format contract, while preserving the oldest messages. The agent is
    then asked to answer with the instructions on how to answer deleted, and the
    unparseable turn that follows is indistinguishable from a team that failed
    the task.
    """
    from transformers import AutoTokenizer  # noqa: PLC0415 - optional dependency

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    text = " ".join(f"w{i}" for i in range(400)) + " ANSWER-CONTRACT-SENTINEL"

    tok.truncation_side = "left"
    kept = tok(text, truncation=True, max_length=64)["input_ids"]

    assert "ANSWER-CONTRACT-SENTINEL" in tok.decode(kept)
