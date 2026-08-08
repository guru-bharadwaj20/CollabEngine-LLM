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
