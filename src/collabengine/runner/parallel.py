"""Concurrent episode execution with resume.

Concurrency is async rather than process-based. The bottleneck is the serving
GPU, not client-side Python: episodes spend essentially all their wall time
waiting on HTTP, so hundreds of coroutines against one batching server keeps it
fuller than a process pool would, and shares one connection pool while doing it.
(The CPU-bound pieces -- generation, grading -- are microseconds by comparison.)

Resume is not a nicety. The full grid is many GPU-hours on rented hardware; a
dropped connection at hour five must not cost hours one through four.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from collabengine.transcripts.store import (
    EpisodeRecord,
    TranscriptReader,
    TranscriptWriter,
)

EpisodeFactory = Callable[[], Awaitable[EpisodeRecord]]


@dataclass(slots=True)
class RunStats:
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    started_at: float = field(default_factory=time.monotonic)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def rate_per_min(self) -> float:
        return 60.0 * self.completed / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def summary(self) -> str:
        return (
            f"{self.completed} done, {self.failed} failed, {self.skipped} skipped "
            f"in {self.elapsed_s:.0f}s ({self.rate_per_min:.1f}/min)"
        )


@dataclass(slots=True)
class RunPlan:
    """A named unit of work.

    `episode_id` must be derivable without running anything, so a resumed run can
    decide what to skip before spending a single token.
    """

    episode_id: str
    factory: EpisodeFactory


def completed_episode_ids(path: str | Path) -> set[str]:
    """Episode ids already present in a transcript shard (or an empty set)."""
    p = Path(path)
    if not p.exists():
        return set()
    return {record.episode_id for record in TranscriptReader(p)}


async def run_plan(
    plans: Sequence[RunPlan],
    *,
    out_path: str | Path,
    max_concurrency: int = 32,
    resume: bool = True,
    on_progress: Callable[[RunStats], None] | None = None,
    progress_every: int = 25,
) -> RunStats:
    """Execute plans concurrently, streaming results to a transcript shard.

    Results are written in completion order, not plan order. Nothing downstream
    depends on file ordering -- analysis groups by `condition` and
    `instance_seed` -- and forcing order would mean buffering a slow episode's
    successors in memory for no benefit.
    """
    out = Path(out_path)
    stats = RunStats()

    already = completed_episode_ids(out) if resume else set()
    pending = [p for p in plans if p.episode_id not in already]
    stats.skipped = len(plans) - len(pending)

    if not pending:
        return stats

    sem = asyncio.Semaphore(max_concurrency)
    lock = asyncio.Lock()

    with TranscriptWriter(out, append=resume) as writer:

        async def worker(plan: RunPlan) -> None:
            async with sem:
                try:
                    record = await plan.factory()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # One episode's failure must not take down the batch. The
                    # id stays absent from the transcript, so a later resume
                    # retries it for free.
                    async with lock:
                        stats.failed += 1
                    return

            async with lock:
                writer.write(record)
                stats.completed += 1
                if on_progress and stats.completed % progress_every == 0:
                    on_progress(stats)

        await asyncio.gather(*(worker(p) for p in pending))

    if on_progress:
        on_progress(stats)
    return stats


async def gather_bounded(
    coros: Iterable[Awaitable[object]], *, max_concurrency: int
) -> list[object]:
    """Run awaitables with a concurrency ceiling, preserving input order.

    Used by analysis passes that fan out over a corpus, where order does matter.
    """
    sem = asyncio.Semaphore(max_concurrency)

    async def bounded(c: Awaitable[object]) -> object:
        async with sem:
            return await c

    return list(await asyncio.gather(*(bounded(c) for c in coros)))
