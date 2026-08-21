"""JSONL-backed transcript storage.

Sharded by writer. Parallel workers each own a file and never contend, which
avoids interleaved-line corruption on append -- a real risk at 32-way
concurrency, and one that would be discovered only when parsing a half-written
record days later. `merge_shards` concatenates them afterwards.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from collabengine.protocol import Message
from collabengine.tasks import get_family
from collabengine.tasks.grader import GradeResult


@dataclass(slots=True)
class EpisodeRecord:
    """One complete episode.

    `condition` is the experimental cell (e.g. "baseline", "live:A2",
    "frozen:A2", "capacity:3"). Analysis groups on it, so it must be set by the
    caller that knows why the episode was run rather than inferred later.
    """

    episode_id: str
    condition: str
    instance_seed: int
    difficulty: str
    agents: list[str]
    messages: list[Message]
    solution: Any
    """The task family's solution type. Which one is recorded in
    `config["task"]`; records written before the second family existed carry no
    such key and are read as the allocation family, which is what they are."""
    grade: GradeResult
    turn_order: list[list[str]] = field(default_factory=list)
    """Speaking order per round. Needed for the position-vs-identity test."""
    config: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "condition": self.condition,
            "instance_seed": self.instance_seed,
            "difficulty": self.difficulty,
            "agents": list(self.agents),
            "messages": [m.to_dict() for m in self.messages],
            "solution": self.solution.to_dict(),
            "grade": self.grade.to_dict(),
            "turn_order": [list(r) for r in self.turn_order],
            "config": dict(self.config),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EpisodeRecord:
        return cls(
            episode_id=d["episode_id"],
            condition=d["condition"],
            instance_seed=int(d["instance_seed"]),
            difficulty=d["difficulty"],
            agents=list(d["agents"]),
            messages=[Message.from_dict(m) for m in d["messages"]],
            solution=get_family(
                dict(d.get("config") or {}).get("task")
            ).solution_from_dict(d["solution"]),
            grade=GradeResult.from_dict(d["grade"]),
            turn_order=[list(r) for r in d.get("turn_order", [])],
            config=dict(d.get("config", {})),
            meta=dict(d.get("meta", {})),
        )

    def messages_by(self, agent_id: str) -> list[Message]:
        return [m for m in self.messages if m.author == agent_id]


class TranscriptWriter:
    """Append-only JSONL writer for one shard."""

    def __init__(self, path: str | os.PathLike[str], *, append: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a" if append else "w", encoding="utf-8", newline="\n")
        self._count = 0

    def write(self, record: EpisodeRecord) -> None:
        self._fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        self._count += 1
        # Flush per record: a run that dies at hour six should not lose hour five.
        self._fh.flush()

    @property
    def count(self) -> int:
        return self._count

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> TranscriptWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class TranscriptReader:
    """Streaming JSONL reader.

    Iterates rather than loading: the full corpus is expected to reach hundreds
    of thousands of messages and analysis passes only need one record at a time.
    """

    def __init__(self, path: str | os.PathLike[str], *, strict: bool = False) -> None:
        self.path = Path(path)
        self.strict = strict
        self.skipped = 0

    def __iter__(self) -> Iterator[EpisodeRecord]:
        with self.path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield EpisodeRecord.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    if self.strict:
                        raise ValueError(f"{self.path}:{lineno}: {exc}") from exc
                    # A truncated final line is the expected artifact of a killed
                    # run; skipping beats refusing to read the other 99%.
                    self.skipped += 1

    def load(self) -> list[EpisodeRecord]:
        return list(self)


def merge_shards(
    shards: Iterable[str | os.PathLike[str]],
    out_path: str | os.PathLike[str],
) -> int:
    """Concatenate shard files into one transcript. Returns records written."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w", encoding="utf-8", newline="\n") as dst:
        for shard in shards:
            for record in TranscriptReader(shard):
                dst.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                total += 1
    return total
