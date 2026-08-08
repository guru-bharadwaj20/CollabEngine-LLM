"""Which episodes the instrument, rather than the team, failed to answer.

An episode whose answer will not parse is graded 0.0 on every component. That
number is indistinguishable in the analysis from a team that answered and
answered badly, and the two mean opposite things:

* a team that produced a wrong answer is a *result* -- it belongs in the means;
* a team whose last turn was cut off mid-sentence by `max_tokens` never got to
  answer at all, and scoring it zero measures the token cap.

The distinction matters more than a few lost episodes would suggest, because
truncation is not random with respect to the thing under study. A turn is cut
off when the agent writes a lot, and how much each agent writes is precisely the
emergent behaviour the experiment is trying to detect. Silently scoring the
verbose agent's episodes zero therefore manufactures an agent x component
interaction out of the token budget -- a false positive in the one direction the
study cannot afford.

So the rule here is deliberately narrow. An episode is charged to the instrument
only when *every* agent turn hit the cap: no turn ever had the opportunity to
finish, so no claim about the team can be read out of it. A malformed episode in
which some turn ended on its own is a genuine failure to answer and stays in.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from collabengine.protocol import Message, Speaker
from collabengine.transcripts.store import EpisodeRecord

TRUNCATED = "length"
"""`finish_reason` a backend reports when it stopped at `max_tokens`."""


def _agent_turns(messages: Sequence[Message]) -> list[Message]:
    return [m for m in messages if m.speaker is Speaker.AGENT]


def was_truncated(message: Message) -> bool:
    return (message.meta or {}).get("finish_reason") == TRUNCATED


def is_instrument_failure(record: EpisodeRecord) -> bool:
    """Whether this episode's zero is the token cap talking, not the team.

    Requires the answer to be unparseable *and* every agent turn to have been
    cut off. Both halves are needed: truncation alone is common and usually
    harmless, and a malformed answer alone can be a real refusal to commit.
    """
    if not record.solution.malformed:
        return False
    turns = _agent_turns(record.messages)
    return bool(turns) and all(was_truncated(m) for m in turns)


@dataclass(slots=True)
class ConditionIntegrity:
    condition: str
    episodes: int = 0
    malformed: int = 0
    instrument_failures: int = 0
    agent_turns: int = 0
    truncated_turns: int = 0

    @property
    def truncation_rate(self) -> float:
        return self.truncated_turns / self.agent_turns if self.agent_turns else 0.0

    @property
    def usable(self) -> int:
        return self.episodes - self.instrument_failures


@dataclass(slots=True)
class IntegrityReport:
    """Per-condition accounting of what the token cap cost the corpus."""

    by_condition: dict[str, ConditionIntegrity] = field(default_factory=dict)

    @property
    def episodes(self) -> int:
        return sum(c.episodes for c in self.by_condition.values())

    @property
    def instrument_failures(self) -> int:
        return sum(c.instrument_failures for c in self.by_condition.values())

    @property
    def malformed(self) -> int:
        return sum(c.malformed for c in self.by_condition.values())

    @property
    def agent_turns(self) -> int:
        return sum(c.agent_turns for c in self.by_condition.values())

    @property
    def truncated_turns(self) -> int:
        return sum(c.truncated_turns for c in self.by_condition.values())

    @property
    def truncation_rate(self) -> float:
        return self.truncated_turns / self.agent_turns if self.agent_turns else 0.0

    def lines(self) -> list[str]:
        """A table, widest column first, ordered by how damaged the cell is."""
        head = f"{'condition':<28}{'eps':>5}{'unusable':>10}{'trunc':>9}"
        rows = [head, "-" * len(head)]
        for name in sorted(
            self.by_condition,
            key=lambda k: (-self.by_condition[k].instrument_failures, k),
        ):
            c = self.by_condition[name]
            rows.append(
                f"{name:<28}{c.episodes:>5}{c.instrument_failures:>10}"
                f"{c.truncation_rate:>8.0%}"
            )
        rows.append("-" * len(head))
        rows.append(
            f"{'all':<28}{self.episodes:>5}{self.instrument_failures:>10}"
            f"{self.truncation_rate:>8.0%}"
        )
        return rows


def audit(records: Iterable[EpisodeRecord]) -> IntegrityReport:
    report = IntegrityReport()
    for record in records:
        cell = report.by_condition.setdefault(
            record.condition, ConditionIntegrity(record.condition)
        )
        cell.episodes += 1
        if record.solution.malformed:
            cell.malformed += 1
        if is_instrument_failure(record):
            cell.instrument_failures += 1
        for message in _agent_turns(record.messages):
            cell.agent_turns += 1
            if was_truncated(message):
                cell.truncated_turns += 1
    return report


def finish_reasons(records: Iterable[EpisodeRecord]) -> Counter[str]:
    """Raw `finish_reason` tally, for when the summary looks wrong."""
    tally: Counter[str] = Counter()
    for record in records:
        for message in _agent_turns(record.messages):
            tally[(message.meta or {}).get("finish_reason") or "unset"] += 1
    return tally
