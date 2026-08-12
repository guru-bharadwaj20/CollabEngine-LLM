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

ERRORED = "error"
"""`finish_reason` a backend reports when generation raised.

These carry no model output at all. They arrive in bursts rather than
singly -- one oversized batch OOMs, the halved retries OOM against the same
un-reclaimed memory, and an entire stage records empty turns."""


def _agent_turns(messages: Sequence[Message]) -> list[Message]:
    return [m for m in messages if m.speaker is Speaker.AGENT]


def was_truncated(message: Message) -> bool:
    return (message.meta or {}).get("finish_reason") == TRUNCATED


def errored(message: Message) -> bool:
    """Whether the backend failed to generate this turn at all.

    Distinct from truncation: a truncated turn contains real model output that
    stopped early, while an errored one contains nothing. A CUDA OOM that
    cascades through a stage produces episodes of these, every one of which
    grades 0.0 and is indistinguishable in the means from a team that tried and
    failed.
    """
    return (message.meta or {}).get("finish_reason") == ERRORED


def is_instrument_failure(record: EpisodeRecord) -> bool:
    """Whether this episode's zero came from the harness, not the team.

    Two ways that happens, and they need different rules.

    A generation that *errored* is never attributable to the team, so a single
    such turn condemns the episode regardless of what it scored -- there is no
    reading under which a batch that OOMed tells us something about the agent
    whose turn it was.

    Truncation is weaker evidence, because a turn cut off at `max_tokens` still
    contains real output. It only counts when the answer is also unparseable
    *and* every turn was cut off, i.e. no turn ever had the chance to finish. A
    malformed answer after a turn that ended on its own is a real failure to
    commit and stays in the means.
    """
    turns = _agent_turns(record.messages)
    if not turns:
        return False
    if any(errored(m) for m in turns):
        return True
    if not record.solution.malformed:
        return False
    return all(was_truncated(m) for m in turns)


def final_turn_truncated(record: EpisodeRecord) -> bool:
    """Whether the answer-submitting turn was cut off at `max_tokens`.

    This is the truncation that costs a score, and it does not fall equally on
    the two arms. A solo episode is three turns and the last one carries the
    entire answer; a team episode is twelve and the last one commits an answer
    the transcript already contains. So the same cap lands on solo's answer and
    on the team's summary of one.

    Measured on the served `medium` corpus: of 7 solo episodes whose final turn
    was truncated, 5 were malformed, against 1 of the 17 whose final turn
    finished. In the team arm the final turn was truncated 0 times in 24.

    `is_instrument_failure` does not catch this. Its truncation rule requires
    *every* turn to be cut off, which at the measured per-turn rates is a 1-in-70
    event over three solo turns and a 1-in-10^13 event over twelve team turns --
    so the exclusion it offers is one the team arm can essentially never
    qualify for. That rule stays as the preregistered one; this is the sensitivity
    check reported beside it (RESEARCH-LOG 4.9).
    """
    turns = _agent_turns(record.messages)
    return bool(turns) and was_truncated(turns[-1])


@dataclass(slots=True)
class ConditionIntegrity:
    condition: str
    episodes: int = 0
    malformed: int = 0
    instrument_failures: int = 0
    agent_turns: int = 0
    truncated_turns: int = 0
    errored_turns: int = 0
    final_truncated: int = 0

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
    def errored_turns(self) -> int:
        return sum(c.errored_turns for c in self.by_condition.values())

    @property
    def truncation_rate(self) -> float:
        return self.truncated_turns / self.agent_turns if self.agent_turns else 0.0

    @property
    def final_truncated(self) -> int:
        return sum(c.final_truncated for c in self.by_condition.values())

    def lines(self) -> list[str]:
        """A table, widest column first, ordered by how damaged the cell is.

        `cut@end` is here rather than in a separate report because it is the
        column that reads differently across arms: it counts episodes whose
        answer-bearing turn hit the cap, which is a score lost to the instrument
        even when `unusable` stays 0.
        """
        head = (
            f"{'condition':<28}{'eps':>5}{'unusable':>10}{'trunc':>9}"
            f"{'malformed':>11}{'cut@end':>9}"
        )
        rows = [head, "-" * len(head)]
        for name in sorted(
            self.by_condition,
            key=lambda k: (-self.by_condition[k].instrument_failures, k),
        ):
            c = self.by_condition[name]
            rows.append(
                f"{name:<28}{c.episodes:>5}{c.instrument_failures:>10}"
                f"{c.truncation_rate:>8.0%}{c.malformed:>11}{c.final_truncated:>9}"
            )
        rows.append("-" * len(head))
        rows.append(
            f"{'all':<28}{self.episodes:>5}{self.instrument_failures:>10}"
            f"{self.truncation_rate:>8.0%}{self.malformed:>11}"
            f"{self.final_truncated:>9}"
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
        if final_turn_truncated(record):
            cell.final_truncated += 1
        for message in _agent_turns(record.messages):
            cell.agent_turns += 1
            if was_truncated(message):
                cell.truncated_turns += 1
            if errored(message):
                cell.errored_turns += 1
    return report


def finish_reasons(records: Iterable[EpisodeRecord]) -> Counter[str]:
    """Raw `finish_reason` tally, for when the summary looks wrong."""
    tally: Counter[str] = Counter()
    for record in records:
        for message in _agent_turns(record.messages):
            tally[(message.meta or {}).get("finish_reason") or "unset"] += 1
    return tally
