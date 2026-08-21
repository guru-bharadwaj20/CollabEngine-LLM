"""The truncation and verbosity diagnostic, pointed at someone else's corpus.

`integrity.py` answers one question about this project's own transcripts: how
much of a reported gap between arms is the token cap rather than the teams. The
answer there was that a per-turn cap identical in specification lands unequally
in practice, because it lands as a function of how much an arm writes -- and how
much an arm writes is the behaviour under study (RESEARCH-LOG 4.10, 4.14, 4.23).
A fourth form of the same asymmetry is the answer parser: 25 of 149 solo
episodes emitted text the parser scored 0.000, against 0 of 150 team episodes,
which moved the solo mean by more than the entire measured gap (RESEARCH-LOG
4.24).

None of that is a property of this codebase. Any comparison that gives both arms
the same per-turn cap, the same turn count, and the same answer format inherits
it, and most published multi-agent-vs-solo comparisons do all three. So the
accounting is worth more outside this repo than inside it, and this module is
the outside-facing half: an adapter from a minimal foreign transcript schema
(documented in docs/AUDIT-SCHEMA.md) onto `EpisodeRecord`, so that
`integrity.audit` -- the same code that produced this project's own numbers --
runs unchanged on a corpus it has never seen.

Two design commitments follow from what the tool is for.

**It never guesses at a missing field.** A foreign corpus that records no
`finish_reason` cannot support a truncation rate, and inferring one from text
length or trailing punctuation would manufacture exactly the kind of number this
project exists to warn about. Every derived quantity is either computed from
fields that are present on *every* relevant row or reported as not computable,
with the reason. A partial tally is worse than none, because a corpus that
records `finish_reason` on only the turns some layer thought interesting is not
missing those rows at random.

**A clean audit is a result.** The report says "not present" as plainly as it
says "flagged". A diagnostic that only speaks when it finds something is a
diagnostic no one can use to clear a system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from collabengine.analysis.integrity import ERRORED, TRUNCATED, IntegrityReport
from collabengine.analysis.integrity import audit as integrity_audit
from collabengine.protocol import Message, Speaker
from collabengine.tasks.grader import GradeResult
from collabengine.transcripts.store import EpisodeRecord

DEFAULT_RATIO_THRESHOLD = 1.5
"""Verbosity ratio above which the report raises the artifact flag.

Not a convention borrowed from anywhere. It sits below the 1.87 measured on this
project's own matched-budget arm -- one agent generating 25,901 characters per
episode against the four-agent team's 13,873, on an identical turn count and an
identical per-turn cap (RESEARCH-LOG 4.23) -- which is the smallest ratio at
which the artifact has been demonstrated to distort a headline here. A ratio
under it is not a clean bill of health, only an absence of evidence at the
sensitivity this tool has; the report says so.
"""

GENERATING_ROLES = frozenset({"assistant", "agent", "model", "ai"})
"""Roles whose text the audited system generated.

Prompt-side rows (`system`, `user`, `human`, `tool`) are excluded from every
count. Counting them would inflate the arm with the longer scaffold, which on a
team-vs-solo comparison is systematically the team -- the opposite bias to the
one being measured, and just as manufactured.
"""

_TRUNCATION_ALIASES = frozenset(
    {"length", "max_tokens", "max_output_tokens", "max_tokens_reached", "truncated"}
)
"""Spellings of "stopped at the cap" seen across published harnesses.

Normalised rather than matched loosely: an unrecognised value passes through as
itself and simply does not count as truncation, so a schema this list does not
know about under-reports visibly instead of guessing.
"""

_ERROR_ALIASES = frozenset({"error", "exception", "failed", "content_filter"})


class SchemaError(ValueError):
    """A record that does not meet the minimal schema.

    Carries the source location, because the usual cause is a converter that got
    one field name wrong for an entire corpus and the fix is a one-line change to
    the converter rather than to this module.
    """


def _first(d: dict[str, Any], *names: str) -> Any:
    for name in names:
        if d.get(name) is not None:
            return d[name]
    return None


def normalize_finish_reason(raw: Any) -> str | None:
    """Map a foreign stop reason onto the vocabulary `integrity` understands.

    Returns `None` when the field is absent, and that `None` propagates all the
    way to "truncation not computable for this arm" rather than being read as
    "finished normally". The two are different claims and only one of them is
    supported by a corpus that does not record the field.
    """
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if not value:
        return None
    if value in _TRUNCATION_ALIASES:
        return TRUNCATED
    if value in _ERROR_ALIASES:
        return ERRORED
    return value


@dataclass(slots=True)
class ForeignSolution:
    """Stands in for a task family's solution type across the adapter.

    `integrity` reads exactly one attribute off `EpisodeRecord.solution`, and a
    foreign corpus has no instance to grade against, so this carries that one
    attribute and nothing else. `malformed` is tri-state: `None` means the corpus
    did not say whether the answer parsed, which is not the same as saying it
    did.
    """

    malformed: bool | None = None


@dataclass(slots=True)
class ForeignTurn:
    """One generated turn, after adaptation."""

    author: str
    text: str
    finish_reason: str | None = None
    tokens: int | None = None


def turn_from_dict(d: dict[str, Any], *, where: str) -> ForeignTurn:
    """Adapt one turn. The field list is in docs/AUDIT-SCHEMA.md."""
    if not isinstance(d, dict):
        raise SchemaError(f"{where}: turn is {type(d).__name__}, expected an object")
    text = _first(d, "text", "content", "message")
    if text is None:
        raise SchemaError(f"{where}: turn has no `text`")
    tokens = _first(d, "tokens", "completion_tokens", "output_tokens")
    if tokens is not None:
        try:
            tokens = int(tokens)
        except (TypeError, ValueError) as exc:
            raise SchemaError(f"{where}: `tokens` is not an integer") from exc
    return ForeignTurn(
        author=str(_first(d, "agent", "author", "name") or "agent"),
        text=str(text),
        finish_reason=normalize_finish_reason(
            _first(d, "finish_reason", "stop_reason")
        ),
        tokens=tokens,
    )


def is_generated(d: dict[str, Any]) -> bool:
    """Whether this row is model output rather than prompt scaffolding.

    A row with no `role` is treated as generated. That is the permissive
    direction on purpose: the minimal schema asks a converter to emit only the
    generated turns, so the omission means "all of these", and a converter that
    also emits prompts is expected to label them.
    """
    role = _first(d, "role", "speaker")
    return role is None or str(role).strip().lower() in GENERATING_ROLES


def record_from_dict(d: dict[str, Any], *, where: str = "<record>") -> EpisodeRecord:
    """Adapt one foreign episode onto the internal record shape.

    The point of going through `EpisodeRecord` rather than tallying directly is
    that `integrity.audit`, `is_instrument_failure` and `final_turn_truncated`
    then run on foreign data as the identical code that produced this project's
    published numbers. An audit whose accounting differed from the accounting it
    audits against would not be evidence of anything.

    `instance_seed`, `difficulty` and `grade` are filled with placeholders: no
    quantity this module reports reads them, and a foreign corpus has no
    obligation to carry them.
    """
    if not isinstance(d, dict):
        raise SchemaError(f"{where}: record is {type(d).__name__}, expected an object")
    arm = _first(d, "arm", "condition", "system", "group")
    if arm is None:
        raise SchemaError(f"{where}: record has no `arm`")
    raw_turns = _first(d, "turns", "messages")
    if raw_turns is None:
        raise SchemaError(f"{where}: record has no `turns`")
    if not isinstance(raw_turns, list):
        raise SchemaError(f"{where}: `turns` is not a list")

    turns = [
        turn_from_dict(t, where=f"{where} turn {i}")
        for i, t in enumerate(raw_turns)
        if not isinstance(t, dict) or is_generated(t)
    ]

    malformed = d.get("malformed")
    if malformed is not None:
        malformed = bool(malformed)
    elif d.get("answer_parsed") is not None:
        malformed = not bool(d["answer_parsed"])

    messages = [
        Message(
            turn=i,
            speaker=Speaker.AGENT,
            author=t.author,
            content=t.text,
            meta=_turn_meta(t),
        )
        for i, t in enumerate(turns)
    ]
    return EpisodeRecord(
        episode_id=str(_first(d, "episode_id", "id", "task_id") or where),
        condition=str(arm),
        instance_seed=0,
        difficulty=str(d.get("difficulty") or "unknown"),
        agents=sorted({t.author for t in turns}),
        messages=messages,
        solution=ForeignSolution(malformed=malformed),
        grade=GradeResult(per_component={}, overall=0.0, satisfied={}),
        meta={"source": where},
    )


def _turn_meta(turn: ForeignTurn) -> dict[str, Any]:
    """Only the keys the corpus actually supplied.

    Absent keys stay absent rather than being written as `None`, because the
    volume tally distinguishes "no `finish_reason` on this turn" from any value
    of it by key membership -- writing the key with a null would count the turn
    as instrumented.
    """
    meta: dict[str, Any] = {}
    if turn.finish_reason is not None:
        meta["finish_reason"] = turn.finish_reason
    if turn.tokens is not None:
        meta["tokens"] = turn.tokens
    return meta


@dataclass(slots=True)
class ArmVolume:
    """Generation volume for one arm, and how much of it is trustworthy.

    The `*_with_*` counters exist so the report can distinguish "this arm was
    never truncated" from "this arm did not record whether it was truncated".
    Collapsing those two into a zero is the single most damaging thing a tool
    like this could do, because it would report the least instrumented system as
    the cleanest one.
    """

    arm: str
    episodes: int = 0
    turns: int = 0
    characters: int = 0
    tokens: int = 0
    turns_with_tokens: int = 0
    turns_with_finish_reason: int = 0
    episodes_with_parse_status: int = 0

    @property
    def characters_per_episode(self) -> float:
        return self.characters / self.episodes if self.episodes else 0.0

    @property
    def tokens_known(self) -> bool:
        """True only when every generated turn carried a token count."""
        return bool(self.turns) and self.turns_with_tokens == self.turns

    @property
    def tokens_per_episode(self) -> float | None:
        if not self.tokens_known or not self.episodes:
            return None
        return self.tokens / self.episodes

    @property
    def truncation_known(self) -> bool:
        """True only when every generated turn carried a `finish_reason`.

        Deliberately all-or-nothing. A corpus that records the field on some
        turns is not a corpus missing it at random -- the usual reason a row
        lacks it is that some wrapper layer dropped it, and wrapper layers
        correlate with arm.
        """
        return bool(self.turns) and self.turns_with_finish_reason == self.turns

    @property
    def parse_status_known(self) -> bool:
        return bool(self.episodes) and self.episodes_with_parse_status == self.episodes


@dataclass(slots=True)
class AuditReport:
    """What a foreign corpus does and does not show about arm asymmetry."""

    by_arm: dict[str, ArmVolume] = field(default_factory=dict)
    integrity: IntegrityReport = field(default_factory=IntegrityReport)
    threshold: float = DEFAULT_RATIO_THRESHOLD
    records_read: int = 0
    records_skipped: int = 0
    skip_reasons: list[str] = field(default_factory=list)

    @property
    def arms(self) -> list[str]:
        return sorted(self.by_arm)

    @property
    def verbosity_ratio(self) -> float | None:
        """Characters per episode, most verbose arm over least.

        Per episode rather than per turn, because the asymmetry this looks for
        survives a matched *turn* budget -- matching turns is exactly what
        produced the 1.87 in RESEARCH-LOG 4.23. A per-turn ratio would have
        divided that finding away.
        """
        rates = [v.characters_per_episode for v in self.by_arm.values() if v.episodes]
        if len(rates) < 2 or min(rates) <= 0:
            return None
        return max(rates) / min(rates)

    def _ranked_arms(self) -> list[str]:
        return sorted(
            (a for a in self.by_arm if self.by_arm[a].episodes),
            key=lambda a: self.by_arm[a].characters_per_episode,
        )

    @property
    def verbose_arm(self) -> str | None:
        ranked = self._ranked_arms()
        return ranked[-1] if len(ranked) >= 2 else None

    @property
    def spare_arm(self) -> str | None:
        ranked = self._ranked_arms()
        return ranked[0] if len(ranked) >= 2 else None

    @property
    def flagged(self) -> bool:
        ratio = self.verbosity_ratio
        return ratio is not None and ratio > self.threshold

    @property
    def truncation_computable(self) -> bool:
        return bool(self.by_arm) and all(
            v.truncation_known for v in self.by_arm.values()
        )

    @property
    def parse_status_computable(self) -> bool:
        return bool(self.by_arm) and all(
            v.parse_status_known for v in self.by_arm.values()
        )

    def malformed_asymmetry(self) -> tuple[str, str] | None:
        """The two arms between which the unparseable-answer rate differs most.

        Returned as a pair rather than as a number because the interesting case
        is categorical: RESEARCH-LOG 4.24 is 25 malformed against 0, and a rate
        ratio against zero is not a quantity. `lines` prints both counts.
        """
        if not self.parse_status_computable or len(self.by_arm) < 2:
            return None
        rates = {
            arm: self.integrity.by_condition[arm].malformed / v.episodes
            for arm, v in self.by_arm.items()
            if v.episodes and arm in self.integrity.by_condition
        }
        if len(rates) < 2:
            return None
        worst = max(rates, key=lambda a: rates[a])
        best = min(rates, key=lambda a: rates[a])
        if rates[worst] == rates[best]:
            return None
        return worst, best

    def to_dict(self) -> dict[str, Any]:
        return {
            "records_read": self.records_read,
            "records_skipped": self.records_skipped,
            "threshold": self.threshold,
            "verbosity_ratio": self.verbosity_ratio,
            "verbose_arm": self.verbose_arm,
            "flagged": self.flagged,
            "truncation_computable": self.truncation_computable,
            "parse_status_computable": self.parse_status_computable,
            "arms": {arm: self._arm_dict(arm) for arm in self.arms},
        }

    def _arm_dict(self, arm: str) -> dict[str, Any]:
        v = self.by_arm[arm]
        cell = self.integrity.by_condition.get(arm)
        return {
            "episodes": v.episodes,
            "turns": v.turns,
            "characters": v.characters,
            "characters_per_episode": v.characters_per_episode,
            "tokens_per_episode": v.tokens_per_episode,
            "truncated_turns": (
                cell.truncated_turns if cell and v.truncation_known else None
            ),
            "answer_turn_cut": (
                cell.final_truncated if cell and v.truncation_known else None
            ),
            "errored_turns": (
                cell.errored_turns if cell and v.truncation_known else None
            ),
            "malformed": cell.malformed if cell and v.parse_status_known else None,
        }

    def lines(self) -> list[str]:
        """The table, then the verdict, then what could not be computed.

        In that order because the verdict is the part a reader acts on and the
        caveats are the part that qualifies it; a reader who stops early should
        stop having read the finding, not having read the column widths.
        """
        head = (
            f"{'arm':<20}{'eps':>5}{'turns':>7}{'chars/ep':>11}{'tok/ep':>9}"
            f"{'trunc':>8}{'cut@end':>9}{'malformed':>11}"
        )
        rows = [head, "-" * len(head)]
        for arm in self.arms:
            v = self.by_arm[arm]
            d = self._arm_dict(arm)
            rows.append(
                f"{arm:<20}{v.episodes:>5}{v.turns:>7}"
                f"{v.characters_per_episode:>11,.0f}"
                f"{_num(d['tokens_per_episode'], 9, ',.0f')}"
                f"{_num(d['truncated_turns'], 8)}"
                f"{_num(d['answer_turn_cut'], 9)}"
                f"{_num(d['malformed'], 11)}"
            )
        rows.append("")
        rows.extend(self._verdict())
        caveats = self._caveats()
        if caveats:
            rows.append("")
            rows.extend(caveats)
        return rows

    def _verdict(self) -> list[str]:
        ratio = self.verbosity_ratio
        if ratio is None:
            return [
                "verbosity ratio: not computable -- fewer than two arms carry "
                "episodes, or an arm generated nothing. This tool compares arms; "
                "it says nothing about a single one."
            ]
        out = [
            f"verbosity ratio: {ratio:.2f}x  ({self.verbose_arm} over "
            f"{self.spare_arm}, generated characters per episode)"
        ]
        if self.flagged:
            out.append(
                f"FLAG: above the {self.threshold:.2f}x threshold. If both arms "
                f"ran under the same per-turn generation cap, that cap did not "
                f"fall on them equally -- it binds on {self.verbose_arm} first, "
                f"and the truncation and parse failures that follow are charged "
                f"to {self.verbose_arm}'s score rather than to the instrument."
            )
        else:
            out.append(
                f"artifact not present at this sensitivity: the arms generate "
                f"within {self.threshold:.2f}x of each other, so a shared "
                f"per-turn cap has no large asymmetric purchase here."
            )
        out.extend(self._truncation_verdict())
        out.extend(self._malformed_verdict())
        return out

    def _truncation_verdict(self) -> list[str]:
        if not self.truncation_computable:
            return []
        cut = {a: self.integrity.by_condition[a].final_truncated for a in self.arms}
        if not sum(cut.values()):
            return ["answer-turn cuts: none in any arm."]
        worst = max(cut, key=lambda a: cut[a])
        return [
            f"answer-turn cuts: {sum(cut.values())} across {len(self.arms)} arm(s), "
            f"most in {worst} ({cut[worst]}). Each one is a score lost to the cap "
            f"on the turn that carried the answer."
        ]

    def _malformed_verdict(self) -> list[str]:
        pair = self.malformed_asymmetry()
        if pair is None:
            if self.parse_status_computable and self.by_arm:
                return ["unparseable answers: the same rate in every arm."]
            return []
        worst, best = pair
        return [
            f"unparseable answers: "
            f"{self.integrity.by_condition[worst].malformed} of "
            f"{self.by_arm[worst].episodes} in {worst} against "
            f"{self.integrity.by_condition[best].malformed} of "
            f"{self.by_arm[best].episodes} in {best}. Each one scores zero and is "
            f"counted, so the difference moves {worst}'s mean directly "
            f"(RESEARCH-LOG 4.24)."
        ]

    def _caveats(self) -> list[str]:
        """Every quantity the corpus could not support, and why.

        Printed even when nothing is flagged, because "we looked and the field
        was not there" is the finding in that case, and a reader who takes an
        unflagged report for a clean audit without seeing this list has been
        misled by the tool.
        """
        out: list[str] = []
        missing = [a for a in self.arms if not self.by_arm[a].truncation_known]
        if missing:
            out.append(
                f"not computable -- truncation and answer-turn cuts for "
                f"{', '.join(missing)}: not every generated turn carries "
                f"`finish_reason`. This is not evidence of no truncation, it is "
                f"the absence of the field that would show it."
            )
        missing = [a for a in self.arms if not self.by_arm[a].tokens_known]
        if missing:
            out.append(
                f"not computable -- tokens per episode for {', '.join(missing)}: "
                f"not every generated turn carries `tokens`. The character ratio "
                f"above stands without them; tokens would only sharpen it."
            )
        missing = [a for a in self.arms if not self.by_arm[a].parse_status_known]
        if missing:
            out.append(
                f"not computable -- unparseable-answer counts for "
                f"{', '.join(missing)}: not every episode carries `answer_parsed` "
                f"or `malformed`. The fourth artifact (RESEARCH-LOG 4.24) cannot "
                f"be checked for on this corpus."
            )
        if self.records_skipped:
            out.append(
                f"{self.records_skipped} record(s) did not meet the schema and "
                f"were skipped; re-run with --strict to stop on the first. "
                f"First: {self.skip_reasons[0]}"
            )
        return out


def _num(value: float | int | None, width: int, fmt: str = ",d") -> str:
    """A right-aligned cell, or `n/a` where the corpus cannot support one."""
    if value is None:
        return f"{'n/a':>{width}}"
    return f"{value:>{width}{fmt}}"


def read_records(
    path: str | Path, *, strict: bool = False
) -> Iterator[tuple[dict[str, Any], str]]:
    """Yield `(record, where)` from a JSONL file, a JSON file, or a directory.

    Three shapes because foreign releases come in all three, and requiring an
    auditor to reshape a corpus before auditing it is how audits do not happen.
    A `.json` file may hold a bare list or an object with an `episodes` (or
    `records`, or `data`) list.
    """
    p = Path(path)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in {".json", ".jsonl"})
        if not files:
            raise SchemaError(f"{p}: directory holds no .json or .jsonl files")
        for f in files:
            yield from read_records(f, strict=strict)
        return
    if not p.exists():
        raise SchemaError(f"{p}: no such file or directory")

    if p.suffix.lower() == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = _first(payload, "episodes", "records", "data") or []
        if not isinstance(payload, list):
            raise SchemaError(f"{p}: expected a list of episodes")
        for i, item in enumerate(payload):
            yield item, f"{p.name}[{i}]"
        return

    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line), f"{p.name}:{lineno}"
            except json.JSONDecodeError as exc:
                if strict:
                    raise SchemaError(f"{p.name}:{lineno}: {exc}") from exc


def audit_records(
    raw: Iterable[tuple[dict[str, Any], str]],
    *,
    threshold: float = DEFAULT_RATIO_THRESHOLD,
    strict: bool = False,
) -> AuditReport:
    """Adapt, then hand the result to the unmodified internal accounting.

    Volume is tallied here rather than in `integrity` because `integrity` has no
    reason to count characters on its own corpus: the cap there is enforced in
    tokens by the backend, and characters are a diagnostic laid over it.
    Everything else on the report comes out of `integrity.audit`.
    """
    report = AuditReport(threshold=threshold)
    records: list[EpisodeRecord] = []
    for item, where in raw:
        try:
            record = record_from_dict(item, where=where)
        except SchemaError as exc:
            if strict:
                raise
            report.records_skipped += 1
            report.skip_reasons.append(str(exc))
            continue
        records.append(record)
        report.records_read += 1

        volume = report.by_arm.setdefault(record.condition, ArmVolume(record.condition))
        volume.episodes += 1
        if record.solution.malformed is not None:
            volume.episodes_with_parse_status += 1
        for message in record.messages:
            volume.turns += 1
            volume.characters += len(message.content)
            if "finish_reason" in message.meta:
                volume.turns_with_finish_reason += 1
            if "tokens" in message.meta:
                volume.turns_with_tokens += 1
                volume.tokens += int(message.meta["tokens"])

    report.integrity = integrity_audit(records)
    return report


def audit_path(
    path: str | Path,
    *,
    threshold: float = DEFAULT_RATIO_THRESHOLD,
    strict: bool = False,
) -> AuditReport:
    """Read and audit a foreign corpus in one call. The CLI subcommand's body."""
    return audit_records(
        read_records(path, strict=strict), threshold=threshold, strict=strict
    )
