"""Data model for the allocation task.

Everything here is plain-dataclass + JSON-round-trippable. Transcripts and
instances are persisted as JSONL and re-read during ablation replay, so no type
in this module may carry non-serializable state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Component(str, Enum):
    """Task-component axis.

    These are the columns of the interaction test. Each loads on a different
    capability, so an agent that genuinely specialized should damage its own
    component disproportionately when ablated.
    """

    ARITHMETIC = "arithmetic"
    """Summation and capacity bookkeeping. Exactly checkable."""

    SEARCH = "search"
    """Enumeration over feasible sets under skill/exclusion restrictions."""

    VERIFICATION = "verification"
    """Catching planted errors in a proposed assignment, rather than producing."""

    SYNTHESIS = "synthesis"
    """Constraints coupling two otherwise-disjoint job blocks."""

    # -- Code-generation family (`collabengine.tasks.code`). Four members again,
    # -- carrying the same four roles as the allocation set: produce, search the
    # -- awkward corners, catch a planted error, hold the whole artifact
    # -- together. They live in this enum rather than a parallel one so that
    # -- `GradeResult` round-trips through JSONL unchanged for both families and
    # -- a component name is unambiguous about which task wrote it.

    SYNTAX = "syntax"
    """The submission parses and defines every function the spec named."""

    BASE = "base"
    """Base-case hidden tests. The ordinary path through the specification."""

    EDGE = "edge"
    """Edge-case hidden tests: empty input, boundaries, values that filter away."""

    VERIFY = "verify"
    """Catching the bug planted in the starter code, rather than writing code."""


#: The allocation family's component axis. Deliberately *not* every member of
#: `Component`: analyses that iterate this tuple are analyses of the allocation
#: corpus, and widening it would silently change what they average over. Each
#: family names its own tuple -- see `collabengine.tasks.code.schema`.
ALL_COMPONENTS: tuple[Component, ...] = (
    Component.ARITHMETIC,
    Component.SEARCH,
    Component.VERIFICATION,
    Component.SYNTHESIS,
)


@dataclass(frozen=True, slots=True)
class Job:
    jid: str
    duration: int
    skill: str
    value: int
    block: int
    """Block index. Synthesis constraints couple jobs across different blocks."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "jid": self.jid,
            "duration": self.duration,
            "skill": self.skill,
            "value": self.value,
            "block": self.block,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Job:
        return cls(
            jid=d["jid"],
            duration=int(d["duration"]),
            skill=d["skill"],
            value=int(d["value"]),
            block=int(d["block"]),
        )


@dataclass(frozen=True, slots=True)
class Worker:
    wid: str
    skills: tuple[str, ...]
    capacity: int

    def to_dict(self) -> dict[str, Any]:
        return {"wid": self.wid, "skills": list(self.skills), "capacity": self.capacity}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Worker:
        return cls(
            wid=d["wid"],
            skills=tuple(d["skills"]),
            capacity=int(d["capacity"]),
        )


@dataclass(frozen=True, slots=True)
class Constraint:
    """A single independently-checkable requirement.

    `kind` selects the grader predicate; `params` carries its arguments. Keeping
    constraints data rather than closures is what lets instances round-trip
    through JSONL for replay.
    """

    cid: str
    kind: str
    component: Component
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cid": self.cid,
            "kind": self.kind,
            "component": self.component.value,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Constraint:
        return cls(
            cid=d["cid"],
            kind=d["kind"],
            component=Component(d["component"]),
            params=dict(d["params"]),
        )


@dataclass(frozen=True, slots=True)
class Instance:
    """One task instance.

    `ground_truth` is retained for calibration and satisfiability auditing only.
    It is never rendered into any agent-visible prompt -- see
    `collabengine.tasks.render`.
    """

    instance_id: str
    seed: int
    difficulty: str
    jobs: tuple[Job, ...]
    workers: tuple[Worker, ...]
    constraints: tuple[Constraint, ...]
    proposed_assignment: dict[str, str]
    """A partially-wrong assignment the team must audit (verification component)."""
    planted_errors: tuple[str, ...]
    """Job ids that are wrong in `proposed_assignment`. Withheld from agents."""
    ground_truth: dict[str, str]
    """A known-satisfying assignment. Withheld from agents."""

    def job(self, jid: str) -> Job | None:
        for j in self.jobs:
            if j.jid == jid:
                return j
        return None

    def worker(self, wid: str) -> Worker | None:
        for w in self.workers:
            if w.wid == wid:
                return w
        return None

    def constraints_for(self, component: Component) -> tuple[Constraint, ...]:
        return tuple(c for c in self.constraints if c.component is component)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "jobs": [j.to_dict() for j in self.jobs],
            "workers": [w.to_dict() for w in self.workers],
            "constraints": [c.to_dict() for c in self.constraints],
            "proposed_assignment": dict(self.proposed_assignment),
            "planted_errors": list(self.planted_errors),
            "ground_truth": dict(self.ground_truth),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Instance:
        return cls(
            instance_id=d["instance_id"],
            seed=int(d["seed"]),
            difficulty=d["difficulty"],
            jobs=tuple(Job.from_dict(x) for x in d["jobs"]),
            workers=tuple(Worker.from_dict(x) for x in d["workers"]),
            constraints=tuple(Constraint.from_dict(x) for x in d["constraints"]),
            proposed_assignment=dict(d["proposed_assignment"]),
            planted_errors=tuple(d["planted_errors"]),
            ground_truth=dict(d["ground_truth"]),
        )


@dataclass(slots=True)
class Solution:
    """What a team is expected to produce."""

    assignment: dict[str, str] = field(default_factory=dict)
    """job id -> worker id."""

    flagged_errors: list[str] = field(default_factory=list)
    """Job ids the team believes are wrong in `proposed_assignment`."""

    malformed: bool = False
    """True when the team never emitted a parseable final answer."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignment": dict(self.assignment),
            "flagged_errors": list(self.flagged_errors),
            "malformed": self.malformed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Solution:
        return cls(
            assignment=dict(d.get("assignment") or {}),
            flagged_errors=list(d.get("flagged_errors") or []),
            malformed=bool(d.get("malformed", False)),
        )
