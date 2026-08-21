"""Data model for the code-generation task family.

Same contract as `collabengine.tasks.schema`: plain dataclasses, JSON round-
trippable, no non-serializable state. Instances are persisted into transcripts
and regenerated from `(seed, difficulty)` during rescoring, so anything that
cannot survive a trip through JSONL cannot live here.

The hidden material -- the expected outputs of the test suite, which stage
carries the planted bug, and the reference module -- sits on the instance beside
the agent-visible spec, exactly as `ground_truth` does for the allocation task.
`render.py` is the only place that has to keep the two apart, and
`tests/test_code_task.py` asserts structurally that it does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from collabengine.tasks.schema import Component

#: This family's component axis. Four members, mirroring the allocation task's
#: four so the agent x component analyses carry across unchanged in shape:
#: something is produced, the awkward corners are searched, a planted error is
#: caught rather than written, and the artifact has to hold together at all.
CODE_COMPONENTS: tuple[Component, ...] = (
    Component.SYNTAX,
    Component.BASE,
    Component.EDGE,
    Component.VERIFY,
)


class TestKind(str, Enum):
    """Which component a hidden test scores.

    The three kinds partition the suite: no test counts toward two components.
    That partition is what makes the component scores move independently, and it
    is enforced at generation time -- see `generator._sort_tests_around_the_bug`.
    """

    BASE = "base"
    EDGE = "edge"
    VERIFY = "verify"


#: Test kind -> the component it scores.
KIND_COMPONENT: dict[TestKind, Component] = {
    TestKind.BASE: Component.BASE,
    TestKind.EDGE: Component.EDGE,
    TestKind.VERIFY: Component.VERIFY,
}


@dataclass(frozen=True, slots=True)
class RuleSpec:
    """One stage of the transformation pipeline.

    `kind` selects the reference implementation and its planted mutation;
    `params` carries the numbers the spec sentence quotes. Data rather than
    closures, for the same reason `Constraint` is: an instance has to round-trip
    through JSONL to be replayable.
    """

    rid: str
    kind: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"rid": self.rid, "kind": self.kind, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RuleSpec:
        return cls(rid=d["rid"], kind=d["kind"], params=dict(d["params"]))


@dataclass(frozen=True, slots=True)
class HiddenTest:
    """One hidden test: call `function(list(args))` and expect `expected`.

    Expected values are computed by the generator from the reference pipeline and
    stored, not recomputed at grading time. Grading is then pure comparison
    against data on the instance, which is what makes it offline-rescorable and
    identical on every machine.
    """

    tid: str
    kind: TestKind
    function: str
    args: tuple[int, ...]
    expected: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tid": self.tid,
            "kind": self.kind.value,
            "function": self.function,
            "args": list(self.args),
            "expected": self.expected,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HiddenTest:
        return cls(
            tid=d["tid"],
            kind=TestKind(d["kind"]),
            function=d["function"],
            args=tuple(int(x) for x in d["args"]),
            expected=int(d["expected"]),
        )


@dataclass(frozen=True, slots=True)
class CodeInstance:
    """One implementation problem.

    Everything from `tests` down is withheld from agents. A leak there would do
    to this family what leaking `planted_errors` would do to the allocation one:
    make a component free and inflate every score in the study.
    """

    instance_id: str
    seed: int
    difficulty: str
    rules: tuple[RuleSpec, ...]
    aggregate: str
    empty_default: int
    """What `solve` must return when nothing survives the rules.

    Never the value a naive implementation would produce by accident (0 for a
    sum, 0 for a count), because the empty case is what the `edge` component is
    mostly measuring and a default of 0 would hand it out for free."""
    required_functions: tuple[str, ...]
    starter_code: str
    """Agent-visible. Contains the planted bug."""

    tests: tuple[HiddenTest, ...]
    """Withheld. The graded suite, partitioned by `TestKind`."""
    planted_bug_rule: str
    """Withheld. `rid` of the stage whose starter implementation is wrong."""
    planted_bug_kind: str
    """Withheld. One-word name of the mutation, for error analysis."""
    reference_code: str
    """Withheld. A module that scores 1.0, so the ceiling is known by construction."""

    def rule(self, rid: str) -> RuleSpec | None:
        for r in self.rules:
            if r.rid == rid:
                return r
        return None

    def tests_of(self, kind: TestKind) -> tuple[HiddenTest, ...]:
        return tuple(t for t in self.tests if t.kind is kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "seed": self.seed,
            "difficulty": self.difficulty,
            "rules": [r.to_dict() for r in self.rules],
            "aggregate": self.aggregate,
            "empty_default": self.empty_default,
            "required_functions": list(self.required_functions),
            "starter_code": self.starter_code,
            "tests": [t.to_dict() for t in self.tests],
            "planted_bug_rule": self.planted_bug_rule,
            "planted_bug_kind": self.planted_bug_kind,
            "reference_code": self.reference_code,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CodeInstance:
        return cls(
            instance_id=d["instance_id"],
            seed=int(d["seed"]),
            difficulty=d["difficulty"],
            rules=tuple(RuleSpec.from_dict(x) for x in d["rules"]),
            aggregate=d["aggregate"],
            empty_default=int(d["empty_default"]),
            required_functions=tuple(d["required_functions"]),
            starter_code=d["starter_code"],
            tests=tuple(HiddenTest.from_dict(x) for x in d["tests"]),
            planted_bug_rule=d["planted_bug_rule"],
            planted_bug_kind=d["planted_bug_kind"],
            reference_code=d["reference_code"],
        )


@dataclass(slots=True)
class CodeSolution:
    """What a team is expected to produce: one Python module, as text.

    `malformed` means no code block was found at all -- the analogue of never
    emitting a parseable final answer. Code that *was* found but does not parse
    is not malformed; it is a submission that scores 0 on `syntax`, which is a
    different and more informative outcome.
    """

    code: str = ""
    malformed: bool = False
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "malformed": self.malformed,
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CodeSolution:
        return cls(
            code=str(d.get("code") or ""),
            malformed=bool(d.get("malformed", False)),
            detail=dict(d.get("detail") or {}),
        )
