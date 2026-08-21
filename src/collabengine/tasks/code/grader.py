"""Deterministic per-component grading for the code family.

Never a scalar, for the same reason the allocation grader is never a scalar:
the primary analysis is an agent x component interaction, so component scores
are the unit of measurement and `overall` exists for calibration plots and
nothing else.

The four columns are scored from disjoint evidence, which is the property that
makes them independent rather than four views of one number:

  syntax   does it compile, and are the functions the spec named actually there
  base     the base-case hidden tests
  edge     the edge-case hidden tests
  verify   the tests that separate the correct pipeline from the starter's bug

`generator._sort_tests_around_the_bug` is what guarantees the last three
partitions do not overlap. This module only counts.

Grading spawns one child process per submission and compares against expected
values already stored on the instance, so it is deterministic, offline, and
rescorable from `(seed, difficulty)` alone -- the property
`scripts/analysis/rescore.py` relies on for the allocation corpus.
"""

from __future__ import annotations

from typing import Any

from collabengine.tasks.code.sandbox import SandboxResult, run_submission
from collabengine.tasks.code.schema import (
    CODE_COMPONENTS,
    KIND_COMPONENT,
    CodeInstance,
    CodeSolution,
    TestKind,
)
from collabengine.tasks.grader import GradeResult
from collabengine.tasks.schema import Component


def grade(instance: CodeInstance, solution: CodeSolution) -> GradeResult:
    """Score a submission against an instance.

    A malformed solution scores zero everywhere rather than raising, so a team
    that never emits a code block still contributes a data point instead of
    dropping the episode and biasing the sample toward successful runs.
    """
    if solution.malformed or not solution.code.strip():
        return GradeResult(
            per_component={c: 0.0 for c in CODE_COMPONENTS},
            overall=0.0,
            satisfied=_all_false(instance),
            detail={"malformed": True},
        )

    outcome = run_submission(
        solution.code,
        instance.required_functions,
        [t.to_dict() for t in instance.tests],
    )

    satisfied: dict[str, bool] = {"parses": outcome.parsed, "loads": outcome.loaded}
    for name in instance.required_functions:
        satisfied[f"defines:{name}"] = outcome.defined.get(name, False)
    for t in instance.tests:
        satisfied[t.tid] = outcome.passed.get(t.tid, False)

    per_component: dict[Component, float] = {
        Component.SYNTAX: _score_syntax(instance, outcome)
    }
    for kind in (TestKind.BASE, TestKind.EDGE, TestKind.VERIFY):
        cases = instance.tests_of(kind)
        per_component[KIND_COMPONENT[kind]] = (
            sum(satisfied[t.tid] for t in cases) / len(cases) if cases else 1.0
        )

    overall = sum(per_component[c] for c in CODE_COMPONENTS) / len(CODE_COMPONENTS)
    return GradeResult(
        per_component=per_component,
        overall=overall,
        satisfied=satisfied,
        detail={
            "parsed": outcome.parsed,
            "loaded": outcome.loaded,
            "timed_out": outcome.timed_out,
            "submitted_chars": len(solution.code),
            "first_error": outcome.error,
            **outcome.detail,
        },
    )


def _score_syntax(instance: CodeInstance, outcome: SandboxResult) -> float:
    """Compile, then the required names.

    All-or-nothing on compilation and graded on the names, rather than graded on
    both: a module that does not parse has not partially defined anything, and
    averaging a half-credit for "it nearly compiled" would put weight on a
    distinction the score cannot support.
    """
    if not outcome.parsed:
        return 0.0
    required = instance.required_functions
    if not required:
        return 1.0
    return sum(outcome.defined.get(n, False) for n in required) / len(required)


def _all_false(instance: CodeInstance) -> dict[str, bool]:
    out: dict[str, Any] = {"parses": False, "loads": False}
    for name in instance.required_functions:
        out[f"defines:{name}"] = False
    for t in instance.tests:
        out[t.tid] = False
    return out


def audit_satisfiable(instance: CodeInstance) -> bool:
    """Confirm the reference module actually scores 1.0.

    Generation is reference-first, so this should always hold. It is asserted in
    tests because the reference is *emitted as source* and then run through the
    sandbox: any drift between the Python functions the generator computes
    expected values with and the source it writes shows up here, and nowhere
    else, and would silently cap the ceiling below 1.0 and corrupt every
    downstream difficulty calibration.
    """
    return grade(instance, CodeSolution(code=instance.reference_code)).overall == 1.0
