"""Deterministic per-component grading.

The grader deliberately never returns only a scalar. Phase 3's primary analysis
is an agent x component interaction, so component scores are the unit of
measurement; `overall` exists for calibration plots and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from collabengine.tasks.schema import (
    ALL_COMPONENTS,
    Component,
    Constraint,
    Instance,
    Solution,
)


@dataclass(slots=True)
class GradeResult:
    per_component: dict[Component, float]
    overall: float
    satisfied: dict[str, bool]
    """Constraint id -> satisfied. Kept for error analysis and debugging."""
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_component": {c.value: v for c, v in self.per_component.items()},
            "overall": self.overall,
            "satisfied": dict(self.satisfied),
            "detail": dict(self.detail),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GradeResult:
        return cls(
            per_component={
                Component(k): float(v) for k, v in d["per_component"].items()
            },
            overall=float(d["overall"]),
            satisfied={k: bool(v) for k, v in d.get("satisfied", {}).items()},
            detail=dict(d.get("detail", {})),
        )


def grade(instance: Instance, solution: Solution) -> GradeResult:
    """Score a solution against an instance.

    A malformed solution scores zero everywhere rather than raising, so a team
    that never converges still contributes a data point instead of dropping the
    episode and biasing the sample toward successful runs.
    """
    if solution.malformed:
        return GradeResult(
            per_component={c: 0.0 for c in ALL_COMPONENTS},
            overall=0.0,
            satisfied={c.cid: False for c in instance.constraints},
            detail={"malformed": True},
        )

    assignment = _sanitize(instance, solution.assignment)
    satisfied: dict[str, bool] = {}
    for c in instance.constraints:
        satisfied[c.cid] = _check(c, instance, assignment)

    per_component: dict[Component, float] = {}
    for comp in ALL_COMPONENTS:
        if comp is Component.VERIFICATION:
            continue
        cs = instance.constraints_for(comp)
        per_component[comp] = (
            sum(satisfied[c.cid] for c in cs) / len(cs) if cs else 1.0
        )

    verification, vdetail = _score_verification(instance, solution)
    per_component[Component.VERIFICATION] = verification

    overall = sum(per_component[c] for c in ALL_COMPONENTS) / len(ALL_COMPONENTS)
    return GradeResult(
        per_component=per_component,
        overall=overall,
        satisfied=satisfied,
        detail={
            "assigned_jobs": len(assignment),
            "total_jobs": len(instance.jobs),
            "dropped_invalid": len(solution.assignment) - len(assignment),
            **vdetail,
        },
    )


def _sanitize(instance: Instance, assignment: dict[str, str]) -> dict[str, str]:
    """Drop entries naming a job or worker that does not exist.

    Hallucinated ids are a real failure mode at 7-8B. Dropping them (rather than
    erroring) makes the constraint they touch fail naturally, which is the
    behavior we want to score.
    """
    valid_jobs = {j.jid for j in instance.jobs}
    valid_workers = {w.wid for w in instance.workers}
    return {
        jid: wid
        for jid, wid in assignment.items()
        if jid in valid_jobs and wid in valid_workers
    }


def _check(c: Constraint, instance: Instance, assignment: dict[str, str]) -> bool:
    kind = c.kind
    p = c.params

    if kind == "capacity":
        used = sum(
            j.duration for j in instance.jobs if assignment.get(j.jid) == p["worker"]
        )
        return used <= int(p["limit"])

    if kind == "value_floor":
        got = sum(j.value for j in instance.jobs if j.jid in assignment)
        return got >= int(p["threshold"])

    if kind == "skill_match":
        wid = assignment.get(p["job"])
        if wid is None:
            return False
        worker = instance.worker(wid)
        return worker is not None and p["skill"] in worker.skills

    if kind == "exclusion":
        wid = assignment.get(p["job"])
        if wid is None:
            return False
        return wid not in set(p["banned"])

    if kind == "co_assign":
        a, b = assignment.get(p["job_a"]), assignment.get(p["job_b"])
        return a is not None and a == b

    if kind == "separate":
        a, b = assignment.get(p["job_a"]), assignment.get(p["job_b"])
        return a is not None and b is not None and a != b

    raise ValueError(f"unknown constraint kind {kind!r}")


def _score_verification(
    instance: Instance, solution: Solution
) -> tuple[float, dict[str, Any]]:
    """F1 over flagged errors against the planted set.

    F1 rather than accuracy: flagging every job would otherwise score well, and
    the component is meant to measure discriminating audit, not blanket suspicion.
    """
    planted = set(instance.planted_errors)
    valid_jobs = {j.jid for j in instance.jobs}
    flagged = {f for f in solution.flagged_errors if f in valid_jobs}

    if not planted:
        return (1.0 if not flagged else 0.0), {"verification_note": "no planted errors"}

    tp = len(planted & flagged)
    if tp == 0:
        f1 = 0.0
    else:
        precision = tp / len(flagged)
        recall = tp / len(planted)
        f1 = 2 * precision * recall / (precision + recall)

    return f1, {
        "verification_tp": tp,
        "verification_flagged": len(flagged),
        "verification_planted": len(planted),
    }


def audit_satisfiable(instance: Instance) -> bool:
    """Confirm the ground truth actually scores 1.0.

    Generation is ground-truth-first, so this should always hold; it is asserted
    in tests to catch generator regressions that would silently cap the ceiling
    below 1.0 and corrupt every downstream difficulty calibration.
    """
    perfect = Solution(
        assignment=dict(instance.ground_truth),
        flagged_errors=list(instance.planted_errors),
    )
    return grade(instance, perfect).overall == 1.0
