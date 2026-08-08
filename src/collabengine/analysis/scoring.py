"""Alternative gradings of a solution already on disk.

The grader scores each component as the *fraction* of its constraints that hold.
That is the natural choice and it turns out to have very little dynamic range at
this model scale: a 50% increase in instance size moved a single agent's score
from 0.879 to 0.842, because extra constraints enlarge the numerator and the
denominator together. At `hard` the same agent scores 0.842 while producing a
schedule that is actually feasible in one episode out of twelve.

A fraction near ceiling is not merely unflattering, it is unusable for this
study. An ablation drop has to be visible in the component the removed agent
owned, and a component already at 0.92 for a *single* agent has nowhere to fall.

So two stricter readings are offered alongside the original:

* **strict** -- a component scores 1 only if every constraint of its class
  holds. A schedule that violates one capacity limit is not 90% feasible.
* **feasible** -- the whole instance, all or nothing. Closest to what someone
  using such a schedule would care about.

Neither replaces `fraction`. Measured against the medium corpus, the stricter
metrics moved the team-vs-solo gap from +0.026 to +0.167 but moved Cohen's *d*
only from +0.27 to +0.36 -- they change the numbers far more than they change
the separation. Reporting all three is what keeps that distinction visible
instead of letting a choice of metric look like a finding.
"""

from __future__ import annotations

from dataclasses import dataclass

from collabengine.tasks.generator import generate
from collabengine.tasks.grader import GradeResult
from collabengine.tasks.schema import ALL_COMPONENTS, Component, Instance
from collabengine.transcripts.store import EpisodeRecord

FRACTION = "fraction"
STRICT = "strict"
FEASIBLE = "feasible"
METRICS = (FRACTION, STRICT, FEASIBLE)


@dataclass(frozen=True, slots=True)
class Rescored:
    """One episode under every metric. `per_component` drives the interaction."""

    overall: dict[str, float]
    per_component: dict[str, dict[Component, float]]


def strict_components(
    instance: Instance, grade: GradeResult
) -> dict[Component, float]:
    """All-or-nothing per component.

    Verification is passed through unchanged: it is already scored as errors
    caught against errors planted, which is a proportion of a small integer
    rather than a fraction of a long constraint list, and has usable range.
    """
    out: dict[Component, float] = {}
    for comp in ALL_COMPONENTS:
        if comp is Component.VERIFICATION:
            out[comp] = grade.per_component[comp]
            continue
        cs = instance.constraints_for(comp)
        out[comp] = float(all(grade.satisfied[c.cid] for c in cs)) if cs else 1.0
    return out


def is_feasible(grade: GradeResult) -> bool:
    """Whether every constraint in the instance holds."""
    return bool(grade.satisfied) and all(grade.satisfied.values())


def rescore(record: EpisodeRecord) -> Rescored:
    """Score one recorded episode under all three metrics.

    The instance is regenerated from `(instance_seed, difficulty)` rather than
    stored: generation is deterministic in exactly those two, which is the
    property that makes re-analysis free. No model call is involved, so a whole
    corpus can be re-read under a new metric without touching the GPU.
    """
    instance = generate(record.instance_seed, record.difficulty)
    grade = record.grade

    strict = strict_components(instance, grade)
    feasible = float(is_feasible(grade))

    return Rescored(
        overall={
            FRACTION: grade.overall,
            STRICT: sum(strict.values()) / len(strict),
            FEASIBLE: feasible,
        },
        per_component={
            FRACTION: dict(grade.per_component),
            STRICT: strict,
            # Feasibility is a property of the whole answer, so every component
            # carries the same value. Kept in the same shape so callers can loop
            # over metrics uniformly; it contributes no interaction by
            # construction, which is itself worth being able to see.
            FEASIBLE: {c: feasible for c in ALL_COMPONENTS},
        },
    )
