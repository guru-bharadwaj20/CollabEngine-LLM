"""Stricter readings of a graded episode.

These exist because the default fraction metric sits near ceiling at this model
scale, and a component with no room to fall cannot show an ablation drop. The
risk in adding metrics is that one of them quietly becomes the headline because
it produces a bigger number, so the tests pin down what each one actually
claims.
"""

from __future__ import annotations

from collabengine.analysis.scoring import (
    FEASIBLE,
    FRACTION,
    METRICS,
    STRICT,
    is_feasible,
    rescore,
    strict_components,
)
from collabengine.tasks.generator import generate
from collabengine.tasks.grader import grade
from collabengine.tasks.schema import ALL_COMPONENTS, Component, Solution
from collabengine.transcripts.store import EpisodeRecord


def _record(seed: int = 3, difficulty: str = "tiny") -> EpisodeRecord:
    """An episode whose answer is the instance's own ground truth."""
    instance = generate(seed, difficulty)
    solution = Solution(
        assignment=dict(instance.ground_truth),
        flagged_errors=list(instance.planted_errors),
    )
    return EpisodeRecord(
        episode_id="e1",
        condition="baseline",
        instance_seed=seed,
        difficulty=difficulty,
        agents=["A1"],
        messages=[],
        solution=solution,
        grade=grade(instance, solution),
    )


def test_a_perfect_answer_scores_one_under_every_metric() -> None:
    scored = rescore(_record())
    for metric in METRICS:
        assert scored.overall[metric] == 1.0, metric


def test_strict_is_all_or_nothing_where_fraction_gives_partial_credit() -> None:
    """One violated capacity limit does not leave a schedule 90% feasible.

    Uses `medium`, which has enough constraints per class that a single broken
    one leaves the fraction strictly between 0 and 1. On `tiny` a class can hold
    one constraint, so fraction collapses to 0 too and the distinction the
    metric exists to draw is invisible.
    """
    instance = generate(3, "medium")
    truth = dict(instance.ground_truth)
    victim = sorted(truth)[0]
    others = [w.wid for w in instance.workers if w.wid != truth[victim]]
    truth[victim] = others[0]

    solution = Solution(assignment=truth, flagged_errors=list(instance.planted_errors))
    g = grade(instance, solution)
    strict = strict_components(instance, g)

    scored = [c for c in ALL_COMPONENTS if c is not Component.VERIFICATION]
    # Strict can never flatter fraction.
    for comp in scored:
        assert strict[comp] <= g.per_component[comp] + 1e-9, comp

    partial = [c for c in scored if 0.0 < g.per_component[c] < 1.0]
    assert partial, "expected the broken job to leave some class partly satisfied"
    for comp in partial:
        assert strict[comp] == 0.0, comp
    assert not is_feasible(g)


def test_verification_passes_through_strict_unchanged() -> None:
    """It is already errors-caught over errors-planted, not a long-list fraction.

    Collapsing it to all-or-nothing would throw away the only component whose
    scale is already small enough to have usable range.
    """
    instance = generate(5, "tiny")
    solution = Solution(assignment=dict(instance.ground_truth), flagged_errors=[])
    g = grade(instance, solution)

    strict = strict_components(instance, g)
    assert strict[Component.VERIFICATION] == g.per_component[Component.VERIFICATION]


def test_feasible_is_flat_across_components_by_construction() -> None:
    """It is a property of the whole answer, so it can carry no interaction.

    Pinned deliberately: an agent x component interaction computed on this
    metric must come out at zero, and a non-zero one would mean a bug rather
    than a discovery.
    """
    scored = rescore(_record())
    values = set(scored.per_component[FEASIBLE].values())
    assert len(values) == 1


def test_a_malformed_answer_is_zero_everywhere_not_feasible() -> None:
    instance = generate(3, "tiny")
    g = grade(instance, Solution(malformed=True))

    assert not is_feasible(g)
    assert all(v == 0.0 for v in strict_components(instance, g).values())


def test_rescore_regenerates_the_instance_from_seed_and_difficulty() -> None:
    """The property that makes re-analysis free -- no stored instance, no GPU."""
    a = rescore(_record(seed=7))
    b = rescore(_record(seed=7))
    assert a.overall == b.overall

    other = rescore(_record(seed=8))
    assert other.overall[FRACTION] == 1.0     # still ground truth, different instance
    assert a.per_component[STRICT].keys() == set(ALL_COMPONENTS)
