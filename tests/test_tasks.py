"""Generator and grader invariants.

The satisfiability audit is the load-bearing test in this file: if the ground
truth stops scoring 1.0, every difficulty calibration downstream is measuring a
broken ceiling rather than model capability.
"""

from __future__ import annotations

import pytest

from collabengine.tasks.generator import PRESETS, generate
from collabengine.tasks.grader import audit_satisfiable, grade
from collabengine.tasks.schema import ALL_COMPONENTS, Component, Instance, Solution

DIFFICULTIES = sorted(PRESETS)
SEEDS = list(range(25))


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
@pytest.mark.parametrize("seed", SEEDS)
def test_ground_truth_scores_perfect(difficulty: str, seed: int) -> None:
    assert audit_satisfiable(generate(seed, difficulty))


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_generation_is_deterministic(difficulty: str) -> None:
    a = generate(7, difficulty)
    b = generate(7, difficulty)
    assert a.to_dict() == b.to_dict()


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_instance_round_trips_through_json(difficulty: str) -> None:
    inst = generate(11, difficulty)
    assert Instance.from_dict(inst.to_dict()).to_dict() == inst.to_dict()


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_every_component_is_represented(difficulty: str) -> None:
    inst = generate(3, difficulty)
    for comp in ALL_COMPONENTS:
        if comp is Component.VERIFICATION:
            assert inst.planted_errors, "verification needs planted errors"
        else:
            assert inst.constraints_for(comp), f"no constraints for {comp}"


def test_empty_solution_scores_zero_ish() -> None:
    inst = generate(1, "medium")
    result = grade(inst, Solution())
    assert result.overall < 0.5
    assert result.per_component[Component.VERIFICATION] == 0.0


def test_malformed_solution_scores_zero_not_raises() -> None:
    inst = generate(1, "medium")
    result = grade(inst, Solution(malformed=True))
    assert result.overall == 0.0
    assert all(v == 0.0 for v in result.per_component.values())


def test_hallucinated_ids_are_dropped_not_fatal() -> None:
    inst = generate(2, "easy")
    poisoned = dict(inst.ground_truth)
    poisoned["J999"] = "W1"
    poisoned[inst.jobs[0].jid] = "W_NONEXISTENT"

    result = grade(inst, Solution(assignment=poisoned, flagged_errors=list(inst.planted_errors)))
    assert result.detail["dropped_invalid"] == 2
    # The job whose worker was invalid now fails its skill constraint.
    assert result.per_component[Component.SEARCH] < 1.0


def test_flagging_everything_does_not_win_verification() -> None:
    """F1, not accuracy -- blanket suspicion must score worse than precision."""
    inst = generate(5, "medium")
    all_jobs = [j.jid for j in inst.jobs]

    blanket = grade(inst, Solution(assignment=dict(inst.ground_truth), flagged_errors=all_jobs))
    precise = grade(
        inst,
        Solution(assignment=dict(inst.ground_truth), flagged_errors=list(inst.planted_errors)),
    )
    assert blanket.per_component[Component.VERIFICATION] < precise.per_component[Component.VERIFICATION]


def test_capacity_slack_makes_arithmetic_bind() -> None:
    """Overloading one worker must break the arithmetic component specifically."""
    inst = generate(4, "hard")
    piled = {j.jid: inst.workers[0].wid for j in inst.jobs}
    result = grade(inst, Solution(assignment=piled, flagged_errors=[]))
    assert result.per_component[Component.ARITHMETIC] < 1.0


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_planted_errors_are_objectively_detectable(difficulty: str) -> None:
    """Each planted error must violate a skill constraint visible in the instance.

    Otherwise the team would have to guess our hidden ground truth rather than
    reason from what it was given.
    """
    inst = generate(9, difficulty)
    for jid in inst.planted_errors:
        job = inst.job(jid)
        worker = inst.worker(inst.proposed_assignment[jid])
        assert job is not None and worker is not None
        assert job.skill not in worker.skills
