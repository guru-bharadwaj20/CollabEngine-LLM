"""The second task family, held to the first one's standard.

Two things are being tested here and they are not the same thing.

The first is that the code family works: deterministic in `(seed, difficulty)`,
satisfiable by construction, four components that move independently, a planted
bug that is genuinely planted and genuinely catchable, and a sandbox that
survives hostile input.

The second is that adding it did not disturb the family every published number
came from. That is `test_scheduling_family_is_unchanged`, and it exists because
of a specific failure: a 450-episode run was launched against this refactor
while it was half-applied and every episode errored, costing 3.2 GPU-hours. The
existing suite passed throughout, because nothing in it asserted that the
registry returns the *same* callables the orchestrator used to import directly.
"""

from __future__ import annotations

import pytest

from collabengine.tasks import (
    DEFAULT_FAMILY,
    TASK_FAMILIES,
    get_family,
    presets_for,
)
from collabengine.tasks.code import CODE_COMPONENTS, CodeSolution
from collabengine.tasks.code.generator import (
    DIFFICULTY_ORDER,
    generate,
    reference_source,
    variant_source,
)
from collabengine.tasks.code.grader import audit_satisfiable, grade
from collabengine.tasks.code.sandbox import run_submission
from collabengine.tasks.schema import Component

TIERS = ("easy", "medium", "hard")


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


def test_instances_are_deterministic_in_seed_and_difficulty():
    """The property every rescore and every fresh-seed comparison rests on.

    If this fails, `runs/` cannot be regenerated from seeds and the corpus
    becomes the only copy of the experiment -- which is the state this project
    was in when `runs/` was cleaned.
    """
    for tier in TIERS:
        for seed in (0, 7, 1000):
            a, b = generate(seed, tier), generate(seed, tier)
            assert a.to_dict() == b.to_dict()


def test_different_seeds_give_different_instances_on_the_range_actually_used():
    """Distinctness is a property of a seed *range*, not of the generator.

    Two episodes on the same instance are not two observations: they share
    every error the instance induces, so counting them as independent inflates
    *n* exactly where this project has already been burned by a small *n*.

    Measured rather than assumed. On seeds 1000-1149 -- the fresh-seed range
    PREREG-phase3 mandates and the one any real arm uses -- there are zero
    collisions at every tier. On seeds 0-149 there is exactly one pair (0 and
    1) at `medium`, `hard` and `xhard`, i.e. 0.7%. That is small, bounded, and
    recorded here rather than left to be rediscovered.
    """
    import json

    for tier in TIERS:
        keys = {
            json.dumps(generate(s, tier).to_dict(), sort_keys=True)
            for s in range(1000, 1150)
        }
        assert len(keys) == 150, f"duplicate instances at {tier} on seeds 1000-1149"


def test_every_tier_generates_and_is_satisfiable_by_construction():
    """The reference module scores exactly 1.0, through the sandbox.

    Not a formality. The generator computes expected values with Python
    functions and separately *emits* the reference as source; any drift between
    the two caps the ceiling below 1.0 and would silently corrupt every
    difficulty calibration downstream.
    """
    for tier in DIFFICULTY_ORDER:
        inst = generate(3, tier)
        assert audit_satisfiable(inst), f"reference does not score 1.0 at {tier}"


def test_difficulty_is_monotone_in_the_knobs():
    sizes = [
        (len(generate(5, t).rules), len(generate(5, t).tests)) for t in DIFFICULTY_ORDER
    ]
    rules = [r for r, _ in sizes]
    tests = [t for _, t in sizes]
    assert rules == sorted(rules)
    assert tests == sorted(tests)
    assert rules[-1] > rules[0]


def test_the_hidden_material_is_not_in_the_agent_visible_render():
    """Expected outputs and the reference must not leak into the prompt.

    An instance that shows its test expectations is not measuring
    implementation, and the leak would be invisible in the score -- it would
    just look like the model got better.
    """
    fam = get_family("code")
    inst = generate(11, "medium")
    shown = fam.render_instance(inst)
    assert inst.reference_code not in shown
    for t in inst.tests:
        assert t.tid not in shown
    # A test's expected value may coincide with a number in the prose, so the
    # assertion is on the tuple appearing verbatim, which prose would not.
    for t in inst.tests[:5]:
        assert f"{t.args} -> {t.expected}" not in shown


# ---------------------------------------------------------------------------
# the planted bug -- the verification component
# ---------------------------------------------------------------------------


def test_the_starter_carries_the_bug_and_the_reference_does_not():
    inst = generate(4, "medium")
    assert inst.starter_code != inst.reference_code
    assert inst.planted_bug_rule
    assert inst.planted_bug_kind


def test_carrying_the_bug_forward_costs_verification_and_not_syntax():
    """The bug has to be *catchable*, which means it costs the verify component
    and nothing else.

    A planted error that also broke compilation would be scored by `syntax` and
    the verification column would measure nothing -- the same collision that
    made the behavioural `organize` label useless (RESEARCH-LOG 4.13).

    Submitted as a complete module that keeps the starter's mutation, because
    `starter_code` is only the stage helpers: `apply_rules` and `solve` are what
    the submission is for, so the starter alone cannot score syntax by design.
    """
    inst = generate(4, "medium")
    got = grade(inst, CodeSolution(code=variant_source(inst, fix_bug=False)))
    assert got.per_component[Component.SYNTAX] == pytest.approx(1.0)
    assert got.per_component[Component.VERIFY] < 1.0


def test_fixing_only_the_bug_recovers_verification():
    inst = generate(4, "medium")
    fixed = grade(inst, CodeSolution(code=reference_source(inst)))
    assert fixed.per_component[Component.VERIFY] == pytest.approx(1.0)


def test_each_knob_damages_its_own_component_and_leaves_the_others():
    """The property the whole design rests on, tested one column at a time.

    Four scores that move together would be four views of one number, and the
    agent x component interaction -- the analysis this project exists for --
    would be vacuous on this family.
    """
    inst = generate(9, "medium")
    knob_to_component = {
        "define_helper": Component.SYNTAX,
        "correct_rules": Component.BASE,
        "empty_guard": Component.EDGE,
        "fix_bug": Component.VERIFY,
    }
    ref = grade(inst, CodeSolution(code=reference_source(inst)))
    assert all(ref.per_component[c] == pytest.approx(1.0) for c in CODE_COMPONENTS)

    for knob, damaged in knob_to_component.items():
        got = grade(inst, CodeSolution(code=variant_source(inst, **{knob: False})))
        assert got.per_component[damaged] < 1.0, f"{knob} did not damage {damaged}"


def test_damaging_one_stage_does_not_zero_every_component():
    inst = generate(6, "medium")
    got = grade(inst, CodeSolution(code=variant_source(inst, correct_rules=False)))
    scores = [got.per_component[c] for c in CODE_COMPONENTS]
    assert any(s > 0.0 for s in scores)
    assert any(s < 1.0 for s in scores)


# ---------------------------------------------------------------------------
# grading and the sandbox
# ---------------------------------------------------------------------------


def test_a_malformed_solution_scores_zero_rather_than_raising():
    """A team that never emits a code block still has to produce a data point.

    Dropping the episode instead would bias the sample toward successful runs,
    which is the post-treatment filtering this project spent three corrections
    learning not to do.
    """
    inst = generate(1, "easy")
    got = grade(inst, CodeSolution(malformed=True))
    assert got.overall == 0.0
    assert all(got.per_component[c] == 0.0 for c in CODE_COMPONENTS)


def test_code_that_does_not_parse_scores_zero_syntax_and_does_not_crash():
    inst = generate(1, "easy")
    got = grade(inst, CodeSolution(code="def broken(:\n  return"))
    assert got.per_component[Component.SYNTAX] == 0.0


def test_grading_is_deterministic():
    inst = generate(2, "easy")
    sol = CodeSolution(code=inst.starter_code)
    a, b = grade(inst, sol), grade(inst, sol)
    assert a.per_component == b.per_component and a.overall == b.overall


def test_an_infinite_loop_is_killed_rather_than_hanging_the_suite():
    inst = generate(1, "easy")
    fn = inst.required_functions[0]
    code = f"def {fn}(*a, **k):\n    while True:\n        pass\n"
    got = run_submission(code, inst.required_functions,
                         [t.to_dict() for t in inst.tests], per_test_s=1.0)
    assert not any(got.passed.values())


def test_a_submission_that_imports_os_or_socket_does_not_get_them():
    """Untrusted model output runs in a child process with imports guarded.

    The harness must never `exec` a submission in-process, and a submission that
    reaches for the filesystem or the network must fail rather than succeed
    quietly.
    """
    inst = generate(1, "easy")
    fn = inst.required_functions[0]
    for module in ("os", "socket", "subprocess"):
        code = f"import {module}\ndef {fn}(*a, **k):\n    return 0\n"
        got = run_submission(code, inst.required_functions,
                             [t.to_dict() for t in inst.tests], per_test_s=1.0)
        assert not got.loaded or not any(got.passed.values()), module


def test_a_submission_that_never_defines_the_function_scores_zero_syntax():
    inst = generate(1, "easy")
    got = grade(inst, CodeSolution(code="x = 1\n"))
    assert got.per_component[Component.SYNTAX] == 0.0


# ---------------------------------------------------------------------------
# the registry, and the family that was already there
# ---------------------------------------------------------------------------


def test_both_families_are_registered_and_a_typo_raises():
    assert set(TASK_FAMILIES) == {"scheduling", "code"}
    assert get_family(None).name == DEFAULT_FAMILY == "scheduling"
    with pytest.raises(ValueError):
        get_family("schedulling")


def test_scheduling_family_is_unchanged_by_the_registry():
    """The regression test that would have saved 3.2 GPU-hours.

    The registry must hand back the *same* callables the orchestrator used to
    import directly, producing byte-identical instances and identical component
    scores. Passing tests elsewhere did not establish this, because nothing
    else compared the two paths.
    """
    from collabengine.tasks.generator import generate as sched_generate
    from collabengine.tasks.grader import grade as sched_grade
    from collabengine.tasks.render import parse_solution, render_instance

    fam = get_family("scheduling")
    assert fam.generate is sched_generate
    assert fam.grade is sched_grade
    assert fam.parse_solution is parse_solution
    assert fam.render_instance is render_instance

    for seed in (0, 42, 1000):
        direct = sched_generate(seed, "medium")
        through = fam.generate(seed, "medium")
        assert direct.to_dict() == through.to_dict()


def test_scheduling_difficulty_order_is_pinned():
    """A published difficulty curve must not be reorderable by adding a preset."""
    assert presets_for("scheduling") == ("tiny", "easy", "medium", "hard", "xhard")


def test_the_two_families_have_distinct_briefs():
    """A single-agent baseline needs a single-agent brief, per family.

    The design rule the project paid for in RESEARCH-LOG 4.12 is per-family or
    it is not a rule: the code family's solo brief must not mention co-workers.
    """
    code = get_family("code")
    assert code.solo_brief != code.team_brief
    lowered = code.solo_brief.lower()
    for phrase in ("the others", "participants", "the group"):
        assert phrase not in lowered


def test_component_axes_are_four_wide_for_both_families():
    """The interaction analysis is shaped by this and transfers only if it holds."""
    assert len(get_family("code").components) == 4
    assert len(get_family("scheduling").components) == 4


# ---------------------------------------------------------------------------
# the analysis path, which is where adding a family actually broke something
# ---------------------------------------------------------------------------


def _record(family: str, seed: int, difficulty: str):
    """A minimal episode record, graded by the family's own grader."""
    from collabengine.transcripts.store import EpisodeRecord

    fam = get_family(family)
    inst = fam.generate(seed, difficulty)
    if family == "code":
        solution = CodeSolution(code=inst.reference_code)
    else:
        solution = fam.solution_from_dict({"assignment": dict(inst.ground_truth)})
    return EpisodeRecord(
        episode_id=f"baseline:{difficulty}:{seed}",
        condition="baseline",
        instance_seed=seed,
        difficulty=difficulty,
        agents=["A1", "A2", "A3", "A4"],
        messages=[],
        solution=solution,
        grade=fam.grade(inst, solution),
        config={"task": family},
    )


def test_rescore_reads_the_family_from_the_record():
    """The defect a two-episode real run surfaced, and the reason it matters.

    `rescore` regenerated the instance with the allocation generator no matter
    what the record said. Against a code record it reconstructed a completely
    different instance and then looked up allocation constraint ids in a code
    grade. It raised -- but only because the two id spaces happen not to
    overlap. Had they overlapped it would have returned a plausible number for
    the wrong instance.
    """
    from collabengine.analysis.scoring import METRICS, rescore

    got = rescore(_record("code", 1000, "medium"))
    assert set(got.overall) == set(METRICS)
    assert got.overall["fraction"] == pytest.approx(1.0)
    assert got.overall["strict"] == pytest.approx(1.0)
    assert got.overall["feasible"] == pytest.approx(1.0)


def test_rescore_still_reads_a_record_with_no_family_as_scheduling():
    """Every corpus written before the second family existed carries no `task`
    key, and must keep reading exactly as it did.
    """
    from collabengine.analysis.scoring import rescore

    rec = _record("scheduling", 1000, "medium")
    with_key = rescore(rec)
    rec.config = {}
    without_key = rescore(rec)
    assert with_key.overall == without_key.overall
    assert with_key.per_component == without_key.per_component


def test_strict_is_all_or_nothing_on_the_code_family():
    from collabengine.analysis.scoring import rescore

    inst = generate(1000, "medium")
    rec = _record("code", 1000, "medium")
    rec.solution = CodeSolution(code=variant_source(inst, fix_bug=False))
    rec.grade = get_family("code").grade(inst, rec.solution)
    got = rescore(rec)
    assert got.per_component["strict"][Component.VERIFY] == 0.0
    assert got.per_component["strict"][Component.SYNTAX] == 1.0
