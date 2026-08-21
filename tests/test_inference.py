"""Multiplicity, equivalence and power.

Held to the same standard as the rest of the instrument: each function has to
give the known answer on a case worked by hand, and — the part that matters —
has to *fail* where it should. An equivalence test that always concludes
"equivalent" would launder every null in this project into a finding, which is
the mirror image of the significance test that fires on a null world
(`test_mixed.py`).
"""

from __future__ import annotations

import random

import pytest

from collabengine.analysis.inference import (
    adjust,
    bh_fdr,
    cohens_d,
    holm,
    mde,
    n_for,
    plan,
    sd_table,
    smallest_equivalence_bound,
    tost,
    welch,
)

pytest.importorskip("scipy")


# ---------------------------------------------------------------------------
# multiplicity
# ---------------------------------------------------------------------------


def test_holm_matches_hand_computation():
    # m = 4. Sorted: .01, .02, .03, .04 with multipliers 4, 3, 2, 1.
    # Raw products: .04, .06, .06, .04 -> monotone: .04, .06, .06, .06.
    p = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04}
    got = holm(p)
    assert got["a"] == pytest.approx(0.04)
    assert got["b"] == pytest.approx(0.06)
    assert got["c"] == pytest.approx(0.06)
    assert got["d"] == pytest.approx(0.06)


def test_holm_is_monotone_and_never_below_the_raw_value():
    rng = random.Random(11)
    p = {f"h{i}": rng.random() for i in range(12)}
    got = holm(p)
    for k, raw in p.items():
        assert got[k] >= raw - 1e-12
    ordered = sorted(p, key=lambda k: p[k])
    for earlier, later in zip(ordered, ordered[1:]):
        assert got[later] >= got[earlier] - 1e-12


def test_holm_on_a_single_hypothesis_changes_nothing():
    assert holm({"only": 0.031})["only"] == pytest.approx(0.031)


def test_bh_is_never_more_conservative_than_holm():
    # This is the whole reason both are reported: if the conclusion differed
    # between them, the choice of correction would be doing the work.
    rng = random.Random(3)
    p = {f"h{i}": rng.random() for i in range(20)}
    h, b = holm(p), bh_fdr(p)
    for k in p:
        assert b[k] <= h[k] + 1e-12


def test_bh_matches_hand_computation():
    # m = 3, sorted .01, .04, .05 -> .03, .06, .05 -> step-up min: .03, .05, .05
    got = bh_fdr({"a": 0.01, "b": 0.04, "c": 0.05})
    assert got["a"] == pytest.approx(0.03)
    assert got["b"] == pytest.approx(0.05)
    assert got["c"] == pytest.approx(0.05)


def test_the_phase3_family_at_five_percent_each_is_not_five_percent():
    """The concrete reason this module exists.

    Seven preregistered hypotheses each read at 0.05. A single raw p of 0.03
    survives neither correction once the family is declared.
    """
    family = {
        "H1": 0.03, "H1e": 0.44, "H2": 0.90, "H3": 0.93,
        "H3b": 0.12, "H4": 0.61, "H5": 0.55,
    }
    out = adjust(family)
    assert out["H1"]["raw"] == pytest.approx(0.03)
    assert out["H1"]["holm"] > 0.05
    assert out["H1"]["bh"] > 0.05


def test_adjust_returns_every_label_with_all_three_columns():
    out = adjust({"x": 0.2, "y": 0.4})
    assert set(out) == {"x", "y"}
    assert set(out["x"]) == {"raw", "holm", "bh"}


def test_out_of_range_p_is_rejected():
    with pytest.raises(ValueError):
        holm({"a": 1.5})


# ---------------------------------------------------------------------------
# equivalence
# ---------------------------------------------------------------------------


def _arm(rng: random.Random, n: int, mean: float, sd: float) -> list[float]:
    return [rng.gauss(mean, sd) for _ in range(n)]


def test_tost_finds_equivalence_when_arms_are_identical_and_large():
    rng = random.Random(0)
    a = _arm(rng, 400, 0.575, 0.15)
    b = _arm(rng, 400, 0.575, 0.15)
    r = tost(a, b, delta=0.05)
    assert r.equivalent
    assert r.ci_low > -0.05 and r.ci_high < 0.05


def test_tost_refuses_equivalence_when_the_effect_is_real():
    rng = random.Random(1)
    a = _arm(rng, 400, 0.50, 0.15)
    b = _arm(rng, 400, 0.65, 0.15)
    assert not tost(a, b, delta=0.05).equivalent


def test_tost_refuses_equivalence_when_the_sample_is_too_small_to_say():
    """The failure mode that matters.

    Two arms with the same mean and only 12 episodes each are *not* evidence of
    equivalence at a 0.05 margin. A test that called this equivalent would turn
    every underpowered null in this project into a finding.
    """
    rng = random.Random(2)
    a = _arm(rng, 12, 0.575, 0.15)
    b = _arm(rng, 12, 0.575, 0.15)
    assert not tost(a, b, delta=0.05).equivalent


def test_a_wider_margin_is_easier_to_satisfy():
    rng = random.Random(4)
    a, b = _arm(rng, 60, 0.5, 0.15), _arm(rng, 60, 0.52, 0.15)
    tight, loose = tost(a, b, delta=0.02), tost(a, b, delta=0.25)
    assert loose.p <= tight.p
    assert loose.equivalent


def test_tost_p_is_the_larger_of_the_two_one_sided_tests():
    rng = random.Random(5)
    a, b = _arm(rng, 50, 0.5, 0.1), _arm(rng, 50, 0.53, 0.1)
    r = tost(a, b, delta=0.1)
    assert r.p == pytest.approx(max(r.p_lower, r.p_upper))


def test_smallest_bound_is_exactly_where_the_verdict_flips():
    rng = random.Random(6)
    a, b = _arm(rng, 120, 0.5, 0.12), _arm(rng, 120, 0.51, 0.12)
    d = smallest_equivalence_bound(a, b)
    assert tost(a, b, delta=d * 1.001).equivalent
    assert not tost(a, b, delta=d * 0.999).equivalent


def test_smallest_bound_shrinks_as_n_grows():
    """The bound is what a larger corpus buys, stated as a number.

    This is why the fresh-seed re-run at n = 150 says more than the pilot at
    n = 48 even though both report "no effect".
    """
    rng = random.Random(7)
    small = smallest_equivalence_bound(_arm(rng, 24, 0.5, 0.15), _arm(rng, 24, 0.5, 0.15))
    large = smallest_equivalence_bound(_arm(rng, 600, 0.5, 0.15), _arm(rng, 600, 0.5, 0.15))
    assert large < small


def test_smallest_bound_is_never_below_the_observed_difference():
    rng = random.Random(8)
    a, b = _arm(rng, 40, 0.4, 0.1), _arm(rng, 40, 0.6, 0.1)
    diff, _, _ = welch(a, b)
    assert smallest_equivalence_bound(a, b) >= abs(diff)


def test_welch_handles_unequal_variance_arms():
    """Solo's spread was 0.281 against the team's 0.107. Pooling would lie."""
    rng = random.Random(9)
    a, b = _arm(rng, 50, 0.5, 0.28), _arm(rng, 50, 0.5, 0.10)
    _, se, df = welch(a, b)
    assert se > 0
    assert 50 < df < 98      # Welch discounts, a pooled test would say 98


def test_constant_arms_do_not_divide_by_zero():
    r = tost([0.5] * 10, [0.5] * 10, delta=0.05)
    assert r.equivalent and r.diff == pytest.approx(0.0)
    assert smallest_equivalence_bound([0.5] * 10, [0.5] * 10) == pytest.approx(0.0)


def test_delta_must_be_positive():
    with pytest.raises(ValueError):
        tost([1.0, 2.0], [1.0, 2.0], delta=0.0)


def test_cohens_d_recovers_a_planted_standardised_effect():
    rng = random.Random(10)
    a, b = _arm(rng, 2000, 0.0, 1.0), _arm(rng, 2000, 0.5, 1.0)
    assert cohens_d(a, b) == pytest.approx(0.5, abs=0.08)


# ---------------------------------------------------------------------------
# power
# ---------------------------------------------------------------------------


def test_mde_and_n_for_round_trip():
    for delta in (0.02, 0.05, 0.1):
        n = n_for(delta, sd=0.15)
        assert mde(n, sd=0.15) <= delta + 1e-9


def test_mde_falls_as_n_rises_and_rises_with_sd():
    assert mde(400, 0.15) < mde(48, 0.15)
    assert mde(48, 0.30) > mde(48, 0.15)


def test_the_forty_eight_episode_baseline_could_not_have_seen_its_own_effect():
    """LOG 4.22, as a number computed before the run rather than after it.

    The withdrawn participation effect was +0.055, read off a 48-episode arm
    with sd ~= 0.15. The MDE there is larger than the effect, so the pilot was
    never able to distinguish it from noise — which is what happened.
    """
    assert mde(48, sd=0.15) > 0.055
    assert n_for(0.055, sd=0.15) > 48


def test_plan_row_reports_what_the_arm_can_see():
    row = plan("medium team vs solo", n_per_arm=150, sd_prior=0.15)
    assert row.mde == pytest.approx(mde(150, 0.15))
    assert "MDE" in row.line() and "150" in row.line()


def test_sd_table_reports_n_and_spread_per_arm():
    got = sd_table({"solo": [0.1, 0.2, 0.3], "team": [0.5, 0.5, 0.5]})
    assert got["solo"][0] == 3
    assert got["solo"][1] == pytest.approx(0.1)
    assert got["team"][1] == pytest.approx(0.0)


def test_degenerate_inputs_are_rejected_rather_than_guessed():
    with pytest.raises(ValueError):
        mde(1, 0.15)
    with pytest.raises(ValueError):
        n_for(0.05, 0.0)
    with pytest.raises(ValueError):
        welch([1.0], [1.0, 2.0])
