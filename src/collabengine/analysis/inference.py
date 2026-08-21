"""Multiplicity correction, equivalence testing, and power — in one place.

Four gaps in this project's statistics are the same gap: it reports *p*-values
from a family of hypotheses as though each stood alone, and it makes null claims
with an instrument built to reject nulls.

**Why multiplicity matters here.** `PREREG-xhard` registers a gate at three
tiers on three metrics; `PREREG-phase3` registers H1, H1e, H2, H3, H3b, H4, H5.
Each was read at alpha = 0.05. Under a true null, seven independent tests at
0.05 produce at least one "significant" result 30% of the time. Every headline
in this paper is a null, so correction cannot manufacture a finding here — which
is exactly why omitting it costs nothing to fix and looks like an oversight.

**Why equivalence matters more.** Failing to reject H0 is not evidence for H0.
The project's central claims — team size does nothing, the interaction is flat,
fungibility is zero — are all null claims, and a wide interval that happens to
contain zero supports none of them. TOST is the test built for this: it asks
whether the effect is small enough to *exclude* effects larger than some margin
`delta`, and it can fail, which is what makes it evidence.

`smallest_equivalence_bound` is the number worth reporting. It converts "we
could not detect an effect" into "we exclude effects larger than X", which is
the strongest form a negative result can take and does not require anyone to
agree with our choice of `delta` in advance.

**Why power is here and post-hoc power is not.** Post-hoc power is a
deterministic function of the observed *p*-value and carries no information
beyond it. What the prereg postscripts were reaching for when they wrote it is
either the a-priori MDE (`mde`, computed before the run from a prior `sd`) or
the equivalence bound (above). Both are in this module; observed power is not,
deliberately.

`scipy` is imported lazily and the error says so, matching `analysis.mixed`: a
laptop checkout that only reads transcripts should not need it.
"""

from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

__all__ = [
    "holm",
    "bh_fdr",
    "adjust",
    "TostResult",
    "tost",
    "smallest_equivalence_bound",
    "welch",
    "cohens_d",
    "mde",
    "n_for",
    "PowerPlan",
    "plan",
]


def _scipy_stats():
    try:
        from scipy import stats
    except ImportError as exc:  # pragma: no cover - exercised by hand, not CI
        raise ImportError(
            "inference needs scipy: pip install -e '.[analysis]'"
        ) from exc
    return stats


# ---------------------------------------------------------------------------
# multiplicity
# ---------------------------------------------------------------------------


def _ordered(pvalues: Mapping[str, float]) -> list[tuple[str, float]]:
    for k, p in pvalues.items():
        if not (0.0 <= p <= 1.0):
            raise ValueError(f"p-value out of range for {k!r}: {p}")
    return sorted(pvalues.items(), key=lambda kv: kv[1])


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, controlling family-wise error.

    Holm is the default rather than Bonferroni because it is uniformly more
    powerful and requires no extra assumptions, and rather than BH because the
    question a reviewer asks about "we ran H1-H5 at 0.05 each" is a family-wise
    one: *did any of these fire by chance?*

    The step-down enforces monotonicity — an adjusted value never falls below
    one earlier in the ordering — so the returned numbers can be sorted and read
    directly.

    The keys are hypothesis labels and they matter: a family assembled after
    seeing which tests were significant is not a correction. Declare the family
    from the preregistration, then call this.
    """
    items = _ordered(pvalues)
    m = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[key] = running
    return {k: out[k] for k in pvalues}


def bh_fdr(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Benjamini-Hochberg adjusted p-values, controlling false-discovery rate.

    Reported *alongside* Holm, never instead of it. BH answers a different and
    more permissive question — what fraction of the things I call significant
    are wrong — which is the right question for a screen and the wrong one for a
    preregistered confirmatory family. Both are printed so a reader can see that
    the conclusion does not depend on which is chosen.
    """
    items = _ordered(pvalues)
    m = len(items)
    out: dict[str, float] = {}
    running = 1.0
    for i in range(m - 1, -1, -1):
        key, p = items[i]
        adj = min(1.0, p * m / (i + 1))
        running = min(running, adj)
        out[key] = running
    return {k: out[k] for k in pvalues}


def adjust(pvalues: Mapping[str, float]) -> dict[str, dict[str, float]]:
    """Both corrections plus the raw value, keyed by hypothesis label.

    Returns `{label: {"raw": p, "holm": p, "bh": p}}` — the shape the report
    tables want, so no call site has to zip three dicts together and get the
    ordering wrong.
    """
    h, b = holm(pvalues), bh_fdr(pvalues)
    return {k: {"raw": p, "holm": h[k], "bh": b[k]} for k, p in pvalues.items()}


# ---------------------------------------------------------------------------
# equivalence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TostResult:
    """The outcome of two one-sided tests at margin `delta`.

    `p` is the TOST p-value — the *larger* of the two one-sided p-values,
    because equivalence requires both bounds to be excluded. `equivalent` is
    that value against alpha.
    """

    diff: float
    delta: float
    p_lower: float
    p_upper: float
    p: float
    ci_low: float
    ci_high: float
    df: float
    alpha: float

    @property
    def equivalent(self) -> bool:
        return self.p < self.alpha

    def line(self) -> str:
        verdict = "equivalent" if self.equivalent else "NOT equivalent"
        return (
            f"diff {self.diff:+.3f}  "
            f"{100 * (1 - 2 * self.alpha):.0f}% CI [{self.ci_low:+.3f}, {self.ci_high:+.3f}]  "
            f"delta {self.delta:.3f}  TOST p = {self.p:.3f}  {verdict}"
        )


def welch(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    """(difference in means, standard error, Welch degrees of freedom).

    Welch rather than pooled because the arms in this project routinely differ
    in spread — solo's variance at `medium` was 0.281 against the team's 0.107
    before the cap was controlled (README) — and equal-variance assumptions are
    exactly what that disagreement violates.
    """
    if len(a) < 2 or len(b) < 2:
        raise ValueError("need at least two observations per arm")
    na, nb = len(a), len(b)
    va, vb = st.variance(a), st.variance(b)
    se2 = va / na + vb / nb
    if se2 <= 0.0:
        # Both arms constant. The difference is exact; report it with a zero
        # standard error rather than dividing by it at the call site.
        return st.mean(b) - st.mean(a), 0.0, float(na + nb - 2)
    df = se2**2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return st.mean(b) - st.mean(a), math.sqrt(se2), df


def tost(
    a: Sequence[float],
    b: Sequence[float],
    delta: float,
    alpha: float = 0.05,
) -> TostResult:
    """Two one-sided tests for equivalence of mean(b) - mean(a) within +/- delta.

    **Choose `delta` from the task, before looking at the data.** Choosing it
    after seeing the interval is the same failure as choosing a hypothesis after
    seeing the corpus, and this project has a rule about that (the fresh-seed
    rule, LOG 4.22). If you cannot defend a margin in advance, report
    `smallest_equivalence_bound` instead — it makes no such choice.

    Equivalent at level `alpha` iff the (1 - 2*alpha) confidence interval on the
    difference lies entirely inside (-delta, +delta). The interval is returned
    so a reader can check that directly rather than trusting the verdict.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    stats = _scipy_stats()
    diff, se, df = welch(a, b)

    if se == 0.0:
        p = 0.0 if abs(diff) < delta else 1.0
        return TostResult(diff, delta, p, p, p, diff, diff, df, alpha)

    t_lower = (diff + delta) / se     # H0: diff <= -delta
    t_upper = (diff - delta) / se     # H0: diff >= +delta
    p_lower = float(stats.t.sf(t_lower, df))
    p_upper = float(stats.t.cdf(t_upper, df))
    crit = float(stats.t.ppf(1 - alpha, df))
    return TostResult(
        diff=diff,
        delta=delta,
        p_lower=p_lower,
        p_upper=p_upper,
        p=max(p_lower, p_upper),
        ci_low=diff - crit * se,
        ci_high=diff + crit * se,
        df=df,
        alpha=alpha,
    )


def smallest_equivalence_bound(
    a: Sequence[float], b: Sequence[float], alpha: float = 0.05
) -> float:
    """The smallest `delta` at which these arms are equivalent at `alpha`.

    This is the number to put in the paper. It requires no prior agreement on a
    margin, it cannot be gamed after the fact, and it says the useful thing:
    *effects larger than this are excluded by the data.* A null result reported
    this way is a bound; reported as "p = 0.87" it is an absence of evidence.

    It is the half-width of the (1 - 2*alpha) interval measured from zero:
    `|diff| + t_{1-alpha, df} * se`.
    """
    stats = _scipy_stats()
    diff, se, df = welch(a, b)
    if se == 0.0:
        return abs(diff)
    return abs(diff) + float(stats.t.ppf(1 - alpha, df)) * se


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    """Standardised mean difference, pooled sd. Sign follows mean(b) - mean(a)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        raise ValueError("need at least two observations per arm")
    pooled2 = ((na - 1) * st.variance(a) + (nb - 1) * st.variance(b)) / (na + nb - 2)
    if pooled2 <= 0:
        return 0.0
    return (st.mean(b) - st.mean(a)) / math.sqrt(pooled2)


# ---------------------------------------------------------------------------
# power, computed before the run
# ---------------------------------------------------------------------------


def mde(n_per_arm: int, sd: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable effect for a planned two-arm comparison.

    Normal approximation, equal arms. Deliberately not exact: it is used to size
    a run from a *prior* `sd`, and the prior is the dominant error term. Quoting
    a non-central t to four decimals on top of an `sd` guessed from a pilot is
    false precision.

    The number this project most needed and never computed: at `sd` = 0.15 and
    n = 48 per arm, the MDE is ~0.086 — nearly twice the +0.055 participation
    effect that was read off a 48-episode baseline and then failed to replicate
    (LOG 4.22).
    """
    if n_per_arm < 2:
        raise ValueError("need at least two observations per arm")
    if sd <= 0:
        raise ValueError("sd must be positive")
    stats = _scipy_stats()
    z_a = float(stats.norm.ppf(1 - alpha / 2))
    z_b = float(stats.norm.ppf(power))
    return (z_a + z_b) * sd * math.sqrt(2.0 / n_per_arm)


def n_for(delta: float, sd: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Episodes per arm needed to detect `delta` at `power`. The inverse of `mde`.

    Round-tripping is exact to within the ceiling: `mde(n_for(d, sd), sd) <= d`.
    """
    if delta <= 0:
        raise ValueError("delta must be positive")
    if sd <= 0:
        raise ValueError("sd must be positive")
    stats = _scipy_stats()
    z_a = float(stats.norm.ppf(1 - alpha / 2))
    z_b = float(stats.norm.ppf(power))
    return max(2, math.ceil(2.0 * (sd * (z_a + z_b) / delta) ** 2))


@dataclass(frozen=True)
class PowerPlan:
    """What a planned arm can and cannot see, recorded before it runs."""

    label: str
    n_per_arm: int
    sd_prior: float
    alpha: float
    power: float
    mde: float

    def line(self) -> str:
        return (
            f"{self.label:<24} n = {self.n_per_arm:>4}/arm  "
            f"sd prior {self.sd_prior:.3f}  "
            f"MDE {self.mde:.3f} at {self.power:.0%} power, alpha = {self.alpha}"
        )


def plan(
    label: str,
    n_per_arm: int,
    sd_prior: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> PowerPlan:
    """Build the row that belongs in a preregistration, not in a postscript."""
    return PowerPlan(
        label=label,
        n_per_arm=n_per_arm,
        sd_prior=sd_prior,
        alpha=alpha,
        power=power,
        mde=mde(n_per_arm, sd_prior, alpha, power),
    )


def sd_table(arms: Mapping[str, Iterable[float]]) -> dict[str, tuple[int, float]]:
    """Realised (n, sd) per arm — the prior every future sizing should use.

    Published as a table in its own right. Anyone sizing a study like this one
    currently has to guess `sd`, and guessing it low is how a 48-episode arm
    ends up carrying a headline.
    """
    out: dict[str, tuple[int, float]] = {}
    for name, values in arms.items():
        v = list(values)
        out[name] = (len(v), st.stdev(v) if len(v) > 1 else 0.0)
    return out
