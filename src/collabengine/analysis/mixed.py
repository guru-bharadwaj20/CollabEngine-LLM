"""Mixed-effects test of the agent x component interaction.

docs/PLAN.md Phase 3 names this as the primary analysis: *"mixed-effects model with
episode as a random effect. The specialization claim lives or dies on this
term, not on the main effect."*

`interaction.py` computes the same quantity by double-centering, and that
remains the reported effect size -- it is transparent, it is what the tests
assert on, and it needs no distributional assumptions. What it cannot do is say
whether the interaction is distinguishable from noise, because it collapses
every episode into one cell mean and throws away the variance. This module adds
that, and only that.

**Why episode has to be a random effect.** The same instance is played by the
un-ablated team and by every ablated variant, so those observations are not
independent -- an unusually hard instance drags all of its rows down together.
Treating them as independent would shrink the standard errors and manufacture
significance. A per-episode random intercept absorbs that shared difficulty and
leaves the within-episode contrast, which is the comparison the design actually
makes.

pandas and statsmodels are optional (`pip install -e ".[analysis]"`), so this
module is imported lazily and reports a clear message when they are absent
rather than breaking the rest of the analysis stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from collabengine.tasks.schema import Component
from collabengine.transcripts.store import EpisodeRecord


@dataclass(slots=True)
class MixedEffectsReport:
    """The interaction term, with an episode-level random intercept."""

    n_observations: int
    n_episodes: int
    n_agents: int
    interaction_p: float | None
    """Joint p-value for all agent x component terms. None when not estimable."""

    interaction_chi2: float | None = None
    group_variance: float | None = None
    """Variance absorbed by the per-episode intercept.

    Large relative to the residual means instances differ a lot in difficulty --
    which is precisely the confound the random effect exists to remove, and a
    direct argument against the fixed-effects version of this model."""

    converged: bool = True
    note: str = ""
    terms: dict[str, float] = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        return self.interaction_p is not None and self.interaction_p < 0.05

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "n_episodes": self.n_episodes,
            "n_agents": self.n_agents,
            "interaction_p": self.interaction_p,
            "interaction_chi2": self.interaction_chi2,
            "group_variance": self.group_variance,
            "converged": self.converged,
            "significant": self.significant,
            "note": self.note,
        }


def build_frame(records: Iterable[EpisodeRecord]):
    """Long-form observations: one row per (episode, ablated agent, component).

    Only live-ablation records are used. The frozen modes are derived from a
    recorded transcript rather than independently sampled, so their rows would
    be duplicates of the baseline's randomness rather than new observations, and
    including them would inflate n without adding information.
    """
    import pandas as pd

    rows = []
    for record in records:
        if not record.condition.startswith("live:"):
            continue
        agent = record.condition.split(":", 1)[1]
        for component, score in record.grade.per_component.items():
            rows.append(
                {
                    "episode": record.instance_seed,
                    "agent": agent,
                    "component": component.value
                    if isinstance(component, Component)
                    else str(component),
                    "score": float(score),
                }
            )
    return pd.DataFrame(rows)


def fit_interaction(records: Iterable[EpisodeRecord]) -> MixedEffectsReport:
    """Fit `score ~ agent * component + (1 | episode)` and test the interaction.

    The reported p-value is a joint Wald test over every interaction
    coefficient, not a scan for the largest one. Testing terms individually and
    reporting the smallest would be a multiple-comparisons error dressed up as a
    finding -- with four agents and four components there are nine interaction
    coefficients, so one below 0.05 is the expected outcome under a true null.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return MixedEffectsReport(
            n_observations=0,
            n_episodes=0,
            n_agents=0,
            interaction_p=None,
            converged=False,
            note='statsmodels not installed; pip install -e ".[analysis]"',
        )

    frame = build_frame(records)
    if frame.empty:
        return MixedEffectsReport(
            n_observations=0,
            n_episodes=0,
            n_agents=0,
            interaction_p=None,
            converged=False,
            note="no live-ablation records to fit",
        )

    n_agents = frame["agent"].nunique()
    n_episodes = frame["episode"].nunique()
    if n_agents < 2 or frame["component"].nunique() < 2:
        return MixedEffectsReport(
            n_observations=len(frame),
            n_episodes=n_episodes,
            n_agents=n_agents,
            interaction_p=None,
            converged=False,
            note="an interaction needs at least two agents and two components",
        )

    import warnings

    model = smf.mixedlm(
        "score ~ C(agent) * C(component)", frame, groups=frame["episode"]
    )
    with warnings.catch_warnings():
        # Convergence warnings are reported through `converged` rather than
        # printed, so a caller cannot miss them in a wall of run output.
        warnings.simplefilter("ignore")
        try:
            result = model.fit(reml=False)
        except Exception as exc:  # noqa: BLE001 - a failed fit is a result
            return MixedEffectsReport(
                n_observations=len(frame),
                n_episodes=n_episodes,
                n_agents=n_agents,
                interaction_p=None,
                converged=False,
                note=f"fit failed: {type(exc).__name__}: {exc}",
            )

    interaction_terms = [n for n in result.params.index if ":" in n]
    if not interaction_terms:
        return MixedEffectsReport(
            n_observations=len(frame),
            n_episodes=n_episodes,
            n_agents=n_agents,
            interaction_p=None,
            converged=bool(getattr(result, "converged", True)),
            note="no interaction terms were estimable",
        )

    chi2 = p_value = None
    try:
        test = result.wald_test(
            _contrast(result.params.index, interaction_terms), scalar=True
        )
        chi2 = float(test.statistic)
        p_value = float(test.pvalue)
    except Exception as exc:  # noqa: BLE001
        return MixedEffectsReport(
            n_observations=len(frame),
            n_episodes=n_episodes,
            n_agents=n_agents,
            interaction_p=None,
            converged=bool(getattr(result, "converged", True)),
            note=f"Wald test failed: {type(exc).__name__}: {exc}",
        )

    return MixedEffectsReport(
        n_observations=len(frame),
        n_episodes=n_episodes,
        n_agents=n_agents,
        interaction_p=p_value,
        interaction_chi2=chi2,
        group_variance=float(result.cov_re.iloc[0, 0]) if result.cov_re.size else None,
        converged=bool(getattr(result, "converged", True)),
        terms={n: float(result.params[n]) for n in interaction_terms},
    )


def _contrast(all_terms, tested_terms):
    """One row per tested coefficient, selecting it and nothing else."""
    import numpy as np

    index = list(all_terms)
    matrix = np.zeros((len(tested_terms), len(index)))
    for row, name in enumerate(tested_terms):
        matrix[row, index.index(name)] = 1.0
    return matrix
