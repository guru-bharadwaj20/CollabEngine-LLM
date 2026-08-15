"""Phase 4: do transcript labels predict causal contribution?

This is the question the project exists to answer, and the claim in docs/PLAN.md 7
turns on it:

> Emergent role differentiation ... is behaviorally observable and statistically
> stable -- but its transcript-derived labels predict causal contribution only
> weakly.

Two matrices over the same (agent x component) grid:

  behavioral  from the judge's coding. Cell (i, j) is how much of agent i's
              activity was the action associated with component j.
  causal      from ablation. Cell (i, j) is how much worse component j got when
              agent i was removed, after both main effects are stripped out.

Convergent validity is the correlation between them. A high correlation says
reading the transcript tells you who mattered; a low one says it does not, and
independently replicates *Agents that Matter*' finding that introspective
judgment diverges from ablation.

**Both matrices are double-centered before correlating, and that is not
cosmetic.** Raw cells carry two main effects -- talkative agents and fragile
components -- that appear in *both* matrices for reasons having nothing to do
with specialization. A verbose agent scores high across the behavioral row, and
removing it hurts every component, so the correlation would be strongly positive
in a team with no division of labor at all. Centering leaves only the
interaction in each, which is the only part the claim is about.

**A weak result here is not a failed experiment.** It is the stronger form of the
thesis, and this module reports the number either way -- with a permutation test
over agent labels, so that "weak" and "indistinguishable from chance" stay
distinguishable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from collabengine.analysis.coding import (
    ACTION_TO_COMPONENT,
    MessageCode,
    action_distributions,
)
from collabengine.analysis.interaction import AblationMatrix, double_center
from collabengine.tasks.schema import ALL_COMPONENTS, Component


def behavioral_matrix(
    codes: Sequence[MessageCode],
    *,
    agents: Sequence[str] | None = None,
    components: Sequence[Component] = ALL_COMPONENTS,
) -> tuple[list[str], list[Component], np.ndarray]:
    """Agent x component activity, read off the coded transcript.

    Cell (i, j) is the share of agent i's turns that performed the action mapped
    to component j. Components with no corresponding action get a zero column;
    they contribute nothing to the correlation rather than silently dropping the
    agents' other behavior into them.
    """
    dists = action_distributions(codes)
    agent_list = list(agents) if agents is not None else sorted(dists)
    comp_list = list(components)

    values = np.zeros((len(agent_list), len(comp_list)), dtype=float)
    for i, agent in enumerate(agent_list):
        dist = dists.get(agent, {})
        for j, component in enumerate(comp_list):
            for action, mapped in ACTION_TO_COMPONENT.items():
                if mapped is component:
                    values[i, j] = dist.get(action, 0.0)
                    break
    return agent_list, comp_list, values


@dataclass(slots=True)
class ConvergenceReport:
    r: float
    """Pearson correlation between the two double-centered matrices."""

    p_value: float
    """Permutation p-value over agent-label assignments."""

    n_cells: int
    agents: list[str] = field(default_factory=list)
    components: list[Component] = field(default_factory=list)
    behavioral: np.ndarray | None = None
    causal: np.ndarray | None = None

    @property
    def verdict(self) -> str:
        """Plain-language reading, so the number is not left to interpretation."""
        if self.p_value > 0.05:
            return (
                "transcript labels do not predict causal contribution beyond chance"
            )
        if abs(self.r) < 0.3:
            return "transcript labels predict causal contribution weakly"
        if abs(self.r) < 0.6:
            return "transcript labels predict causal contribution moderately"
        return "transcript labels predict causal contribution strongly"

    def to_dict(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "p_value": self.p_value,
            "n_cells": self.n_cells,
            "agents": list(self.agents),
            "components": [c.value for c in self.components],
            "behavioral": self.behavioral.tolist() if self.behavioral is not None else [],
            "causal": self.causal.tolist() if self.causal is not None else [],
            "verdict": self.verdict,
        }


def convergent_validity(
    codes: Sequence[MessageCode],
    matrix: AblationMatrix,
    *,
    n_permutations: int = 2000,
    seed: int = 0,
) -> ConvergenceReport:
    """Correlate behavioral labels against causal contribution.

    The permutation null shuffles which agent's *behavioral* row belongs to which
    agent, holding the causal matrix fixed. That is the right null: it asks
    whether this particular assignment of labels to agents predicts contribution
    better than an arbitrary one, which is exactly the convergent-validity
    question. Permuting cells instead would test something weaker and easier to
    pass -- whether the two matrices share any structure at all.
    """
    agents, components, behavioral = behavioral_matrix(
        codes, agents=matrix.agents, components=matrix.components
    )
    causal = double_center(matrix.values)
    centered = double_center(behavioral)

    b = centered.ravel()
    c = causal.ravel()
    r = _pearson(b, c)

    rng = np.random.default_rng(seed)
    n_agents = centered.shape[0]
    if n_agents < 3:
        # With two agents every permutation is the identity or a swap, so a
        # permutation p-value would be meaningless rather than merely weak.
        return ConvergenceReport(
            r=r,
            p_value=1.0,
            n_cells=b.size,
            agents=agents,
            components=components,
            behavioral=centered,
            causal=causal,
        )

    null = np.empty(n_permutations)
    for i in range(n_permutations):
        order = rng.permutation(n_agents)
        null[i] = _pearson(double_center(behavioral[order]).ravel(), c)

    p = float((np.sum(np.abs(null) >= abs(r)) + 1) / (n_permutations + 1))
    return ConvergenceReport(
        r=r,
        p_value=p,
        n_cells=b.size,
        agents=agents,
        components=components,
        behavioral=centered,
        causal=causal,
    )


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r that returns 0.0 rather than nan on a constant vector.

    A constant matrix is a real outcome here -- a null world produces one -- and
    it means "no association", not "undefined". Letting nan through would
    poison the permutation distribution silently.
    """
    if a.size != b.size or a.size == 0:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a**2).sum() * (b**2).sum()))
    if denom == 0.0:
        return 0.0
    return float((a * b).sum() / denom)
