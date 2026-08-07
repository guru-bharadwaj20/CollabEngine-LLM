"""The agent x component interaction.

Ablating an agent lowers scores for two quite different reasons, and the whole
project turns on telling them apart:

  main effect (agent)      the team lost a contributor and plays with one fewer
                           voice per round, so *everything* degrades a little
  main effect (component)  some components are intrinsically harder, so they
                           sit lower in every condition
  interaction              ablating THIS agent hurts THIS component beyond what
                           the two main effects predict

Only the third is evidence of specialization. Double-centering the ablation
matrix -- subtract row means, subtract column means, add back the grand mean --
removes both main effects and leaves exactly the interaction. This is the
standard two-way decomposition; writing it out here rather than reaching for a
model fit keeps the quantity the tests assert on identical to the quantity the
paper reports.

A second point that a row-wise reading gets wrong: "which component fell most
when we removed agent i" is the wrong question, because a component that is
fragile under *any* ablation will win every row. The right question is
column-wise -- "of all the agents we could have removed, which one hurt this
component most" -- and specialization predicts that the answer is the component's
owner. That is diagonal dominance, below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from collabengine.tasks.schema import ALL_COMPONENTS, Component


@dataclass(slots=True)
class AblationMatrix:
    """Per-(agent, component) score drops relative to the un-ablated baseline.

    `values[i][j]` is baseline minus ablated for agent i, component j. Positive
    means removing that agent made that component worse.
    """

    agents: list[str]
    components: list[Component]
    values: np.ndarray
    n_episodes: int = 0

    @classmethod
    def from_means(
        cls,
        baseline: dict[Component, float],
        ablated: dict[str, dict[Component, float]],
        *,
        components: list[Component] | None = None,
        n_episodes: int = 0,
    ) -> AblationMatrix:
        comps = components or list(ALL_COMPONENTS)
        agents = sorted(ablated)
        values = np.array(
            [[baseline[c] - ablated[a][c] for c in comps] for a in agents],
            dtype=float,
        )
        return cls(
            agents=agents, components=comps, values=values, n_episodes=n_episodes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agents": list(self.agents),
            "components": [c.value for c in self.components],
            "values": self.values.tolist(),
            "n_episodes": self.n_episodes,
        }


def double_center(values: np.ndarray) -> np.ndarray:
    """Strip both main effects, leaving the interaction.

    residual[i][j] = x[i][j] - rowmean[i] - colmean[j] + grandmean
    """
    if values.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    row = values.mean(axis=1, keepdims=True)
    col = values.mean(axis=0, keepdims=True)
    return values - row - col + values.mean()


@dataclass(slots=True)
class InteractionReport:
    residuals: np.ndarray
    """Double-centered drops: the interaction term itself."""

    interaction_strength: float
    """RMS of the residuals. Zero when ablation is purely additive."""

    diagonal_dominance: float
    """Fraction of components whose worst ablation is that component's owner.

    Chance is 1/n_agents. This is the statistic that separates identity-bound
    roles from positional artifacts: in a positional world every agent damages
    every component about equally, so ownership does not predict the column max.
    """

    owner_of: dict[Component, str] = field(default_factory=dict)
    column_argmax: dict[Component, str] = field(default_factory=dict)
    chance_level: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "residuals": self.residuals.tolist(),
            "interaction_strength": self.interaction_strength,
            "diagonal_dominance": self.diagonal_dominance,
            "owner_of": {c.value: a for c, a in self.owner_of.items()},
            "column_argmax": {c.value: a for c, a in self.column_argmax.items()},
            "chance_level": self.chance_level,
        }


def analyze_interaction(
    matrix: AblationMatrix,
    ownership: dict[Component, str] | None = None,
) -> InteractionReport:
    """Decompose an ablation matrix into its interaction structure.

    `ownership` maps each component to the agent claimed to own it -- normally
    derived from Phase 2 behavioral coding, which is what makes Phase 4's
    convergent-validity question ("do transcript labels predict causal
    contribution?") answerable at all. When omitted, dominance is scored against
    the empirical argmax, which is trivially 1.0 and reported only for symmetry.
    """
    residuals = double_center(matrix.values)
    strength = float(np.sqrt((residuals**2).mean()))

    column_argmax: dict[Component, str] = {}
    for j, comp in enumerate(matrix.components):
        i = int(np.argmax(residuals[:, j]))
        column_argmax[comp] = matrix.agents[i]

    owner_of = dict(ownership or {})
    if owner_of:
        hits = sum(
            1
            for comp, owner in owner_of.items()
            if comp in column_argmax and column_argmax[comp] == owner
        )
        dominance = hits / len(owner_of)
    else:
        dominance = 1.0

    return InteractionReport(
        residuals=residuals,
        interaction_strength=strength,
        diagonal_dominance=dominance,
        owner_of=owner_of,
        column_argmax=column_argmax,
        chance_level=1.0 / len(matrix.agents) if matrix.agents else 0.0,
    )
