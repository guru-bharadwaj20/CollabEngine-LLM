"""Analysis primitives.

The headline quantity is the agent x component interaction, not any scalar
performance drop. `interaction` implements it; everything else in this package
feeds it.

  interaction  the causal side -- what ablation says each agent contributed
  coding       the observational side -- what the transcript says each agent did
  convergent   whether the second predicts the first, which is the whole study
"""

from collabengine.analysis.coding import (
    ActionType,
    CodingReport,
    MessageCode,
    cohens_kappa,
    code_episode,
    ownership_from_codes,
    summarize,
)
from collabengine.analysis.convergent import (
    ConvergenceReport,
    behavioral_matrix,
    convergent_validity,
)
from collabengine.analysis.interaction import (
    AblationMatrix,
    InteractionReport,
    analyze_interaction,
    double_center,
)

__all__ = [
    "AblationMatrix",
    "ActionType",
    "CodingReport",
    "ConvergenceReport",
    "InteractionReport",
    "MessageCode",
    "analyze_interaction",
    "behavioral_matrix",
    "code_episode",
    "cohens_kappa",
    "convergent_validity",
    "double_center",
    "ownership_from_codes",
    "summarize",
]
