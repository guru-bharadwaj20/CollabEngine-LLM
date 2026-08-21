"""Analysis primitives.

The headline quantity is the agent x component interaction, not any scalar
performance drop. `interaction` implements it; everything else in this package
feeds it.

  interaction  the causal side -- what ablation says each agent contributed
  mixed        whether that interaction survives a significance test
  coding       the observational side -- what the transcript says each agent did
  convergent   whether the second predicts the first, which is the whole study
  inference    multiplicity, equivalence and power -- what a null is evidence for

`inference` is the newest and the one that changes how the others are read.
Every headline in this project is a null, and a null needs an equivalence bound
to be evidence rather than an absence of it.

`mixed` needs pandas and statsmodels (the `analysis` extra) and is imported
lazily, so the rest of the stack works without them; `inference` does the same
with scipy.
"""

from collabengine.analysis.coding import (
    ActionType,
    CodingReport,
    MessageCode,
    code_episode,
    cohens_kappa,
    kappa_interval,
    ownership_from_codes,
    summarize,
)
from collabengine.analysis.convergent import (
    ConvergenceReport,
    behavioral_matrix,
    convergent_validity,
)
from collabengine.analysis.inference import (
    PowerPlan,
    TostResult,
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
    "PowerPlan",
    "TostResult",
    "adjust",
    "analyze_interaction",
    "behavioral_matrix",
    "bh_fdr",
    "code_episode",
    "cohens_d",
    "cohens_kappa",
    "convergent_validity",
    "holm",
    "kappa_interval",
    "double_center",
    "mde",
    "n_for",
    "ownership_from_codes",
    "plan",
    "sd_table",
    "smallest_equivalence_bound",
    "summarize",
    "tost",
    "welch",
]
