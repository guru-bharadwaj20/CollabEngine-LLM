"""Analysis primitives.

The headline quantity is the agent x component interaction, not any scalar
performance drop. `interaction` implements it; everything else in this package
feeds it.
"""

from collabengine.analysis.interaction import (
    AblationMatrix,
    InteractionReport,
    analyze_interaction,
    double_center,
)

__all__ = [
    "AblationMatrix",
    "InteractionReport",
    "analyze_interaction",
    "double_center",
]
