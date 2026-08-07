"""Episode orchestration.

Deliberately hand-written rather than built on an agent framework. Every
mainstream framework ships opinionated role scaffolding in its prompt templates,
which would silently plant the division of labor this project claims to observe
emerging. The orchestration here is simple enough to audit line by line, and
that auditability is the point.
"""

from collabengine.orchestrator.episode import run_episode
from collabengine.orchestrator.team import Agent, SymmetryBreaking, TeamConfig

__all__ = [
    "Agent",
    "SymmetryBreaking",
    "TeamConfig",
    "run_episode",
]
