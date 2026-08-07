"""Synthetic multi-constraint allocation task.

Instances are generated from a hidden ground-truth assignment, so every instance
is satisfiable by construction and the score ceiling is known. Constraints carry
a `component` tag; the grader reports per-component scores rather than a scalar,
which is what makes the agent x component interaction measurable.
"""

from collabengine.tasks.schema import (
    Component,
    Constraint,
    Instance,
    Job,
    Solution,
    Worker,
)

__all__ = [
    "Component",
    "Constraint",
    "Instance",
    "Job",
    "Solution",
    "Worker",
]
