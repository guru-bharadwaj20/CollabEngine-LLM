"""Synthetic Python implementation task with a hidden test suite.

The second task family, and the reason it exists is stated in Final Sweep 2.1:
the paper's claim is that a shared per-turn token cap lands asymmetrically when
one arm must emit a whole artifact in its final turn and the other commits one
its transcript already holds. Code generation is where that asymmetry is
largest, so it is the most adversarial venue available for the claim and
therefore the most convincing one.

Instances are generated rather than scraped -- an 8B that has seen MBPP is being
measured on recall, and the score does not say which. Everything else matches
the allocation family by construction: deterministic in `(seed, difficulty)`,
satisfiable by construction, continuous difficulty knobs, and per-component
scores rather than a scalar pass/fail.
"""

from collabengine.tasks.code.schema import (
    CODE_COMPONENTS,
    CodeInstance,
    CodeSolution,
    HiddenTest,
    RuleSpec,
    TestKind,
)

__all__ = [
    "CODE_COMPONENTS",
    "CodeInstance",
    "CodeSolution",
    "HiddenTest",
    "RuleSpec",
    "TestKind",
]
