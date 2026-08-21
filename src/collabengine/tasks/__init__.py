"""Task families, and the registry a config selects one from.

Two families live here. `collabengine.tasks` itself is the synthetic
multi-constraint allocation task -- generated from a hidden ground-truth
assignment, so every instance is satisfiable by construction and the score
ceiling is known. `collabengine.tasks.code` is a synthetic Python
implementation task graded by a hidden test suite. Both tag their output with a
`component`, and both graders report per-component scores rather than a scalar,
which is what makes the agent x component interaction measurable.

The registry below is what a config selects through (`team.task`). It is
deliberately thin -- a record of callables, not a base class -- because the two
families share no state and inheritance would only invite one to reach into the
other. What it does buy is a single place where "what does this run generate,
render, parse and grade with" is answered, so the orchestrator, the mock
backend and the transcript reader all agree on the answer instead of each
hard-coding the allocation family.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from collabengine.tasks import code as _code
from collabengine.tasks.code import generator as _code_generator
from collabengine.tasks.code import grader as _code_grader
from collabengine.tasks.code import render as _code_render
from collabengine.tasks.code.schema import CODE_COMPONENTS, CodeSolution
from collabengine.tasks.generator import generate as _allocation_generate
from collabengine.tasks.grader import grade as _allocation_grade
from collabengine.tasks.render import SOLO_BRIEF, TEAM_BRIEF
from collabengine.tasks.render import parse_solution as _allocation_parse
from collabengine.tasks.render import render_answer_format as _allocation_answer_format
from collabengine.tasks.render import render_instance as _allocation_render
from collabengine.tasks.schema import (
    ALL_COMPONENTS,
    Component,
    Constraint,
    Instance,
    Job,
    Solution,
    Worker,
)

#: The family every config gets if it does not name one. Every corpus generated
#: before the code family existed was generated under this one, and a run whose
#: family depended on the order of a dict would not be reproducible from config
#: alone -- which is the property `config.py` exists to protect.
DEFAULT_FAMILY = "scheduling"


@dataclass(frozen=True, slots=True)
class TaskFamily:
    """Everything the harness needs to run one family end to end."""

    name: str
    generate: Callable[[int, str], Any]
    grade: Callable[[Any, Any], Any]
    render_instance: Callable[[Any], str]
    render_answer_format: Callable[[], str]
    parse_solution: Callable[[str], Any]
    instance_from_dict: Callable[[dict[str, Any]], Any]
    solution_from_dict: Callable[[dict[str, Any]], Any]
    malformed_solution: Callable[[], Any]
    """A solution meaning "nothing parseable was ever emitted", for the fallback
    at the end of `orchestrator.episode.extract_solution`."""
    team_brief: str
    solo_brief: str
    difficulties: tuple[str, ...]
    """Ordered easy -> hard. `cli.calibrate` sweeps in this order."""
    components: tuple[Component, ...]


SCHEDULING = TaskFamily(
    name="scheduling",
    generate=_allocation_generate,
    grade=_allocation_grade,
    render_instance=_allocation_render,
    render_answer_format=_allocation_answer_format,
    parse_solution=_allocation_parse,
    instance_from_dict=Instance.from_dict,
    solution_from_dict=Solution.from_dict,
    malformed_solution=lambda: Solution(malformed=True),
    team_brief=TEAM_BRIEF,
    solo_brief=SOLO_BRIEF,
    # The order `cli.calibrate` used before the registry existed, which was
    # `sorted(PRESETS, key=n_jobs)`. Spelled out rather than recomputed so that
    # adding a preset cannot silently reorder a published difficulty curve.
    difficulties=("tiny", "easy", "medium", "hard", "xhard"),
    components=ALL_COMPONENTS,
)

CODE = TaskFamily(
    name="code",
    generate=_code_generator.generate,
    grade=_code_grader.grade,
    render_instance=_code_render.render_instance,
    render_answer_format=_code_render.render_answer_format,
    parse_solution=_code_render.parse_solution,
    instance_from_dict=_code.CodeInstance.from_dict,
    solution_from_dict=CodeSolution.from_dict,
    malformed_solution=lambda: CodeSolution(malformed=True),
    team_brief=_code_render.TEAM_BRIEF,
    solo_brief=_code_render.SOLO_BRIEF,
    difficulties=_code_generator.DIFFICULTY_ORDER,
    components=CODE_COMPONENTS,
)

TASK_FAMILIES: dict[str, TaskFamily] = {
    SCHEDULING.name: SCHEDULING,
    CODE.name: CODE,
}


def get_family(name: str | None = None) -> TaskFamily:
    """Look up a family, failing on a typo rather than defaulting past it.

    A silent fallback here would run a whole night on the wrong task and label
    the corpus with the family the config asked for, which is the class of
    error the config module's docstring is about.
    """
    key = name or DEFAULT_FAMILY
    if key not in TASK_FAMILIES:
        raise ValueError(
            f"unknown task family {key!r}; expected one of {sorted(TASK_FAMILIES)}"
        )
    return TASK_FAMILIES[key]


def presets_for(name: str | None = None) -> tuple[str, ...]:
    """The difficulty names a family accepts, easy to hard."""
    return get_family(name).difficulties


__all__ = [
    "ALL_COMPONENTS",
    "CODE",
    "DEFAULT_FAMILY",
    "SCHEDULING",
    "TASK_FAMILIES",
    "Component",
    "Constraint",
    "Instance",
    "Job",
    "Solution",
    "TaskFamily",
    "Worker",
    "get_family",
    "presets_for",
]
