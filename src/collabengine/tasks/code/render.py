"""Rendering code instances to prompts, and parsing submissions back out.

This module is the only place a `CodeInstance` becomes agent-visible text. It
must never emit `tests`, `planted_bug_rule` or `reference_code` -- a leak there
would make the whole suite free and inflate every score in the study.
`tests/test_code_task.py` asserts this structurally, by scrambling the hidden
fields and demanding the prompt not change by one byte.
"""

from __future__ import annotations

import ast
import re

from collabengine.tasks.code.generator import RULE_KINDS, describe_aggregate
from collabengine.tasks.code.schema import CodeInstance, CodeSolution
from collabengine.tasks.render import ANSWER_CLOSE, ANSWER_OPEN
from collabengine.tasks.render import TEAM_BRIEF as _ALLOCATION_TEAM_BRIEF
from collabengine.tasks.schema import Component

#: Shared verbatim with the allocation family, and deliberately so: it says
#: nothing about what the task is, only how the group's answer is taken. Holding
#: it identical means a cross-family comparison of the team arm is not also a
#: comparison of two differently-worded briefs.
TEAM_BRIEF = _ALLOCATION_TEAM_BRIEF

#: The single-agent brief, which cannot be shared, because the allocation
#: version licenses the agent not to "restate the full assignment" and this
#: family has no assignment. Same three properties that made C5 work
#: (RESEARCH-LOG 4.12): no phantom co-workers, no instruction that the last
#: message is the answer of record, and an explicit licence to think without
#: re-emitting the whole artifact every turn.
SOLO_BRIEF = (
    "You are working alone on the task below, across {rounds} turns.\n\n"
    "Use the early turns to think: sketch the implementation, walk it through "
    "the specification one rule at a time, and check what the starter code "
    "actually does against what the specification says it should do. You do not "
    "need to repeat the whole module every turn -- work on whatever part still "
    "needs it.\n\n"
    "Your **final** turn is what gets scored, and it must contain the complete "
    "module in the format below."
)


def render_instance(instance: CodeInstance) -> str:
    """The task brief, identical for every agent on the team.

    Contains no role language, no suggested decomposition, and no hint that
    division of labor is expected -- any such hint would plant the structure the
    study claims to observe emerging.

    It also does not say which stage of the starter code is wrong, or that a
    specific number of things are wrong beyond "exactly one". Naming the stage
    would give away the `verify` component; saying nothing at all would turn it
    into a guess rather than a check against the specification above it.
    """
    lines: list[str] = []
    lines.append("# Implementation task")
    lines.append("")
    lines.append(
        "Write a Python module that implements the specification below. Your "
        "module is scored by running it against hidden tests."
    )
    lines.append("")

    lines.append("## Input")
    lines.append(
        "`values` is a list of integers. It may contain repeats, zero and "
        "negative numbers, and it may be empty."
    )
    lines.append("")

    lines.append("## Rules, applied in this order")
    for n, r in enumerate(instance.rules):
        lines.append(f"{n + 1}. [{r.rid}] {RULE_KINDS[r.kind].describe(r.params)}")
    lines.append("")

    lines.append("## Result")
    lines.append(
        f"Once every rule above has been applied in order, return "
        f"{describe_aggregate(instance.aggregate)}."
    )
    lines.append(
        f"If no values are left after the rules, return {instance.empty_default}."
    )
    lines.append("")

    lines.append("## Functions the module must define")
    for name in instance.required_functions:
        lines.append(f"- `{name}(values)`: {_describe_required(name)}")
    lines.append("")

    lines.append("## Starter code")
    lines.append(
        "A colleague wrote the stage helpers below. Exactly one of them does "
        "not do what the specification above says it should. The module is "
        "scored on what it computes, not on which lines survive, so correct or "
        "replace whatever does not match."
    )
    lines.append("")
    lines.append("```python")
    lines.append(instance.starter_code.rstrip())
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def _describe_required(name: str) -> str:
    if name == "solve":
        return "returns the result described under **Result**"
    if name == "apply_rules":
        return "returns the list of values left after every rule has been applied"
    return "as described above"


def render_answer_format() -> str:
    """The final-answer contract, issued by the moderator, not by any agent.

    Worded for one agent or several without changing -- "when the team has
    settled" would smuggle co-workers into the solo arm through the answer
    format after `SOLO_BRIEF` had carefully kept them out.
    """
    return (
        f"When the answer is settled, emit the complete module wrapped in "
        f"{ANSWER_OPEN}...{ANSWER_CLOSE} tags. Put nothing but Python source "
        f"inside the tags -- no prose, no explanation, no markdown.\n"
        f"Example:\n"
        f"{ANSWER_OPEN}\n"
        f"def solve(values):\n"
        f"    return 0\n"
        f"{ANSWER_CLOSE}"
    )


def parse_solution(text: str) -> CodeSolution:
    """Extract a module from free-form model output.

    Same shape as the allocation parser and for the same reasons: small models
    wrap the artifact in prose and code fences, and often emit several candidate
    versions across a turn. We take the last *parseable* one -- later revisions
    supersede earlier drafts -- and fall back through progressively looser
    strategies.

    The one difference is the last fallback. If a block was clearly offered as
    the answer but does not compile, that block is returned rather than declared
    malformed: "they submitted broken code" and "they never submitted anything"
    are different outcomes, and only the first is something the `syntax`
    component can score.
    """
    blocks = _candidate_blocks(text)

    for candidate in reversed(blocks):
        if _parses(candidate):
            return CodeSolution(code=candidate, detail={"parsed": True})
    for candidate in reversed(blocks):
        if candidate.strip():
            return CodeSolution(code=candidate, detail={"parsed": False})
    return CodeSolution(malformed=True)


def _candidate_blocks(text: str) -> list[str]:
    """Ordered best-guess submissions, most authoritative last."""
    blocks: list[str] = []

    # Loosest first: the whole turn, on the chance the model wrote bare source.
    if "def " in text:
        blocks.append(_strip_fences(text))

    for fenced in re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL):
        blocks.append(fenced)

    for tagged in re.findall(
        re.escape(ANSWER_OPEN) + r"(.*?)" + re.escape(ANSWER_CLOSE),
        text,
        flags=re.DOTALL,
    ):
        blocks.append(_strip_fences(tagged))

    return [b for b in blocks if b.strip()]


def _strip_fences(text: str) -> str:
    """Drop markdown fence lines, keeping the source between them.

    Models put the fence inside the answer tags about as often as outside, and a
    stray ``` is a SyntaxError -- which would score `syntax` at zero for a
    formatting habit rather than for anything about the code.
    """
    kept = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
    return "\n".join(kept).strip("\n")


def _parses(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False
    # A block with no def in it is prose that happened to be valid Python, not a
    # submission -- `"I am not sure."` parses as an expression statement.
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )


def component_glossary() -> dict[Component, str]:
    """Human-readable component descriptions, for analysis output only.

    Never rendered into a prompt: naming the components to agents would suggest
    the very decomposition we are testing for.
    """
    return {
        Component.SYNTAX: "module compiles and defines the required functions",
        Component.BASE: "base-case hidden tests",
        Component.EDGE: "edge-case hidden tests",
        Component.VERIFY: "catching the bug planted in the starter code",
    }
