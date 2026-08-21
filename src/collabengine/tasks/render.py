"""Rendering instances to prompts, and parsing answers back out.

This module is the only place an `Instance` becomes agent-visible text. It must
never emit `ground_truth` or `planted_errors` -- a leak there would silently
invalidate the verification component and inflate every score in the study.
`tests/test_render.py` asserts this on every preset.
"""

from __future__ import annotations

import json
import re
from typing import Any

from collabengine.tasks.schema import Component, Constraint, Instance, Solution

ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"

#: Templated on the agent count, which reads badly at `n=1`: the solo and
#: `solo_budget` arms open with "You are one of 1 participants working together"
#: and a reference to "the others". Left exactly as-is on purpose. The worry --
#: that an agent told it has absent partners would defer or wait, biasing the
#: gate toward the team -- was measured on the served `medium` corpus instead of
#: assumed: across 72 solo agent messages, 0% name an absent agent and 0% defer
#: or wait (RESEARCH-LOG 4.9). Rewording it now would change the solo brief and
#: the C4 turn budget in the same run, which is the two-variables-at-once move
#: that cost the `medium` write-up in 4.1c. Change it in a run where it is the
#: only thing that changes, and regenerate every arm that uses it.
#:
#: It lives here rather than in the orchestrator because a brief belongs to a
#: task family: the second family needs its own pair, and the orchestrator has
#: to be able to ask which pair is in force. `orchestrator.episode` re-exports
#: both names, so every existing import of them still resolves.
TEAM_BRIEF = (
    "You are one of {n} participants working together on the task below. "
    "You share a single answer: whatever the group last submits is what gets "
    "scored. You can see everything the others write.\n\n"
    "You are {agent_id}. Prefix your messages with ({agent_id})."
)


#: The honest long-form single-agent brief, for `solo_long` (C5).
#:
#: `solo_budget` (C4) gave one agent the team's turn budget under TEAM_BRIEF and
#: lost to the team at two of three tiers -- but it restated its own answer on
#: 25-33% of consecutive turns, because the brief tells it the group's *last*
#: message is what gets scored and it is the only member of the group
#: (RESEARCH-LOG 4.12). That is the harness degrading the baseline, not the
#: model, and it makes the C4 margin an upper bound rather than an estimate.
#:
#: This brief removes the three things that caused it: no phantom co-workers, no
#: instruction that the last message is the answer of record, and an explicit
#: licence to think without re-emitting the whole assignment every turn. It is
#: otherwise the same task, the same answer format, and the same budget.
SOLO_BRIEF = (
    "You are working alone on the task below, across {rounds} turns.\n\n"
    "Use the early turns to think: try an assignment, check it against the "
    "requirements, and revise what does not hold. You do not need to restate "
    "the full assignment every turn -- work on whatever part still needs it.\n\n"
    "Your **final** turn is what gets scored, and it must contain the complete "
    "answer in the format below."
)


def render_instance(instance: Instance) -> str:
    """The task brief, identical for every agent on the team.

    Deliberately contains no role language, no suggested decomposition, and no
    hint that division of labor is expected. Any such hint would plant the
    structure the study claims to observe emerging.
    """
    lines: list[str] = []
    lines.append("# Allocation task")
    lines.append("")
    lines.append(
        "Assign every job to exactly one worker so that all listed requirements hold."
    )
    lines.append("")

    lines.append("## Workers")
    for w in instance.workers:
        lines.append(
            f"- {w.wid}: skills=[{', '.join(w.skills)}] capacity={w.capacity} hours"
        )
    lines.append("")

    lines.append("## Jobs")
    for j in instance.jobs:
        lines.append(
            f"- {j.jid}: needs={j.skill} duration={j.duration}h value={j.value} block={j.block}"
        )
    lines.append("")

    lines.append("## Requirements")
    for c in instance.constraints:
        lines.append(f"- [{c.cid}] {describe_constraint(c)}")
    lines.append("")

    lines.append("## Draft assignment under review")
    lines.append(
        "A colleague produced the draft below. It contains a small number of "
        "mistakes. Report the job ids that are assigned incorrectly."
    )
    for jid in sorted(instance.proposed_assignment, key=_job_sort_key):
        lines.append(f"- {jid} -> {instance.proposed_assignment[jid]}")
    lines.append("")

    return "\n".join(lines)


def describe_constraint(c: Constraint) -> str:
    p = c.params
    if c.kind == "capacity":
        return f"Total duration assigned to {p['worker']} must not exceed {p['limit']} hours."
    if c.kind == "value_floor":
        return f"Total value of all assigned jobs must be at least {p['threshold']}."
    if c.kind == "skill_match":
        return f"{p['job']} must go to a worker having the '{p['skill']}' skill."
    if c.kind == "exclusion":
        return f"{p['job']} must NOT be assigned to any of: {', '.join(p['banned'])}."
    if c.kind == "co_assign":
        return f"{p['job_a']} and {p['job_b']} must be assigned to the same worker."
    if c.kind == "separate":
        return f"{p['job_a']} and {p['job_b']} must be assigned to different workers."
    raise ValueError(f"unknown constraint kind {c.kind!r}")


def render_answer_format() -> str:
    """The final-answer contract, issued by the moderator, not by any agent."""
    return (
        f"When the team has settled on an answer, emit it wrapped in "
        f"{ANSWER_OPEN}...{ANSWER_CLOSE} tags as a single JSON object with exactly "
        f"two keys:\n"
        f'  "assignment": an object mapping every job id to one worker id\n'
        f'  "flagged_errors": a list of job ids that are wrong in the draft above\n'
        f"Example:\n"
        f'{ANSWER_OPEN}{{"assignment": {{"J1": "W2", "J2": "W1"}}, '
        f'"flagged_errors": ["J2"]}}{ANSWER_CLOSE}'
    )


def parse_solution(text: str) -> Solution:
    """Extract a solution from free-form model output.

    Small models wrap JSON in prose, code fences, and commentary, and often emit
    several candidate blocks across a turn. We take the last parseable one --
    later revisions supersede earlier drafts -- and fall back through
    progressively looser strategies before declaring the output malformed.
    """
    for candidate in reversed(_candidate_blocks(text)):
        parsed = _try_parse(candidate)
        if parsed is not None:
            return parsed
    return Solution(malformed=True)


def _candidate_blocks(text: str) -> list[str]:
    """Ordered best-guess JSON payloads, most authoritative last."""
    blocks: list[str] = []

    # Loosest first, so that later (more authoritative) matches win.
    blocks.extend(_balanced_objects(text))

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    for f in fenced:
        blocks.extend(_balanced_objects(f))

    tagged = re.findall(
        re.escape(ANSWER_OPEN) + r"(.*?)" + re.escape(ANSWER_CLOSE),
        text,
        flags=re.DOTALL,
    )
    for t in tagged:
        blocks.extend(_balanced_objects(t))

    return blocks


def _balanced_objects(text: str) -> list[str]:
    """Every brace-balanced `{...}` span, ignoring braces inside strings."""
    out: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    out.append(text[start : i + 1])
    return out


def _try_parse(raw: str) -> Solution | None:
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or "assignment" not in obj:
        return None

    assignment = obj.get("assignment")
    if not isinstance(assignment, dict):
        return None

    clean: dict[str, str] = {}
    for k, v in assignment.items():
        if isinstance(k, str) and isinstance(v, str):
            clean[k.strip()] = v.strip()

    flagged_raw: Any = obj.get("flagged_errors", [])
    flagged: list[str] = []
    if isinstance(flagged_raw, list):
        flagged = [str(x).strip() for x in flagged_raw if isinstance(x, (str, int))]
    elif isinstance(flagged_raw, str):
        flagged = [s.strip() for s in flagged_raw.split(",") if s.strip()]

    return Solution(assignment=clean, flagged_errors=flagged)


def _job_sort_key(jid: str) -> tuple[int, str]:
    m = re.match(r"J(\d+)$", jid)
    return (int(m.group(1)), "") if m else (10**9, jid)


def component_glossary() -> dict[Component, str]:
    """Human-readable component descriptions, for analysis output only.

    Never rendered into a prompt: naming the components to agents would suggest
    the very decomposition we are testing for.
    """
    return {
        Component.ARITHMETIC: "capacity and value bookkeeping",
        Component.SEARCH: "skill and exclusion feasibility search",
        Component.VERIFICATION: "auditing the draft for planted errors",
        Component.SYNTHESIS: "cross-block coupling constraints",
    }
