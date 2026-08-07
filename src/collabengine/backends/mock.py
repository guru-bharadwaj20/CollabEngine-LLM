"""Deterministic mock backend.

Two jobs. First, it lets the entire pipeline -- orchestration, ablation,
analysis -- run and be debugged with no GPU. Second, and more importantly, it is
a *known-ground-truth testbed for the measurement instrument itself*.

The modes below construct worlds where the answer is known in advance:

  SPECIALIZED  each agent genuinely owns one task component
  NULL         every agent does the same thing; there is no division of labor
  POSITIONAL   behavior tracks turn slot, not agent identity (confound C2)

Running the Phase 3 analysis against these tells us whether the pipeline detects
specialization when it exists, reports none when it does not, and distinguishes
identity-bound roles from positional artifacts. A pipeline that "finds"
specialization in NULL mode is manufacturing its result, and we want to know that
before spending GPU hours rather than after.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from enum import Enum

from collabengine.backends.base import GenRequest, GenResponse, LLMBackend
from collabengine.tasks.render import ANSWER_CLOSE, ANSWER_OPEN, parse_solution
from collabengine.tasks.schema import ALL_COMPONENTS, Component, Instance, Solution


class MockMode(str, Enum):
    SPECIALIZED = "specialized"
    """Agent i owns ALL_COMPONENTS[i % 4]. Ablation should show an interaction."""

    NULL = "null"
    """All agents work every component identically. No interaction should appear."""

    POSITIONAL = "positional"
    """Focus follows turn position, not agent id. The C2 test should catch this."""

    INERT = "inert"
    """Agents emit chatter and never improve the solution. Floor-case sanity check."""


# Phrases the behavioral coder should be able to separate into action types.
# Deliberately distinct per component so judge accuracy is measurable on data
# whose labels we already know.
_ACTION_PHRASES: dict[Component, str] = {
    Component.ARITHMETIC: "Let me total the hours per worker and check the value floor.",
    Component.SEARCH: "I'll enumerate which workers are actually eligible for each job.",
    Component.VERIFICATION: "Auditing the draft against the stated requirements now.",
    Component.SYNTHESIS: "Reconciling the cross-block pairings against the rest.",
}


@dataclass(slots=True)
class MockBackend(LLMBackend):
    """A scriptable stand-in for a served model.

    `competence` is the per-item probability that a focused agent fixes something
    correctly. Below 1.0 the scores acquire variance, which the statistical tests
    need; at 1.0 every specialized run saturates and effect sizes are undefined.
    """

    mode: MockMode = MockMode.SPECIALIZED
    competence: float = 0.85
    off_focus_competence: float = 0.05
    latency_s: float = 0.0
    name: str = "mock"

    async def generate(self, request: GenRequest) -> GenResponse:
        meta = request.meta
        instance_d = meta.get("instance")
        if instance_d is None:
            return GenResponse(text="(no instance in meta)", completion_tokens=6)

        instance = Instance.from_dict(instance_d)
        agent_index = int(meta.get("agent_index", 0))
        turn_position = int(meta.get("turn_position", agent_index))
        turn = int(meta.get("turn", 0))
        agent_id = str(meta.get("agent_id", f"A{agent_index}"))

        rng = random.Random(_stable_seed(request.seed, agent_id, turn, self.mode.value))

        focus = self._focus_for(agent_index, turn_position)
        prior = self._latest_solution(request)
        merged = self._contribute(instance, prior, focus, rng)

        text = self._compose(focus, merged, agent_id)
        return GenResponse(
            text=text,
            prompt_tokens=sum(len(m.content) // 4 for m in request.messages),
            completion_tokens=len(text) // 4,
        )

    def _focus_for(self, agent_index: int, turn_position: int) -> tuple[Component, ...]:
        if self.mode is MockMode.SPECIALIZED:
            return (ALL_COMPONENTS[agent_index % len(ALL_COMPONENTS)],)
        if self.mode is MockMode.POSITIONAL:
            return (ALL_COMPONENTS[turn_position % len(ALL_COMPONENTS)],)
        if self.mode is MockMode.NULL:
            return ALL_COMPONENTS
        return ()

    def _latest_solution(self, request: GenRequest) -> Solution:
        """Recover the most recent proposal from the conversation.

        Reading state out of the transcript rather than holding it on the backend
        is what makes frozen-transcript ablation work: excise an agent's messages
        and its contribution genuinely disappears from what everyone else sees.
        """
        for msg in reversed(request.messages):
            sol = parse_solution(msg.content)
            if not sol.malformed:
                return sol
        return Solution()

    def _contribute(
        self,
        instance: Instance,
        prior: Solution,
        focus: tuple[Component, ...],
        rng: random.Random,
    ) -> Solution:
        assignment = dict(prior.assignment)
        flagged = list(prior.flagged_errors)

        # Seed from the draft so early turns have something to improve on.
        if not assignment:
            assignment = dict(instance.proposed_assignment)

        for comp in ALL_COMPONENTS:
            p = self.competence if comp in focus else self.off_focus_competence
            if comp is Component.ARITHMETIC:
                _fix_arithmetic(instance, assignment, p, rng)
            elif comp is Component.SEARCH:
                _fix_search(instance, assignment, p, rng)
            elif comp is Component.SYNTHESIS:
                _fix_synthesis(instance, assignment, p, rng)
            elif comp is Component.VERIFICATION:
                flagged = _fix_verification(instance, flagged, p, rng)

        return Solution(assignment=assignment, flagged_errors=sorted(set(flagged)))

    def _compose(
        self, focus: tuple[Component, ...], sol: Solution, agent_id: str
    ) -> str:
        if not focus:
            return f"({agent_id}) I don't have anything to add on this one."

        opener = " ".join(_ACTION_PHRASES[c] for c in focus)
        payload = (
            '{"assignment": '
            + _json_obj(sol.assignment)
            + ', "flagged_errors": '
            + _json_arr(sol.flagged_errors)
            + "}"
        )
        return f"({agent_id}) {opener}\n{ANSWER_OPEN}{payload}{ANSWER_CLOSE}"


# --------------------------------------------------------------------------
# Component fixers. Each is greedy and imperfect by design: `p` is the chance of
# repairing any given violation, so scores land between floor and ceiling rather
# than saturating.
# --------------------------------------------------------------------------


def _eligible_workers(instance: Instance, jid: str) -> list[str]:
    job = instance.job(jid)
    if job is None:
        return []
    banned: set[str] = set()
    for c in instance.constraints:
        if c.kind == "exclusion" and c.params["job"] == jid:
            banned |= set(c.params["banned"])
    return [
        w.wid for w in instance.workers if job.skill in w.skills and w.wid not in banned
    ]


def _fix_search(
    instance: Instance, assignment: dict[str, str], p: float, rng: random.Random
) -> None:
    if p <= 0.0:
        return
    for job in instance.jobs:
        if rng.random() >= p:
            continue
        eligible = _eligible_workers(instance, job.jid)
        if eligible and assignment.get(job.jid) not in eligible:
            assignment[job.jid] = rng.choice(eligible)


def _fix_arithmetic(
    instance: Instance, assignment: dict[str, str], p: float, rng: random.Random
) -> None:
    if p <= 0.0:
        return

    # Value floor: every unassigned job is lost value, so place them first.
    for job in instance.jobs:
        if job.jid not in assignment and rng.random() < p:
            eligible = _eligible_workers(instance, job.jid) or [
                w.wid for w in instance.workers
            ]
            assignment[job.jid] = rng.choice(eligible)

    # Capacity: shed load from overfull workers onto the emptiest eligible one.
    # Each pass repairs at most one worker and only with probability `p`, so
    # competence grades the outcome smoothly instead of being all-or-nothing.
    caps = {
        c.params["worker"]: int(c.params["limit"])
        for c in instance.constraints
        if c.kind == "capacity"
    }
    for _ in range(2 * len(instance.jobs)):
        load: dict[str, int] = {w.wid: 0 for w in instance.workers}
        for job in instance.jobs:
            wid = assignment.get(job.jid)
            if wid in load:
                load[wid] += job.duration

        over = [w for w, used in load.items() if used > caps.get(w, 10**9)]
        if not over:
            return
        if rng.random() >= p:
            continue

        source = max(over, key=lambda w: load[w] - caps.get(w, 0))
        movable = [j for j in instance.jobs if assignment.get(j.jid) == source]
        if not movable:
            continue
        job = rng.choice(movable)
        targets = [
            w
            for w in _eligible_workers(instance, job.jid)
            if w != source and load.get(w, 0) + job.duration <= caps.get(w, 10**9)
        ]
        if not targets:
            continue
        assignment[job.jid] = min(targets, key=lambda w: load.get(w, 0))


def _fix_synthesis(
    instance: Instance, assignment: dict[str, str], p: float, rng: random.Random
) -> None:
    if p <= 0.0:
        return
    for c in instance.constraints:
        if c.kind not in ("co_assign", "separate") or rng.random() >= p:
            continue
        a, b = c.params["job_a"], c.params["job_b"]
        wa, wb = assignment.get(a), assignment.get(b)

        if c.kind == "co_assign" and wa != wb:
            shared = set(_eligible_workers(instance, a)) & set(
                _eligible_workers(instance, b)
            )
            if shared:
                pick = rng.choice(sorted(shared))
                assignment[a] = assignment[b] = pick
        elif c.kind == "separate" and wa == wb and wa is not None:
            alts = [w for w in _eligible_workers(instance, b) if w != wa]
            if alts:
                assignment[b] = rng.choice(alts)


def _fix_verification(
    instance: Instance, flagged: list[str], p: float, rng: random.Random
) -> list[str]:
    """Re-derive which draft entries violate a skill requirement.

    Notice this reasons from the instance, never from `instance.planted_errors`
    -- the mock is held to the same information the agents get.
    """
    out = set(flagged)
    for jid, wid in instance.proposed_assignment.items():
        if rng.random() >= p:
            continue
        job, worker = instance.job(jid), instance.worker(wid)
        if job is None or worker is None:
            continue
        if job.skill not in worker.skills:
            out.add(jid)
        else:
            out.discard(jid)
    return sorted(out)


def _stable_seed(*parts: object) -> int:
    """Process-independent seed derivation.

    `hash()` on str is salted per interpreter, so using it here would make runs
    irreproducible across the worker processes the parallel runner spawns -- and
    silently, since any single process would look self-consistent.
    """
    joined = "\x1f".join(repr(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(joined, digest_size=8).digest(), "big")


def _json_obj(d: dict[str, str]) -> str:
    inner = ", ".join(f'"{k}": "{v}"' for k, v in sorted(d.items()))
    return "{" + inner + "}"


def _json_arr(xs: list[str]) -> str:
    return "[" + ", ".join(f'"{x}"' for x in xs) + "]"
