"""Running one episode.

The loop is round-based: each round every active agent speaks once, in an order
that is reshuffled per round when `randomize_turn_order` is set. Ablation removes
an agent from the roster rather than shortening the schedule, so an N-1 team
plays the same number of rounds with one fewer voice per round.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from collabengine.backends.base import ChatMessage, GenRequest, LLMBackend
from collabengine.orchestrator.team import Agent, TeamConfig, build_team
from collabengine.protocol import Message, Speaker
from collabengine.tasks.generator import generate
from collabengine.tasks.grader import grade
from collabengine.tasks.render import (
    parse_solution,
    render_answer_format,
    render_instance,
)
from collabengine.tasks.schema import Instance, Solution
from collabengine.transcripts.store import EpisodeRecord

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


def build_system_prompt(
    agent: Agent,
    instance: Instance,
    n_agents: int,
    *,
    solo_long: bool = False,
    rounds: int | None = None,
) -> str:
    """The per-agent system prompt.

    Identical across agents except for the label and, at the highest symmetry-
    breaking level, a vague disposition line. No agent is told what to do, what
    anyone else will do, or that the work can be divided.

    `solo_long` swaps in SOLO_BRIEF, which is the C5 baseline and the only place
    the wording differs. Everything else -- task, answer format, caps, seeds --
    is held.
    """
    if solo_long:
        return "\n\n".join(
            [
                SOLO_BRIEF.format(rounds=rounds if rounds is not None else "several"),
                render_instance(instance),
                render_answer_format(),
            ]
        )
    parts = [TEAM_BRIEF.format(n=n_agents, agent_id=agent.agent_id)]
    if agent.scratch:
        parts.append(agent.scratch)
    parts.append(render_instance(instance))
    parts.append(render_answer_format())
    return "\n\n".join(parts)


def build_context(agent: Agent, history: Sequence[Message]) -> list[ChatMessage]:
    """Render shared history from one agent's point of view.

    Each agent sees its own prior turns as `assistant` and everyone else's as
    `user`. That asymmetry is what gives an agent a sense of its own continuity
    across turns -- the substrate a stable role would have to form on.
    """
    out: list[ChatMessage] = []
    for m in history:
        if m.speaker is Speaker.AGENT and m.author == agent.agent_id:
            out.append(ChatMessage(role="assistant", content=m.content))
        else:
            out.append(ChatMessage(role="user", content=m.content))
    return out


def plan_turn_order(
    agents: Sequence[Agent], config: TeamConfig, rng: random.Random
) -> list[list[str]]:
    """Speaking order for each round.

    Recorded on the episode so the analysis can ask whether an agent's behavior
    tracks who it is or merely when it speaks.
    """
    order: list[list[str]] = []
    for _ in range(config.rounds):
        ids = [a.agent_id for a in agents]
        if config.randomize_turn_order:
            rng.shuffle(ids)
        order.append(ids)
    return order


async def run_episode(
    *,
    backend: LLMBackend,
    config: TeamConfig,
    episode_seed: int,
    condition: str = "baseline",
    exclude: Sequence[str] = (),
    instance: Instance | None = None,
) -> EpisodeRecord:
    """Run a team through one instance and grade the result.

    `exclude` drops agents from the roster before the episode starts -- this is
    live ablation. The remaining agents are free to reorganize around the gap,
    which is exactly the compensation effect that frozen-transcript ablation is
    designed to block. Both numbers are wanted; their difference is the
    fungibility measure.
    """
    if instance is None:
        instance = generate(episode_seed, config.difficulty)

    roster = [a for a in build_team(config, episode_seed) if a.agent_id not in exclude]
    if not roster:
        raise ValueError("every agent was excluded; nothing left to run")

    by_id = {a.agent_id: a for a in roster}
    rng = random.Random(episode_seed ^ 0x5EED)
    turn_order = plan_turn_order(roster, config, rng)

    history: list[Message] = [
        Message(
            turn=0,
            speaker=Speaker.SYSTEM,
            author="system",
            content=render_instance(instance),
            meta={"kind": "task_brief"},
        )
    ]

    turn = 1
    for round_index, round_ids in enumerate(turn_order):
        history.append(
            Message(
                turn=turn,
                speaker=Speaker.MODERATOR,
                author="moderator",
                content=_round_banner(round_index, len(turn_order)),
                meta={"kind": "round_banner", "round": round_index},
            )
        )
        turn += 1

        for position, agent_id in enumerate(round_ids):
            agent = by_id[agent_id]
            # The last agent turn of the last round is the one the parser reads,
            # so it is the only turn where hitting the cap costs the episode its
            # score rather than some of its reasoning. It gets its own budget
            # when the config asks for one -- see TeamConfig.answer_max_tokens
            # and RESEARCH-LOG 4.10 for what a single shared cap did here.
            is_answer_turn = (
                round_index == len(turn_order) - 1
                and position == len(round_ids) - 1
            )
            turn_cap = config.max_tokens
            if is_answer_turn and config.answer_max_tokens is not None:
                turn_cap = config.answer_max_tokens
            request = GenRequest(
                messages=[
                    ChatMessage(
                        role="system",
                        content=build_system_prompt(
                            agent,
                            instance,
                            len(roster),
                            solo_long=(condition == "solo_long"),
                            rounds=config.rounds,
                        ),
                    ),
                    *build_context(agent, history),
                ],
                max_tokens=turn_cap,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=agent.seed,
                meta={
                    "instance": instance.to_dict(),
                    "agent_id": agent.agent_id,
                    "agent_index": agent.index,
                    "turn_position": position,
                    "turn": turn,
                    "round": round_index,
                },
            )
            response = await backend.generate(request)
            history.append(
                Message(
                    turn=turn,
                    speaker=Speaker.AGENT,
                    author=agent.agent_id,
                    content=response.text,
                    meta={
                        "round": round_index,
                        "position": position,
                        "completion_tokens": response.completion_tokens,
                        "finish_reason": response.finish_reason,
                        # The cap in force for this turn, recorded per turn
                        # because it is no longer constant within an episode.
                        # A `length` finish means nothing without the number it
                        # ran into, and lesson 3 of section 8 is that instrument
                        # settings are part of a measurement's identity.
                        "max_tokens": turn_cap,
                        "answer_turn": is_answer_turn,
                        **({"error": response.error} if response.error else {}),
                    },
                )
            )
            turn += 1

    solution = extract_solution(history)
    return EpisodeRecord(
        episode_id=f"{condition}:{config.difficulty}:{episode_seed}",
        condition=condition,
        instance_seed=episode_seed,
        difficulty=config.difficulty,
        agents=[a.agent_id for a in roster],
        messages=history,
        solution=solution,
        grade=grade(instance, solution),
        turn_order=turn_order,
        config=config.to_dict(),
        meta={
            "excluded": list(exclude),
            "backend": backend.name,
            "roster_size": len(roster),
        },
    )


def extract_solution(history: Sequence[Message]) -> Solution:
    """Take the team's answer as the last parseable proposal by any agent.

    Last-writer-wins rather than majority vote: the brief tells the team that
    whatever it last submits is what gets scored, so this matches the incentive
    the agents were actually given.
    """
    for m in reversed(history):
        if m.speaker is not Speaker.AGENT:
            continue
        candidate = parse_solution(m.content)
        if not candidate.malformed:
            return candidate
    return Solution(malformed=True)


def _round_banner(round_index: int, total: int) -> str:
    if round_index == total - 1:
        return (
            f"Round {round_index + 1} of {total}. This is the final round -- "
            "make sure the answer you want scored is submitted before it ends."
        )
    return f"Round {round_index + 1} of {total}."
