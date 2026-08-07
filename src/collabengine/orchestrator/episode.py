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

TEAM_BRIEF = (
    "You are one of {n} participants working together on the task below. "
    "You share a single answer: whatever the group last submits is what gets "
    "scored. You can see everything the others write.\n\n"
    "You are {agent_id}. Prefix your messages with ({agent_id})."
)


def build_system_prompt(agent: Agent, instance: Instance, n_agents: int) -> str:
    """The per-agent system prompt.

    Identical across agents except for the label and, at the highest symmetry-
    breaking level, a vague disposition line. No agent is told what to do, what
    anyone else will do, or that the work can be divided.
    """
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
            request = GenRequest(
                messages=[
                    ChatMessage(
                        role="system",
                        content=build_system_prompt(agent, instance, len(roster)),
                    ),
                    *build_context(agent, history),
                ],
                max_tokens=config.max_tokens,
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
