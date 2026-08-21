"""Running one episode.

The loop is round-based: each round every active agent speaks once, in an order
that is reshuffled per round when `randomize_turn_order` is set. Ablation removes
an agent from the roster rather than shortening the schedule, so an N-1 team
plays the same number of rounds with one fewer voice per round.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence

from collabengine.backends.base import ChatMessage, GenRequest, LLMBackend
from collabengine.orchestrator.team import Agent, TeamConfig, build_team
from collabengine.protocol import Message, Speaker
from collabengine.tasks import TaskFamily, get_family
from collabengine.tasks.render import SOLO_BRIEF, TEAM_BRIEF
from collabengine.transcripts.store import EpisodeRecord

#: Re-exported so that every existing import of these names still resolves. They
#: are defined in `collabengine.tasks.render` because a brief belongs to a task
#: family -- the code family needs its own pair, and the orchestrator has to be
#: able to ask which pair is in force rather than know one of them by heart.
__all__ = [
    "SOLO_BRIEF",
    "TEAM_BRIEF",
    "build_context",
    "build_system_prompt",
    "extract_solution",
    "latest_notes",
    "plan_turn_order",
    "run_episode",
]


def build_system_prompt(
    agent: Agent,
    instance: object,
    n_agents: int,
    *,
    solo_long: bool = False,
    rounds: int | None = None,
    working_notes: bool = False,
    family: TaskFamily | None = None,
) -> str:
    """The per-agent system prompt.

    Identical across agents except for the label and, at the highest symmetry-
    breaking level, a vague disposition line. No agent is told what to do, what
    anyone else will do, or that the work can be divided.

    `solo_long` swaps in the family's solo brief, which is the C5 baseline and
    the only place the wording differs. Everything else -- task, answer format,
    caps, seeds -- is held.
    """
    fam = family or get_family()
    if solo_long:
        parts = [
            fam.solo_brief.format(rounds=rounds if rounds is not None else "several")
        ]
    else:
        parts = [fam.team_brief.format(n=n_agents, agent_id=agent.agent_id)]
        if agent.scratch:
            parts.append(agent.scratch)
    # Appended to whichever brief is in force. The mechanism is symmetric across
    # conditions, so the sentence describing it has to be too -- describing it to
    # one arm only would hand that arm a protocol the other cannot use.
    if working_notes:
        parts.append(NOTES_BRIEF)
    parts.append(fam.render_instance(instance))
    parts.append(fam.render_answer_format())
    return "\n\n".join(parts)


NOTES_RE = re.compile(r"<notes>(.*?)</notes>", re.DOTALL | re.IGNORECASE)

NOTES_BRIEF = (
    "You have a working area. Anything you put inside <notes>...</notes> is "
    "carried forward and shown back to you at the start of every one of your "
    "turns, so you do not have to repeat it to keep it. Use it for the "
    "assignment you are building and what you have already checked. Only your "
    "most recent <notes> block is kept."
)


def latest_notes(agent_id: str, history: Sequence[Message]) -> str:
    """The most recent `<notes>` block this agent wrote, or ''.

    Only the newest is kept. Accumulating them would reintroduce the problem the
    mechanism exists to remove -- an agent re-reading every past state instead of
    one current one -- and would grow the prompt exactly as fast as restating the
    schedule did (RESEARCH-LOG 4.12).
    """
    for m in reversed(history):
        if m.speaker is Speaker.AGENT and m.author == agent_id:
            found = NOTES_RE.findall(m.content or "")
            if found:
                return found[-1].strip()
    return ""


def build_context(
    agent: Agent, history: Sequence[Message], *, working_notes: bool = False
) -> list[ChatMessage]:
    """Render shared history from one agent's point of view.

    Each agent sees its own prior turns as `assistant` and everyone else's as
    `user`. That asymmetry is what gives an agent a sense of its own continuity
    across turns -- the substrate a stable role would have to form on.

    With `working_notes`, the agent's own latest `<notes>` block is surfaced
    ahead of the transcript. The block stays in the transcript too, visible to
    everyone: making notes private would change what agents know about each
    other, which is a different experiment from the one this flag is for.
    """
    out: list[ChatMessage] = []
    if working_notes:
        notes = latest_notes(agent.agent_id, history)
        if notes:
            out.append(ChatMessage(role="user", content=f"Your working notes:\n{notes}"))
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
    instance: object | None = None,
) -> EpisodeRecord:
    """Run a team through one instance and grade the result.

    `exclude` drops agents from the roster before the episode starts -- this is
    live ablation. The remaining agents are free to reorganize around the gap,
    which is exactly the compensation effect that frozen-transcript ablation is
    designed to block. Both numbers are wanted; their difference is the
    fungibility measure.
    """
    family = get_family(config.task)
    if instance is None:
        instance = family.generate(episode_seed, config.difficulty)

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
            content=family.render_instance(instance),
            meta={"kind": "task_brief", "task": family.name},
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
                            working_notes=config.working_notes,
                            family=family,
                        ),
                    ),
                    *build_context(
                        agent, history, working_notes=config.working_notes
                    ),
                ],
                max_tokens=turn_cap,
                temperature=config.temperature,
                top_p=config.top_p,
                seed=agent.seed,
                meta={
                    "instance": instance.to_dict(),
                    # The backend needs to know which family's dict that is
                    # before it can read it -- the mock reconstructs the instance
                    # to score itself against.
                    "task": family.name,
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

    solution = extract_solution(history, family=family)
    return EpisodeRecord(
        episode_id=f"{condition}:{config.difficulty}:{episode_seed}",
        condition=condition,
        instance_seed=episode_seed,
        difficulty=config.difficulty,
        agents=[a.agent_id for a in roster],
        messages=history,
        solution=solution,
        grade=family.grade(instance, solution),
        turn_order=turn_order,
        config=config.to_dict(),
        meta={
            "excluded": list(exclude),
            "backend": backend.name,
            "roster_size": len(roster),
            "task": family.name,
        },
    )


def extract_solution(
    history: Sequence[Message], *, family: TaskFamily | None = None
) -> object:
    """Take the team's answer as the last parseable proposal by any agent.

    Last-writer-wins rather than majority vote: the brief tells the team that
    whatever it last submits is what gets scored, so this matches the incentive
    the agents were actually given.
    """
    fam = family or get_family()
    for m in reversed(history):
        if m.speaker is not Speaker.AGENT:
            continue
        candidate = fam.parse_solution(m.content)
        if not candidate.malformed:
            return candidate
    return fam.malformed_solution()


def _round_banner(round_index: int, total: int) -> str:
    if round_index == total - 1:
        return (
            f"Round {round_index + 1} of {total}. This is the final round -- "
            "make sure the answer you want scored is submitted before it ends."
        )
    return f"Round {round_index + 1} of {total}."
