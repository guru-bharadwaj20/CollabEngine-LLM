"""Ablation modes and their controls."""

from __future__ import annotations

import random
from dataclasses import replace
from enum import Enum

from collabengine.backends.base import ChatMessage, GenRequest, LLMBackend
from collabengine.orchestrator.episode import (
    build_context,
    build_system_prompt,
    extract_solution,
    run_episode,
)
from collabengine.orchestrator.team import TeamConfig, build_team
from collabengine.protocol import Message, Speaker
from collabengine.tasks.generator import generate
from collabengine.tasks.grader import grade
from collabengine.tasks.schema import Instance
from collabengine.transcripts.store import EpisodeRecord


class AblationMode(str, Enum):
    LIVE = "live"
    FROZEN_EXCISE = "frozen_excise"
    FROZEN_REPLAY = "frozen_replay"
    CAPACITY = "capacity"
    RANDOM_MESSAGE = "random_message"


async def live_ablation(
    *,
    backend: LLMBackend,
    config: TeamConfig,
    episode_seed: int,
    agent_id: str,
    instance: Instance | None = None,
) -> EpisodeRecord:
    """Re-run the episode with the agent removed from the roster.

    The survivors see the gap from turn one and are free to reorganize around
    it, so this is the compensation-permitting end of the spectrum.
    """
    return await run_episode(
        backend=backend,
        config=config,
        episode_seed=episode_seed,
        condition=f"{AblationMode.LIVE.value}:{agent_id}",
        exclude=(agent_id,),
        instance=instance,
    )


def frozen_excise(
    record: EpisodeRecord,
    agent_id: str,
    *,
    instance: Instance | None = None,
) -> EpisodeRecord:
    """Delete an agent's messages from a recorded episode and re-grade.

    No model calls: the counterfactual is evaluated purely against what was
    already said. That makes it both the cleanest block on compensation and free
    to run across the entire recorded corpus, which matters because it can be
    applied to every agent of every episode rather than a sampled subset.

    The limitation is the mirror of the strength -- surviving agents cannot react
    to the absence, so this is a lower bound on what the team would have managed.
    Read it alongside the live number, never instead of it.
    """
    inst = instance or generate(record.instance_seed, record.difficulty)
    kept = [
        m for m in record.messages if not (m.is_ablatable() and m.author == agent_id)
    ]
    solution = extract_solution(kept)

    return EpisodeRecord(
        episode_id=f"{AblationMode.FROZEN_EXCISE.value}:{agent_id}:{record.episode_id}",
        condition=f"{AblationMode.FROZEN_EXCISE.value}:{agent_id}",
        instance_seed=record.instance_seed,
        difficulty=record.difficulty,
        agents=[a for a in record.agents if a != agent_id],
        messages=kept,
        solution=solution,
        grade=grade(inst, solution),
        turn_order=record.turn_order,
        config=dict(record.config),
        meta={
            **record.meta,
            "ablated": agent_id,
            "source_episode": record.episode_id,
            "removed_messages": len(record.messages) - len(kept),
            "model_calls": 0,
        },
    )


def random_message_control(
    record: EpisodeRecord,
    agent_id: str,
    *,
    seed: int = 0,
    instance: Instance | None = None,
) -> EpisodeRecord:
    """Excise a volume-matched random subset of messages instead of one agent's.

    Separates "this agent's contribution mattered" from "there was simply less
    text in the context". Without it, any drop under frozen excision is
    confounded with the sheer loss of tokens.

    Messages are drawn from agents *other* than the nominal target so the control
    never accidentally reproduces the treatment.
    """
    inst = instance or generate(record.instance_seed, record.difficulty)
    n_remove = sum(
        1 for m in record.messages if m.is_ablatable() and m.author == agent_id
    )

    pool = [
        i
        for i, m in enumerate(record.messages)
        if m.is_ablatable() and m.author != agent_id
    ]
    rng = random.Random(seed ^ record.instance_seed)
    drop = set(rng.sample(pool, min(n_remove, len(pool))))

    kept = [m for i, m in enumerate(record.messages) if i not in drop]
    solution = extract_solution(kept)

    return EpisodeRecord(
        episode_id=f"{AblationMode.RANDOM_MESSAGE.value}:{agent_id}:{record.episode_id}",
        condition=f"{AblationMode.RANDOM_MESSAGE.value}:{agent_id}",
        instance_seed=record.instance_seed,
        difficulty=record.difficulty,
        agents=list(record.agents),
        messages=kept,
        solution=solution,
        grade=grade(inst, solution),
        turn_order=record.turn_order,
        config=dict(record.config),
        meta={
            **record.meta,
            "matched_to": agent_id,
            "source_episode": record.episode_id,
            "removed_messages": len(drop),
            "model_calls": 0,
        },
    )


async def capacity_control(
    *,
    backend: LLMBackend,
    config: TeamConfig,
    episode_seed: int,
    instance: Instance | None = None,
) -> EpisodeRecord:
    """Run a team that never had the extra agent in the first place.

    Separates specialization from headcount. If an N-1 team assembled from
    scratch performs like an N-team-minus-one, then the "drop" attributed to
    ablation is just the cost of one fewer worker and carries no information
    about roles.
    """
    smaller = replace(config, n_agents=max(1, config.n_agents - 1))
    return await run_episode(
        backend=backend,
        config=smaller,
        episode_seed=episode_seed,
        condition=f"{AblationMode.CAPACITY.value}:{smaller.n_agents}",
        instance=instance,
    )


async def frozen_replay(
    *,
    backend: LLMBackend,
    record: EpisodeRecord,
    config: TeamConfig,
    agent_id: str,
    instance: Instance | None = None,
) -> EpisodeRecord:
    """Regenerate the surviving agents' turns against the excised context.

    The recorded schedule is preserved -- the same agents speak in the same
    slots, minus the ablated one -- so compensation can happen within a turn but
    not by restructuring the episode. This sits between `frozen_excise` (no
    reaction at all) and `live_ablation` (full reorganization).

    Turns before the ablated agent's first contribution are reused verbatim,
    since their context is unchanged; regeneration starts at the first point
    where the transcript actually diverges. On a long episode that saves most of
    the model calls.
    """
    inst = instance or generate(record.instance_seed, record.difficulty)
    roster = {a.agent_id: a for a in build_team(config, record.instance_seed)}

    divergence = _first_divergence(record, agent_id)
    history: list[Message] = []
    calls = 0

    for index, msg in enumerate(record.messages):
        if msg.is_ablatable() and msg.author == agent_id:
            continue
        if index < divergence or not msg.is_ablatable():
            history.append(msg)
            continue

        agent = roster.get(msg.author)
        if agent is None:
            history.append(msg)
            continue

        request = GenRequest(
            messages=[
                ChatMessage(
                    role="system",
                    content=build_system_prompt(agent, inst, len(record.agents) - 1),
                ),
                *build_context(agent, history),
            ],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            seed=agent.seed,
            meta={
                "instance": inst.to_dict(),
                "agent_id": agent.agent_id,
                "agent_index": agent.index,
                "turn_position": int(msg.meta.get("position", 0)),
                "turn": msg.turn,
                "round": int(msg.meta.get("round", 0)),
            },
        )
        response = await backend.generate(request)
        calls += 1
        history.append(
            Message(
                turn=msg.turn,
                speaker=Speaker.AGENT,
                author=agent.agent_id,
                content=response.text,
                meta={**msg.meta, "regenerated": True},
            )
        )

    solution = extract_solution(history)
    return EpisodeRecord(
        episode_id=f"{AblationMode.FROZEN_REPLAY.value}:{agent_id}:{record.episode_id}",
        condition=f"{AblationMode.FROZEN_REPLAY.value}:{agent_id}",
        instance_seed=record.instance_seed,
        difficulty=record.difficulty,
        agents=[a for a in record.agents if a != agent_id],
        messages=history,
        solution=solution,
        grade=grade(inst, solution),
        turn_order=record.turn_order,
        config=config.to_dict(),
        meta={
            **record.meta,
            "ablated": agent_id,
            "source_episode": record.episode_id,
            "divergence_index": divergence,
            "model_calls": calls,
        },
    )


def _first_divergence(record: EpisodeRecord, agent_id: str) -> int:
    """Index of the ablated agent's first message, or past the end if silent."""
    for i, m in enumerate(record.messages):
        if m.is_ablatable() and m.author == agent_id:
            return i
    return len(record.messages)


def fungibility(live_drop: float, frozen_drop: float) -> float:
    """How much of an agent's contribution the team could supply itself.

    The gap between a compensation-blocked drop and a compensation-permitting
    one. Near zero means the contribution was irreplaceable; large means the role
    was real but any of them could have filled it.

    Pass `frozen_replay` rather than `frozen_excise` as `frozen_drop` unless
    `propagation_index` says restatement is low -- see below.
    """
    return frozen_drop - live_drop


def propagation_index(record: EpisodeRecord, agent_id: str) -> float:
    """How much of an agent's content is echoed by others later in the episode.

    This is the diagnostic that says whether `frozen_excise` can be trusted on a
    given corpus, and it exists because the naive assumption failed empirically.

    When agents restate the full working answer on every turn -- which small
    models do constantly, and which our mock does by construction -- an agent's
    contribution is duplicated into everyone else's messages almost as soon as it
    is made. Deleting that agent's own messages then removes almost nothing,
    because the content survives in the copies. Measured on the mock, excision
    drops were ~0.002 against live drops of 0.03-0.23: a hundredfold
    underestimate, and one that would have read as "this agent contributed
    nothing" rather than "this measurement does not work here".

    Returns the fraction of the agent's distinct content tokens that also appear
    in later messages by other agents. High values mean prefer `frozen_replay`;
    low values mean excision is measuring what it claims to.
    """
    own_turns = [m for m in record.messages if m.is_ablatable() and m.author == agent_id]
    if not own_turns:
        return 0.0

    echoed = 0
    total = 0
    for msg in own_turns:
        tokens = {t for t in msg.content.split() if len(t) > 3}
        if not tokens:
            continue
        later = " ".join(
            m.content
            for m in record.messages
            if m.is_ablatable() and m.author != agent_id and m.turn > msg.turn
        )
        later_tokens = set(later.split())
        echoed += len(tokens & later_tokens)
        total += len(tokens)

    return echoed / total if total else 0.0
