"""Ablation modes and their controls."""

from __future__ import annotations

import asyncio
import statistics

import pytest

from collabengine.ablation import (
    capacity_control,
    frozen_excise,
    frozen_replay,
    live_ablation,
    random_message_control,
)
from collabengine.ablation.modes import fungibility, propagation_index
from collabengine.backends import MockBackend, MockMode
from collabengine.orchestrator import SymmetryBreaking, TeamConfig, run_episode
from collabengine.tasks.generator import generate

CONFIG = TeamConfig(
    n_agents=4,
    rounds=3,
    difficulty="hard",
    symmetry=SymmetryBreaking.NAME_SEED,
    randomize_turn_order=True,
)


def _backend() -> MockBackend:
    return MockBackend(
        mode=MockMode.SPECIALIZED, competence=0.5, off_focus_competence=0.0
    )


async def _baseline(seed: int):
    return await run_episode(
        backend=_backend(), config=CONFIG, episode_seed=seed, condition="baseline"
    )


def test_frozen_excise_removes_only_that_agents_messages() -> None:
    record = asyncio.run(_baseline(1))
    out = frozen_excise(record, "A2")

    assert all(m.author != "A2" for m in out.messages if m.is_ablatable())
    # System and moderator scaffolding must survive: removing it would change
    # the task rather than one agent's contribution.
    assert any(not m.is_ablatable() for m in out.messages)
    assert out.meta["removed_messages"] == len(record.messages_by("A2"))


def test_frozen_excise_costs_no_model_calls() -> None:
    """The property that makes it affordable across the whole corpus."""
    record = asyncio.run(_baseline(2))
    assert frozen_excise(record, "A1").meta["model_calls"] == 0


def test_frozen_excise_is_deterministic() -> None:
    record = asyncio.run(_baseline(3))
    a = frozen_excise(record, "A3")
    b = frozen_excise(record, "A3")
    assert a.grade.to_dict() == b.grade.to_dict()


def test_excising_a_silent_agent_changes_nothing() -> None:
    record = asyncio.run(_baseline(4))
    out = frozen_excise(record, "A_NOT_PRESENT")
    assert out.grade.to_dict() == record.grade.to_dict()


def test_random_message_control_matches_volume() -> None:
    """The control must remove as much text as the treatment, or it proves nothing."""
    record = asyncio.run(_baseline(5))
    treated = frozen_excise(record, "A2")
    control = random_message_control(record, "A2", seed=7)

    assert control.meta["removed_messages"] == treated.meta["removed_messages"]
    # And it must never remove the nominal target's own messages.
    assert len(control.messages) == len(record.messages) - treated.meta["removed_messages"]


def test_capacity_control_runs_a_genuinely_smaller_team() -> None:
    record = asyncio.run(
        capacity_control(backend=_backend(), config=CONFIG, episode_seed=6)
    )
    assert len(record.agents) == CONFIG.n_agents - 1
    assert record.condition.startswith("capacity:")


def test_live_ablation_excludes_the_agent_for_the_whole_episode() -> None:
    record = asyncio.run(
        live_ablation(
            backend=_backend(), config=CONFIG, episode_seed=7, agent_id="A2"
        )
    )
    assert "A2" not in record.agents
    assert all(m.author != "A2" for m in record.messages if m.is_ablatable())


def test_frozen_replay_reuses_turns_before_divergence() -> None:
    """Regeneration starts where the transcript actually diverges, not before."""
    record = asyncio.run(_baseline(8))
    out = asyncio.run(
        frozen_replay(
            backend=_backend(), record=record, config=CONFIG, agent_id="A3"
        )
    )

    divergence = out.meta["divergence_index"]
    reused = [m for m in out.messages if not m.meta.get("regenerated")]
    assert divergence < len(record.messages)
    assert reused, "nothing was reused; the saving is being thrown away"
    assert out.meta["model_calls"] < len(record.messages)


def test_frozen_replay_never_contains_the_ablated_agent() -> None:
    record = asyncio.run(_baseline(9))
    out = asyncio.run(
        frozen_replay(
            backend=_backend(), record=record, config=CONFIG, agent_id="A1"
        )
    )
    assert all(m.author != "A1" for m in out.messages if m.is_ablatable())


def test_excision_underreports_when_content_propagates() -> None:
    """Regression guard on the propagation caveat.

    Excision was expected to give the largest drop, since it blocks compensation
    entirely. It gives nearly the smallest, because agents restate the working
    answer every turn and a contribution is copied into everyone else's messages
    as soon as it is made. Deleting the originator removes the words but not the
    content.

    Pinning the ordering here means that if a future change to the protocol
    stops agents restating -- which would make excision meaningful again -- this
    test fails loudly and the guidance gets revisited rather than silently
    remaining wrong.
    """

    async def main() -> None:
        excise_drops, replay_drops, live_drops = [], [], []
        for seed in range(12):
            base = await _baseline(seed)
            inst = generate(seed, CONFIG.difficulty)

            excised = frozen_excise(base, "A3", instance=inst)
            replayed = await frozen_replay(
                backend=_backend(),
                record=base,
                config=CONFIG,
                agent_id="A3",
                instance=inst,
            )
            live = await live_ablation(
                backend=_backend(),
                config=CONFIG,
                episode_seed=seed,
                agent_id="A3",
                instance=inst,
            )
            excise_drops.append(base.grade.overall - excised.grade.overall)
            replay_drops.append(base.grade.overall - replayed.grade.overall)
            live_drops.append(base.grade.overall - live.grade.overall)

        mean_excise = statistics.mean(excise_drops)
        mean_replay = statistics.mean(replay_drops)
        mean_live = statistics.mean(live_drops)

        assert mean_excise < mean_replay, (
            f"excision no longer underreports (excise={mean_excise:.4f}, "
            f"replay={mean_replay:.4f}) -- revisit the propagation guidance"
        )
        assert mean_replay > 0.01, "replay found no contribution at all"
        # Fungibility is computed against replay, not excision, for this reason.
        assert abs(fungibility(mean_live, mean_replay)) < 0.5

    asyncio.run(main())


def test_propagation_index_flags_restating_transcripts() -> None:
    """The diagnostic must actually detect the condition it warns about."""
    record = asyncio.run(_baseline(13))
    scores = [propagation_index(record, a) for a in record.agents]
    assert max(scores) > 0.3, (
        f"propagation went undetected on a transcript built from restatement: {scores}"
    )


def test_propagation_index_is_zero_for_absent_agent() -> None:
    record = asyncio.run(_baseline(14))
    assert propagation_index(record, "A_NOT_PRESENT") == 0.0


@pytest.mark.parametrize("agent", ["A1", "A2", "A3", "A4"])
def test_every_ablation_mode_produces_a_gradeable_record(agent: str) -> None:
    record = asyncio.run(_baseline(11))
    for out in (
        frozen_excise(record, agent),
        random_message_control(record, agent, seed=3),
    ):
        assert 0.0 <= out.grade.overall <= 1.0
        assert out.condition.endswith(agent)
