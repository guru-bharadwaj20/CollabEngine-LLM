"""Validation of the measurement instrument against known ground truth.

This is the most important test file in the project. Everything else checks that
code does what it says; this checks that the *experiment* can tell truth from
artifact. Using MockBackend we construct worlds whose answer we already know:

  SPECIALIZED  agents genuinely own one component each
  NULL         no division of labor exists at all
  POSITIONAL   behavior tracks turn slot, not agent identity

A pipeline that reports specialization in the NULL world is manufacturing its
own result. Finding that out here costs seconds; finding it out after a GPU run
costs the study.

The statistic under test is diagonal dominance of the double-centered ablation
matrix -- see `collabengine.analysis.interaction` for why a row-wise reading
("which component fell most when we removed agent i") is the wrong question.
"""

from __future__ import annotations

import asyncio
import statistics

import pytest

from collabengine.analysis import AblationMatrix, analyze_interaction
from collabengine.backends import MockBackend, MockMode
from collabengine.orchestrator import SymmetryBreaking, TeamConfig, run_episode
from collabengine.tasks.schema import ALL_COMPONENTS, Component

N_EPISODES = 40
DIFFICULTY = "hard"
AGENTS = ("A1", "A2", "A3", "A4")

# MockBackend.SPECIALIZED assigns ALL_COMPONENTS[i] to agent index i.
OWNERSHIP: dict[Component, str] = {
    comp: AGENTS[i] for i, comp in enumerate(ALL_COMPONENTS)
}


def _config(*, randomize: bool) -> TeamConfig:
    return TeamConfig(
        n_agents=4,
        rounds=3,
        difficulty=DIFFICULTY,
        symmetry=SymmetryBreaking.NAME_SEED,
        randomize_turn_order=randomize,
    )


def _backend(mode: MockMode) -> MockBackend:
    # off_focus 0.0 keeps the ground truth crisp: a non-owner contributes
    # nothing to a component, so any drop we measure is attributable.
    return MockBackend(mode=mode, competence=0.5, off_focus_competence=0.0)


async def _component_means(
    mode: MockMode, exclude: tuple[str, ...], *, randomize: bool
) -> dict[Component, float]:
    backend = _backend(mode)
    config = _config(randomize=randomize)
    acc: dict[Component, list[float]] = {c: [] for c in ALL_COMPONENTS}
    for seed in range(N_EPISODES):
        rec = await run_episode(
            backend=backend,
            config=config,
            episode_seed=seed,
            exclude=exclude,
            condition="test",
        )
        for comp, val in rec.grade.per_component.items():
            acc[comp].append(val)
    return {c: statistics.mean(v) for c, v in acc.items()}


async def _ablation_matrix(mode: MockMode, *, randomize: bool) -> AblationMatrix:
    baseline = await _component_means(mode, (), randomize=randomize)
    ablated = {
        agent: await _component_means(mode, (agent,), randomize=randomize)
        for agent in AGENTS
    }
    return AblationMatrix.from_means(baseline, ablated, n_episodes=N_EPISODES)


def _report(mode: MockMode, *, randomize: bool = True):
    matrix = asyncio.run(_ablation_matrix(mode, randomize=randomize))
    return analyze_interaction(matrix, OWNERSHIP)


@pytest.fixture(scope="module")
def specialized():
    return _report(MockMode.SPECIALIZED)


@pytest.fixture(scope="module")
def positional():
    return _report(MockMode.POSITIONAL)


@pytest.fixture(scope="module")
def null_world():
    return _report(MockMode.NULL)


def test_specialized_world_shows_diagonal_dominance(specialized) -> None:
    """The Phase 3 result in miniature.

    Each component's worst ablation should be the agent that owns it. Chance is
    0.25 with four agents; we require a clear majority.
    """
    assert specialized.diagonal_dominance >= 0.75, (
        f"ownership failed to predict causal damage: "
        f"{specialized.column_argmax} vs owners {OWNERSHIP}"
    )


def test_null_world_has_weak_interaction(null_world, specialized) -> None:
    """The negative control.

    Undifferentiated agents must not produce a specialization signature. If this
    fails, the instrument invents structure and every positive finding
    downstream is suspect.
    """
    assert null_world.interaction_strength < specialized.interaction_strength, (
        f"null world interaction ({null_world.interaction_strength:.4f}) is not "
        f"weaker than specialized ({specialized.interaction_strength:.4f})"
    )


def test_positional_world_fails_ownership_prediction(positional, specialized) -> None:
    """Confound C2, and the demonstration that order randomization controls it.

    Under a fixed speaking order, position and identity are perfectly
    confounded -- removing the agent in slot k removes slot k's function either
    way. Randomizing order per round breaks the tie: a positional agent occupies
    different slots across rounds, so no single agent owns a component and
    ownership stops predicting the column max.
    """
    assert positional.diagonal_dominance < specialized.diagonal_dominance, (
        f"positional world matched identity-bound dominance: "
        f"{positional.diagonal_dominance:.2f} vs {specialized.diagonal_dominance:.2f}"
    )


def test_randomizing_order_drives_positional_dominance_to_chance(positional) -> None:
    """Why randomization is not optional, stated precisely.

    A fixed speaking order does not confound position with identity *totally*,
    because ablation itself shifts every later agent's slot. But it leaves the
    positional world scoring meaningfully above chance on ownership prediction
    (0.50 against a 0.25 baseline), which is enough to be mistaken for genuine
    specialization.

    Randomizing order per round removes the residual: a positional agent
    occupies a different slot each round, so ownership stops predicting the
    column max and dominance falls to chance. Meanwhile the specialized world is
    unaffected, because there the role travels with the agent.
    """
    pos_fixed = _report(MockMode.POSITIONAL, randomize=False)
    spec_fixed = _report(MockMode.SPECIALIZED, randomize=False)

    assert positional.diagonal_dominance <= pos_fixed.diagonal_dominance, (
        "randomization should reduce, not increase, spurious dominance: "
        f"{positional.diagonal_dominance:.2f} vs fixed {pos_fixed.diagonal_dominance:.2f}"
    )
    assert positional.diagonal_dominance <= positional.chance_level + 1e-9, (
        f"positional dominance {positional.diagonal_dominance:.2f} still exceeds "
        f"chance {positional.chance_level:.2f} under randomized order"
    )
    assert spec_fixed.diagonal_dominance == 1.0, (
        "identity-bound roles should survive a fixed order too"
    )


def test_interaction_strength_separates_worlds_by_an_order_of_magnitude(
    specialized, positional, null_world
) -> None:
    """Strength, not just dominance, must discriminate.

    Dominance is a rank statistic and saturates at 1.0, so it cannot express
    *how* specialized a team is. Interaction strength (RMS of the double-centered
    residuals) is the continuous companion, and it needs a wide margin over the
    no-specialization worlds to be usable as an effect size on real models.
    """
    floor = max(positional.interaction_strength, null_world.interaction_strength)
    assert specialized.interaction_strength > 5 * floor, (
        f"specialized={specialized.interaction_strength:.4f} is not clearly above "
        f"positional={positional.interaction_strength:.4f} / "
        f"null={null_world.interaction_strength:.4f}"
    )


def test_scores_have_usable_variance() -> None:
    """Saturated scores make effect sizes undefined.

    If the operating point drifts to a ceiling or floor the statistical tests
    silently lose power rather than failing loudly, so assert the band here
    where it is cheap to notice.
    """

    async def main() -> None:
        backend = _backend(MockMode.SPECIALIZED)
        config = _config(randomize=True)
        overall = []
        for seed in range(N_EPISODES):
            rec = await run_episode(
                backend=backend, config=config, episode_seed=seed, condition="test"
            )
            overall.append(rec.grade.overall)

        assert 0.05 < statistics.mean(overall) < 0.98, "operating point saturated"
        assert statistics.pstdev(overall) > 0.01, "no variance to test against"

    asyncio.run(main())


def test_inert_world_floors_the_score() -> None:
    """Sanity floor: agents that never contribute must not score like a team."""

    async def main() -> None:
        inert = await _component_means(MockMode.INERT, (), randomize=True)
        working = await _component_means(MockMode.SPECIALIZED, (), randomize=True)
        assert statistics.mean(inert.values()) < statistics.mean(working.values())

    asyncio.run(main())
