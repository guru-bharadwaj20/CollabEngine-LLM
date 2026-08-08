"""The mixed-effects interaction test.

Same standard as the rest of the instrument: it has to find structure where
structure was planted, and — the part that matters — fail to find it where none
was. A significance test that fires on a null world is worse than no test,
because it launders noise as a result.
"""

from __future__ import annotations

import random

import pytest

from collabengine.analysis.mixed import build_frame, fit_interaction
from collabengine.tasks.grader import GradeResult
from collabengine.tasks.schema import ALL_COMPONENTS, Solution
from collabengine.transcripts.store import EpisodeRecord

pytest.importorskip("statsmodels")
pytest.importorskip("pandas")

AGENTS = ["A1", "A2", "A3", "A4"]


def _record(seed: int, agent: str, per_component: dict) -> EpisodeRecord:
    return EpisodeRecord(
        episode_id=f"live:{agent}:hard:{seed}",
        condition=f"live:{agent}",
        instance_seed=seed,
        difficulty="hard",
        agents=[a for a in AGENTS if a != agent],
        messages=[],
        solution=Solution(malformed=True),
        grade=GradeResult(per_component=per_component, overall=0.0, satisfied={}),
    )


def _corpus(*, specialized: bool, n_episodes: int = 25, seed: int = 0):
    """Ablation records with, or without, an agent x component interaction.

    Both worlds carry the same two main effects -- some components are harder,
    and each episode has its own difficulty -- so a test that fires on the null
    world is picking up a main effect it was supposed to remove.
    """
    rng = random.Random(seed)
    records = []
    for episode in range(n_episodes):
        episode_effect = rng.gauss(0, 0.15)  # shared instance difficulty
        for i, agent in enumerate(AGENTS):
            per_component = {}
            for j, component in enumerate(ALL_COMPONENTS):
                score = 0.6 + episode_effect - 0.05 * j + rng.gauss(0, 0.05)
                if specialized and i == j:
                    score -= 0.35  # this agent owned this component
                per_component[component] = max(0.0, min(1.0, score))
            records.append(_record(episode, agent, per_component))
    return records


def test_frame_has_one_row_per_agent_component_observation() -> None:
    frame = build_frame(_corpus(specialized=True, n_episodes=3))
    assert len(frame) == 3 * len(AGENTS) * len(ALL_COMPONENTS)
    assert set(frame.columns) == {"episode", "agent", "component", "score"}


def test_frozen_modes_are_excluded_from_the_fit() -> None:
    """Frozen records are derived from a recorded transcript, not independently
    sampled, so counting them would inflate n without adding information."""
    records = _corpus(specialized=True, n_episodes=2)
    derived = _record(0, "A1", dict.fromkeys(ALL_COMPONENTS, 0.5))
    derived.condition = "frozen_replay:A1"
    frame = build_frame([*records, derived])

    assert len(frame) == 2 * len(AGENTS) * len(ALL_COMPONENTS)


def test_planted_interaction_is_detected() -> None:
    report = fit_interaction(_corpus(specialized=True))
    assert report.converged
    assert report.significant
    assert report.interaction_p < 0.01


def test_null_world_is_not_reported_as_significant() -> None:
    """The load-bearing test. Main effects are present in both worlds; only the
    interaction differs, and only the interaction may drive the p-value."""
    report = fit_interaction(_corpus(specialized=False, seed=7))
    assert report.converged
    assert not report.significant


def test_episode_variance_is_absorbed_by_the_random_effect() -> None:
    """The instance-difficulty effect is what makes rows non-independent; if the
    model did not absorb it, the standard errors would be too small."""
    report = fit_interaction(_corpus(specialized=False, seed=3))
    assert report.group_variance is not None
    assert report.group_variance > 0


def test_missing_records_report_rather_than_raise() -> None:
    report = fit_interaction([])
    assert report.interaction_p is None
    assert not report.converged
    assert "no live-ablation records" in report.note


def test_single_agent_cannot_support_an_interaction() -> None:
    records = [_record(s, "A1", dict.fromkeys(ALL_COMPONENTS, 0.5)) for s in range(5)]
    report = fit_interaction(records)
    assert report.interaction_p is None
    assert "at least two agents" in report.note
