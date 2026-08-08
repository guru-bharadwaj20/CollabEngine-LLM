"""Behavioral coding and convergent validity.

The tests that matter here are the ones that check the instrument does *not*
find structure in worlds that have none. A coding pipeline that reports
differentiation in a team of interchangeable agents, or a convergence statistic
that correlates two matrices of noise, would manufacture the study's result --
the same failure mode `selftest` guards against on the causal side.
"""

from __future__ import annotations

import asyncio
import random

import numpy as np
import pytest

from collabengine.analysis.coding import (
    ActionType,
    MessageCode,
    action_distributions,
    code_episode,
    cohens_kappa,
    differentiation_vs_null,
    js_divergence,
    mean_pairwise_divergence,
    ownership_from_codes,
    parse_action,
    role_consistency,
    summarize,
    _strip_identity,
)
from collabengine.analysis.convergent import (
    behavioral_matrix,
    convergent_validity,
)
from collabengine.analysis.interaction import AblationMatrix
from collabengine.backends.base import GenResponse, LLMBackend
from collabengine.tasks.schema import ALL_COMPONENTS, Component


def _code(agent: str, action: ActionType, episode: str = "e1", turn: int = 0):
    return MessageCode(episode_id=episode, turn=turn, agent_id=agent, action=action)


def _specialized_codes(n_episodes: int = 8) -> list[MessageCode]:
    """Four agents, each doing exactly one thing, forever."""
    owners = {
        "A1": ActionType.COMPUTE,
        "A2": ActionType.SEARCH,
        "A3": ActionType.VERIFY,
        "A4": ActionType.SYNTHESIZE,
    }
    return [
        _code(agent, action, episode=f"e{e}", turn=t)
        for e in range(n_episodes)
        for t, (agent, action) in enumerate(owners.items())
    ]


def _null_codes(n_episodes: int = 8, seed: int = 0) -> list[MessageCode]:
    """Four agents drawing from one shared distribution."""
    rng = random.Random(seed)
    pool = [
        ActionType.COMPUTE,
        ActionType.SEARCH,
        ActionType.VERIFY,
        ActionType.SYNTHESIZE,
    ]
    return [
        _code(f"A{a + 1}", rng.choice(pool), episode=f"e{e}", turn=t)
        for e in range(n_episodes)
        for t in range(8)
        for a in [rng.randrange(4)]
    ]


# ------------------------------------------------------------------- taxonomy


def test_parse_action_reads_a_bare_label() -> None:
    assert parse_action("verify") is ActionType.VERIFY
    assert parse_action("  Compute\n") is ActionType.COMPUTE


def test_parse_action_survives_a_chatty_judge() -> None:
    assert parse_action("The label is: synthesize.") is ActionType.SYNTHESIZE


def test_parse_action_refuses_to_guess() -> None:
    assert parse_action("I am not sure about this one") is None


def test_identity_prefix_is_stripped_before_the_judge_sees_it() -> None:
    """The judge must not be able to track who is speaking across turns."""
    assert "A2" not in _strip_identity("(A2) I checked the totals and J4 is wrong.")


# -------------------------------------------------------------------- metrics


def test_divergence_is_zero_for_identical_profiles() -> None:
    p = {ActionType.COMPUTE: 0.5, ActionType.VERIFY: 0.5}
    assert js_divergence(p, dict(p)) == pytest.approx(0.0, abs=1e-12)


def test_divergence_is_one_for_disjoint_profiles() -> None:
    a = {ActionType.COMPUTE: 1.0}
    b = {ActionType.VERIFY: 1.0}
    assert js_divergence(a, b) == pytest.approx(1.0)


def test_distributions_normalize_within_agent() -> None:
    codes = [_code("A1", ActionType.COMPUTE), _code("A1", ActionType.VERIFY)]
    dist = action_distributions(codes)["A1"]
    assert sum(dist.values()) == pytest.approx(1.0)
    assert dist[ActionType.COMPUTE] == pytest.approx(0.5)


def test_specialized_agents_are_maximally_differentiated() -> None:
    dists = action_distributions(_specialized_codes())
    assert mean_pairwise_divergence(dists) == pytest.approx(1.0)


def test_specialized_world_beats_the_permutation_null() -> None:
    observed, null_mean, p = differentiation_vs_null(
        _specialized_codes(), n_permutations=200
    )
    assert observed > null_mean
    assert p < 0.05


def test_null_world_does_not_beat_its_own_null() -> None:
    """The load-bearing test: interchangeable agents must not read as specialized.

    Four agents drawing from one distribution still show nonzero pairwise
    divergence from finite samples alone, so comparing to zero would report
    specialization here. Comparing to the permutation null must not."""
    observed, null_mean, p = differentiation_vs_null(_null_codes(), n_permutations=200)
    assert observed == pytest.approx(null_mean, abs=0.15)
    assert p > 0.05


# --------------------------------------------------------------------- kappa


def test_kappa_is_one_when_judges_agree_completely() -> None:
    labels = [ActionType.COMPUTE, ActionType.VERIFY, ActionType.SEARCH]
    assert cohens_kappa(labels, list(labels)) == pytest.approx(1.0)


def test_kappa_is_zero_for_chance_agreement_on_a_dominant_class() -> None:
    """Two judges agreeing 68% of the time while sharing no signal.

    Both use "propose" for 20 of 25 messages, so 68% agreement is exactly what
    independence predicts. Raw agreement would call that substantial; kappa
    correctly calls it nothing."""
    P, V = ActionType.PROPOSE, ActionType.VERIFY
    a = [P] * 20 + [V] * 5
    b = [P] * 16 + [V] * 4 + [P] * 4 + [V]

    raw = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    assert raw == pytest.approx(0.68)
    assert cohens_kappa(a, b) == pytest.approx(0.0, abs=1e-9)


def test_kappa_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same messages"):
        cohens_kappa([ActionType.COMPUTE], [])


# ----------------------------------------------------------------- ownership


def test_ownership_recovers_the_planted_specialists() -> None:
    owners = ownership_from_codes(_specialized_codes())
    assert owners[Component.ARITHMETIC] == "A1"
    assert owners[Component.SEARCH] == "A2"
    assert owners[Component.VERIFICATION] == "A3"
    assert owners[Component.SYNTHESIS] == "A4"


def test_ownership_uses_share_not_volume() -> None:
    """A verbose generalist must not own every component."""
    codes = [_code("A1", ActionType.COMPUTE, turn=t) for t in range(50)]
    codes += [_code("A1", ActionType.VERIFY, turn=100 + t) for t in range(50)]
    codes += [_code("A2", ActionType.VERIFY, turn=200 + t) for t in range(4)]

    owners = ownership_from_codes(codes)
    assert owners[Component.ARITHMETIC] == "A1"
    # A1 emitted 50 verify messages to A2's 4, but verification is all A2 does.
    assert owners[Component.VERIFICATION] == "A2"


def test_role_consistency_is_high_for_stable_agents() -> None:
    assert role_consistency(_specialized_codes()) == pytest.approx(1.0)


def test_summarize_reports_everything_the_phase_2_gate_needs() -> None:
    report = summarize(_specialized_codes(), n_permutations=100).to_dict()
    assert report["n_messages"] > 0
    assert report["p_value"] < 0.05
    assert report["ownership"]["arithmetic"] == "A1"


# ------------------------------------------------------------ convergent (P4)


def _matrix(values) -> AblationMatrix:
    return AblationMatrix(
        agents=["A1", "A2", "A3", "A4"],
        components=list(ALL_COMPONENTS),
        values=np.array(values, dtype=float),
    )


def test_behavioral_matrix_lines_up_with_the_ablation_grid() -> None:
    agents, components, values = behavioral_matrix(
        _specialized_codes(), agents=["A1", "A2", "A3", "A4"]
    )
    assert agents == ["A1", "A2", "A3", "A4"]
    assert components == list(ALL_COMPONENTS)
    assert values.shape == (4, 4)
    # Each agent does exactly one thing, so the matrix is the identity.
    assert np.allclose(values, np.eye(4))


def test_convergence_is_strong_when_ablation_confirms_the_labels() -> None:
    """Agents whose coded specialty is also what breaks when they leave."""
    codes = _specialized_codes()
    causal = _matrix(
        [
            [0.4, 0.1, 0.1, 0.1],
            [0.1, 0.4, 0.1, 0.1],
            [0.1, 0.1, 0.4, 0.1],
            [0.1, 0.1, 0.1, 0.4],
        ]
    )
    report = convergent_validity(codes, causal, n_permutations=200)
    assert report.r > 0.9
    assert report.p_value < 0.05
    assert "strongly" in report.verdict


def test_convergence_is_absent_when_labels_point_at_the_wrong_agents() -> None:
    """The published-either-way case: real roles, real ablation, no agreement."""
    codes = _specialized_codes()
    # Ablation says the owners are rotated by one relative to the transcript.
    causal = _matrix(
        [
            [0.1, 0.4, 0.1, 0.1],
            [0.1, 0.1, 0.4, 0.1],
            [0.1, 0.1, 0.1, 0.4],
            [0.4, 0.1, 0.1, 0.1],
        ]
    )
    report = convergent_validity(codes, causal, n_permutations=200)
    assert report.r < 0.0
    assert "strongly" not in report.verdict


def test_a_flat_ablation_matrix_correlates_with_nothing() -> None:
    """No nan, no spurious r: a constant matrix means no association."""
    report = convergent_validity(
        _specialized_codes(), _matrix(np.full((4, 4), 0.2)), n_permutations=100
    )
    assert report.r == 0.0
    assert report.p_value > 0.05
    assert "not predict" in report.verdict


def test_verbosity_main_effect_does_not_create_convergence() -> None:
    """A talkative agent hurts every component when removed and dominates its
    behavioral row. Without double-centering that alone would read as
    convergent validity in a team with no division of labor."""
    codes = [
        _code("A1", a, episode=f"e{e}", turn=t)
        for e in range(6)
        for t, a in enumerate(
            [ActionType.COMPUTE, ActionType.SEARCH, ActionType.VERIFY] * 4
        )
    ]
    codes += [
        _code(f"A{i}", ActionType.COMPUTE, episode=f"e{e}", turn=99)
        for i in (2, 3, 4)
        for e in range(6)
    ]
    causal = _matrix(
        [
            [0.5, 0.5, 0.5, 0.5],  # A1 removal hurts everything equally
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.1],
        ]
    )
    report = convergent_validity(codes, causal, n_permutations=200)
    assert report.p_value > 0.05


# ------------------------------------------------------------------ coding IO


class ScriptedJudge(LLMBackend):
    """Replays a fixed sequence of labels; records what it was shown."""

    name = "scripted"

    def __init__(self, labels, fail: bool = False):
        self.labels = list(labels)
        self.seen: list[str] = []
        self.fail = fail

    async def generate(self, request):
        self.seen.append(request.messages[-1].content)
        if self.fail:
            return GenResponse(text="", finish_reason="error", error="judge down")
        return GenResponse(text=self.labels[(len(self.seen) - 1) % len(self.labels)])


async def _record():
    from collabengine.orchestrator import run_episode
    from collabengine.orchestrator.team import TeamConfig
    from collabengine.backends.mock import MockBackend, MockMode

    return await run_episode(
        backend=MockBackend(mode=MockMode.SPECIALIZED),
        config=TeamConfig(n_agents=3, rounds=2, difficulty="tiny"),
        episode_seed=1,
    )


async def test_code_episode_labels_every_agent_turn() -> None:
    record = await _record()
    judge = ScriptedJudge(["verify", "compute", "search"])

    codes = await code_episode(backend=judge, record=record, judge_name="j1")

    n_agent_turns = sum(1 for m in record.messages if m.speaker.value == "agent")
    assert len(codes) == n_agent_turns
    assert {c.judge for c in codes} == {"j1"}
    assert {c.agent_id for c in codes} <= set(record.agents)


async def test_the_judge_never_sees_the_agent_label() -> None:
    record = await _record()
    judge = ScriptedJudge(["verify"])
    await code_episode(backend=judge, record=record)

    for shown in judge.seen:
        for agent in record.agents:
            assert f"({agent})" not in shown


async def test_a_failing_judge_is_counted_not_silently_labelled() -> None:
    from collabengine.analysis.coding import CodingStats

    record = await _record()
    stats = CodingStats()
    codes = await code_episode(
        backend=ScriptedJudge([], fail=True), record=record, stats=stats
    )

    assert stats.errors == len(codes)
    assert stats.coded == 0
    assert all(c.action is ActionType.OTHER for c in codes)
