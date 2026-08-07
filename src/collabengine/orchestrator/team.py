"""Team composition and the symmetry-breaking axis.

Identical weights, identical prompt, identical context produce near-identical
outputs -- and then there is nothing to specialize. Something must break the
symmetry between agents. That "something" is not an implementation detail to be
chosen once and forgotten; it is the independent variable of Phase 2, so it is
modelled explicitly and swept.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SymmetryBreaking(str, Enum):
    """How much the agents are allowed to differ at the start.

    The scientific question is whether minimal asymmetry amplifies into stable
    roles, so the levels are ordered from least to most differentiating.
    """

    NAME_ONLY = "name_only"
    """Agents differ only by an opaque label. Shared sampling seed."""

    NAME_SEED = "name_seed"
    """Distinct label and distinct sampling seed."""

    NAME_SEED_SCRATCH = "name_seed_scratch"
    """Adds a private scratchpad line unique to each agent."""


@dataclass(frozen=True, slots=True)
class Agent:
    agent_id: str
    index: int
    seed: int
    scratch: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "index": self.index,
            "seed": self.seed,
            "scratch": self.scratch,
        }


@dataclass(slots=True)
class TeamConfig:
    n_agents: int = 4
    rounds: int = 3
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    symmetry: SymmetryBreaking = SymmetryBreaking.NAME_SEED
    randomize_turn_order: bool = True
    """Shuffle speaking order every round.

    Without this, roles can attach to turn position rather than agent identity
    and the whole emergence result collapses into a protocol artifact (confound
    C2). Randomizing is what lets the analysis tell those apart."""
    difficulty: str = "medium"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_agents": self.n_agents,
            "rounds": self.rounds,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "symmetry": self.symmetry.value,
            "randomize_turn_order": self.randomize_turn_order,
            "difficulty": self.difficulty,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TeamConfig:
        return cls(
            n_agents=int(d.get("n_agents", 4)),
            rounds=int(d.get("rounds", 3)),
            max_tokens=int(d.get("max_tokens", 512)),
            temperature=float(d.get("temperature", 0.8)),
            top_p=float(d.get("top_p", 0.95)),
            symmetry=SymmetryBreaking(d.get("symmetry", "name_seed")),
            randomize_turn_order=bool(d.get("randomize_turn_order", True)),
            difficulty=d.get("difficulty", "medium"),
            extra=dict(d.get("extra", {})),
        )


# Neutral labels. Not "Planner"/"Critic" -- naming a function would assign the
# role rather than let one emerge, which is the entire thing under test.
AGENT_LABELS: tuple[str, ...] = tuple(f"A{i + 1}" for i in range(26))

_SCRATCH_NOTES: tuple[str, ...] = (
    "You tend to write things down before deciding.",
    "You tend to re-read the requirements before agreeing.",
    "You tend to look for the quickest workable answer.",
    "You tend to check totals twice.",
    "You tend to ask what has not been covered yet.",
    "You tend to summarise where the group has got to.",
)


def build_team(config: TeamConfig, episode_seed: int) -> list[Agent]:
    """Construct the roster for one episode.

    Scratch notes are intentionally vague dispositions rather than task roles:
    they perturb behavior without naming a component, so any component-level
    specialization that appears is still emergent rather than instructed.
    """
    if config.n_agents < 1:
        raise ValueError("n_agents must be >= 1")
    if config.n_agents > len(AGENT_LABELS):
        raise ValueError(f"n_agents must be <= {len(AGENT_LABELS)}")

    agents: list[Agent] = []
    for i in range(config.n_agents):
        if config.symmetry is SymmetryBreaking.NAME_ONLY:
            seed = episode_seed
            scratch = ""
        elif config.symmetry is SymmetryBreaking.NAME_SEED:
            seed = episode_seed * 1000 + i
            scratch = ""
        else:
            seed = episode_seed * 1000 + i
            scratch = _SCRATCH_NOTES[i % len(_SCRATCH_NOTES)]
        agents.append(
            Agent(agent_id=AGENT_LABELS[i], index=i, seed=seed, scratch=scratch)
        )
    return agents
