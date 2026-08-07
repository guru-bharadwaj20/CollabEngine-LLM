"""Message protocol shared by the orchestrator, transcripts, and ablation.

Messages are the unit that frozen-transcript ablation excises, so every message
records which agent authored it and at which turn. Both fields are load-bearing:
without `author` we cannot remove one agent's contribution, and without `turn`
we cannot replay the episode deterministically up to a cut point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Speaker(str, Enum):
    SYSTEM = "system"
    """Task brief and protocol instructions. Never ablated."""

    AGENT = "agent"
    """A team member's contribution. The ablation target."""

    MODERATOR = "moderator"
    """Harness-generated scaffolding (round announcements, final-answer call)."""


@dataclass(slots=True)
class Message:
    turn: int
    speaker: Speaker
    author: str
    """Agent id for AGENT messages; "system"/"moderator" otherwise."""
    content: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn": self.turn,
            "speaker": self.speaker.value,
            "author": self.author,
            "content": self.content,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Message:
        return cls(
            turn=int(d["turn"]),
            speaker=Speaker(d["speaker"]),
            author=d["author"],
            content=d["content"],
            meta=dict(d.get("meta", {})),
        )

    def is_ablatable(self) -> bool:
        """Only agent-authored messages may be excised.

        Removing system or moderator messages would change the task itself
        rather than one agent's contribution, confounding the ablation.
        """
        return self.speaker is Speaker.AGENT
