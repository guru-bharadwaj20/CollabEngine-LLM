"""Episode records and their on-disk form.

Transcripts are the primary artifact of the project. Phase 3 replays them for
frozen-transcript ablation and Phase 4 re-codes them for behavioral labels, so
every episode is stored complete enough to reconstruct without re-running the
model: the instance is recoverable from (seed, difficulty), and every message
carries its author and turn.
"""

from collabengine.transcripts.store import (
    EpisodeRecord,
    TranscriptReader,
    TranscriptWriter,
    merge_shards,
)

__all__ = [
    "EpisodeRecord",
    "TranscriptReader",
    "TranscriptWriter",
    "merge_shards",
]
