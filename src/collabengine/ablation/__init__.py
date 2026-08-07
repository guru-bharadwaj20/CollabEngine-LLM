"""Causal ablation -- the core contribution.

Removing an agent and re-running answers "is this agent necessary?". It does not
answer "was this agent's apparent role real?", because the survivors reorganize
and absorb the missing function. That compensation is documented in the emergent-
differentiation literature and is the single largest threat to the design.

So ablation is run three ways, and their differences are results in themselves:

  live           drop the agent from the roster and replay the whole episode.
                 Compensation allowed. Measures necessity of the agent.

  frozen_excise  take the recorded transcript, delete the agent's messages, and
                 re-read the answer from what remains. Costs no model calls, but
                 see the propagation caveat below -- it is a diagnostic, not the
                 primary measure.

  frozen_replay  keep the recorded turn structure but regenerate the surviving
                 agents' turns against the modified context. Compensation is
                 allowed within a turn slot but not across the schedule. This is
                 the primary frozen measure.

Delta(frozen_replay) - Delta(live) is the fungibility measure: how much of an
agent's contribution the rest of the team could have supplied itself.

The propagation caveat
----------------------
Plain excision was originally expected to give the *largest* drop, on the
reasoning that it blocks compensation entirely. Measured against the mock it
gave almost the smallest -- ~0.002 against live drops of 0.03 to 0.23.

The reason is content propagation. Agents restate the whole working answer on
every turn, so a contribution is duplicated into everyone else's messages almost
as soon as it is made. Deleting the originating messages then removes very
little, because the content survives in the copies. Read naively, that reports
"this agent contributed nothing" when the truth is "this measurement does not
work on this transcript".

`propagation_index` measures how much of an agent's content is echoed by others
later in the episode, and decides which frozen mode to trust. Run it before
reporting any excision-based number.
"""

from collabengine.ablation.modes import (
    AblationMode,
    capacity_control,
    frozen_excise,
    frozen_replay,
    live_ablation,
    random_message_control,
)

__all__ = [
    "AblationMode",
    "capacity_control",
    "frozen_excise",
    "frozen_replay",
    "live_ablation",
    "random_message_control",
]
