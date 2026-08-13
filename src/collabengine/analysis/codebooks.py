"""Codebook variants for the judge-validity sweep.

The judge agrees with a human at kappa = 0.241 under codebook v2, which is not
usable for Phase 2 (RESEARCH-LOG 4.13b). v2 already fixed the one *definitional*
defect -- `organize` colliding with a task whose content is dividing up work --
so what is left is ordinary confusion among the three labels that carry the
work: a message that revises an assignment *because* it found a violation is
genuinely both `propose` and `verify`.

Three things could fix that, and they are cheap to test because the validation
sample is 40 messages:

  v4  say which way the ambiguous cases resolve, the way v2 did for `organize`
  v5  the same, plus worked examples -- show rather than tell
  v3  stop asking for a distinction the judge cannot hold, and collapse the
      taxonomy to the one Phase 2 actually found robust (4.8: teams generate,
      lone agents audit)

**Comparing kappa across these is not apples to apples.** v3 has three
categories where v2 has eight, and collapsing categories raises agreement
mechanically -- kappa corrects for chance agreement, not for the fact that a
coarser question is an easier question. A v3 kappa of 0.6 buys a coarser Phase 2,
not the original one. The sweep prints the number of retained distinctions beside
every kappa for that reason.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from collabengine.analysis.coding import JUDGE_SYSTEM


@dataclass(frozen=True, slots=True)
class Codebook:
    """One judge prompt plus the label space it is allowed to emit."""

    version: int
    name: str
    system: str
    labels: tuple[str, ...]
    #: Maps the eight-label human coding into this book's space. `None` means the
    #: spaces are identical. Applied to *both* raters or it would be measuring
    #: the mapping rather than the judge.
    collapse: dict[str, str] | None = None

    def project(self, label: str) -> str:
        if self.collapse is None:
            return label
        return self.collapse.get(label, label)


#: Shared tail. Kept in one place so a variant differs from v2 only where it
#: means to -- an accidental wording drift between arms would be read as the
#: effect of the change under test.
_REPLY = "\n\nReply with the label and nothing else."

_BOUNDARIES = (
    "\n\nThese three overlap constantly. Resolve them in this order:\n"
    "1. If the message states or restates an assignment of jobs to workers, it "
    "is propose -- even when the reason given is that it found an error, and "
    "even when arithmetic appears alongside it. A fix is a proposal.\n"
    "2. If it does not state an assignment but checks one that already exists "
    "-- recomputing a total to test it, naming a violated requirement, saying "
    "something does or does not hold -- it is verify.\n"
    "3. If it does not state an assignment and is not checking one, but is "
    "trying placements out to see what would work, it is search.\n"
    "compute is for arithmetic that is not in service of either: totals worked "
    "through for their own sake."
)

_EXAMPLES = (
    "\n\nWorked examples:\n"
    "'W3 is over capacity at 11 hours, so move J7 to W5 instead.'\n"
    "  -> propose (it found an error, but what it DOES is reassign)\n"
    "'W3 has J2 and J7, which is 6 + 5 = 11 hours against a limit of 9.'\n"
    "  -> verify (checks the draft, states no new assignment)\n"
    "'What if we put J7 on W5? W5 has 4 hours free, so that might fit.'\n"
    "  -> search (tries a placement out, does not commit to it)\n"
    "'Totals: W1 8h, W2 7h, W3 11h, W4 5h.'\n"
    "  -> compute (arithmetic, no claim about correctness)\n"
    "'A2, you take the capacity checks and I will redo the values.'\n"
    "  -> organize (coordinates the participants, not the jobs)"
)

_COARSE = (
    "You are coding messages from a group problem-solving transcript. You will "
    "see one message with no information about who wrote it. Choose the single "
    "label that best describes what the message primarily DOES.\n\n"
    "The task in the transcript is an ALLOCATION PUZZLE: participants assign "
    "jobs (J1, J2, ...) to workers (W1, W2, ...) subject to skill, capacity and "
    "exclusion requirements.\n\n"
    "solve  - moves the answer forward: states or revises an assignment, tries "
    "a placement out, or works through the arithmetic to get there\n"
    "check  - audits work that already exists: recomputes a total to test it, "
    "names a violated requirement, or says whether something holds\n"
    "meta   - anything that is not about the puzzle's content: coordinating who "
    "does what, agreeing without adding anything, or none of the above\n\n"
    "If a message both checks and then reassigns, label it solve -- the "
    "reassignment is what it leaves behind." + _REPLY
)

#: The generate/audit split is not an arbitrary coarsening. It is the one
#: behavioural result Phase 2 found that survived its own null tests: propose as
#: a share of propose+verify runs 0.674 for teams against 0.403 for solo,
#: p < 0.0001 (RESEARCH-LOG 4.8). If the judge can hold any distinction, this is
#: the one worth keeping, because it is the one that carried a finding.
_TO_COARSE = {
    "propose": "solve",
    "compute": "solve",
    "search": "solve",
    "synthesize": "solve",
    "verify": "check",
    "organize": "meta",
    "agree": "meta",
    "other": "meta",
}

_EIGHT = (
    "propose", "compute", "search", "verify",
    "synthesize", "organize", "agree", "other",
)

CODEBOOKS: tuple[Codebook, ...] = (
    Codebook(2, "v2 control", JUDGE_SYSTEM, _EIGHT),
    Codebook(
        4, "v4 boundaries",
        JUDGE_SYSTEM.replace(_REPLY, _BOUNDARIES + _REPLY), _EIGHT,
    ),
    Codebook(
        5, "v5 boundaries+examples",
        JUDGE_SYSTEM.replace(_REPLY, _BOUNDARIES + _EXAMPLES + _REPLY), _EIGHT,
    ),
    Codebook(3, "v3 coarse", _COARSE, ("solve", "check", "meta"), _TO_COARSE),
)


def parse_label(text: str, book: Codebook) -> str | None:
    """Recover a label from the judge's reply within one book's label space.

    Longest-first so that a book containing both a label and a longer label
    ending in it cannot resolve to the shorter one. `coding.parse_action` scans
    the enum in declaration order, which is safe for the eight-label space and
    would not be for an arbitrary one.
    """
    lowered = text.strip().lower()
    for label in sorted(book.labels, key=len, reverse=True):
        if re.search(rf"\b{re.escape(label)}\b", lowered):
            return label
    return None
