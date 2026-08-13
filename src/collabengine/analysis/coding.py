"""Behavioral coding: turning transcripts into per-agent action distributions.

This is the observational half of the study -- the half that prior work
(2604.00026) did thoroughly and that this project exists to *check* against
causal ablation. It produces the labels; `convergent.py` asks whether they
predict anything.

Three design commitments, each of which changes the result if broken:

**One message at a time, no agent identity in the prompt.** The judge sees the
text and nothing else -- not who wrote it, not what condition it came from, not
what the other agents said. Identity is re-attached afterwards from the record.
A judge that could see "A2" across a whole episode would have every opportunity
to invent a consistent character for A2, which is precisely the artifact the
study is trying to distinguish from a real one. This also makes the coding
trivially parallel, which matters at ~14k messages.

**The taxonomy is not the component list.** Actions are things an agent does
(compute, verify, propose); components are things the grader scores. They are
deliberately different vocabularies with a documented mapping, because Phase 4
asks whether the first predicts the second. Collapsing them into one vocabulary
would make that correlation true by construction.

**Two judges, and report the disagreement.** A single judge's labels are an
opinion. 2604.00026 reported Cohen's kappa of 0.78 across judges; anything
materially below that means the labels are too noisy to carry a
convergent-validity claim, and the honest move is to say so rather than to
report the correlation anyway.
"""

from __future__ import annotations

import asyncio
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Sequence

import numpy as np

from collabengine.backends.base import ChatMessage, GenRequest, LLMBackend
from collabengine.protocol import Message, Speaker
from collabengine.tasks.schema import ALL_COMPONENTS, Component
from collabengine.transcripts.store import EpisodeRecord


class ActionType(str, Enum):
    """What an agent's turn *does*, independent of what the grader scores."""

    PROPOSE = "propose"
    """Puts forward or revises a concrete assignment."""

    COMPUTE = "compute"
    """Works through capacity, duration or value arithmetic."""

    SEARCH = "search"
    """Explores alternative placements or tests feasibility."""

    VERIFY = "verify"
    """Audits the draft, or checks another agent's claim, for errors."""

    SYNTHESIZE = "synthesize"
    """Merges partial results from several agents into one answer."""

    ORGANIZE = "organize"
    """Divides the work or sets procedure, without doing the task itself."""

    AGREE = "agree"
    """Endorses what has been said without adding content."""

    OTHER = "other"


ALL_ACTIONS: tuple[ActionType, ...] = tuple(ActionType)

# The mapping Phase 4 tests rather than assumes. ORGANIZE, AGREE, PROPOSE and
# OTHER map to no component on purpose: they are real behaviors that the grader
# has no column for, and pretending otherwise would inflate the correlation.
ACTION_TO_COMPONENT: dict[ActionType, Component] = {
    ActionType.COMPUTE: Component.ARITHMETIC,
    ActionType.SEARCH: Component.SEARCH,
    ActionType.VERIFY: Component.VERIFICATION,
    ActionType.SYNTHESIZE: Component.SYNTHESIS,
}

JUDGE_SYSTEM = (
    "You are coding messages from a group problem-solving transcript into a "
    "fixed behavioral taxonomy. You will see one message with no information "
    "about who wrote it. Choose the single label that best describes what the "
    "message primarily DOES.\n\n"
    "The task in the transcript is an ALLOCATION PUZZLE: participants assign "
    "jobs (J1, J2, ...) to workers (W1, W2, ...) subject to skill, capacity and "
    "exclusion requirements. Read the labels below as descriptions of what a "
    "participant is doing IN THE CONVERSATION, never as descriptions of the "
    "puzzle's content.\n\n"
    "propose     - puts forward or revises an assignment of jobs to workers. This "
    "is the label for any message whose substance is 'J4 -> W2, J5 -> W1, ...', "
    "however it is worded and however much reasoning surrounds it.\n"
    "compute     - works through capacity, duration or value arithmetic\n"
    "search      - explores alternative placements or tests whether one is feasible\n"
    "verify      - audits the draft or another participant's claim for errors\n"
    "synthesize  - merges partial results from several participants into one answer\n"
    "organize    - coordinates the PARTICIPANTS with each other: who should do "
    "which part, what order to work in, what to do next as a group. For example "
    "'A2, you check the capacity constraints and I will do the arithmetic'. "
    "Assigning JOBS to WORKERS is never organize -- that is the puzzle being "
    "solved, not the participants being organised. If the message divides work "
    "among W1..Wn it is propose; only dividing work among the participants "
    "themselves is organize.\n"
    "agree       - endorses what was already said without adding content\n"
    "other       - none of the above\n\n"
    "Reply with the label and nothing else."
)
"""The codebook. `organize` carries the long gloss for a measured reason.

Against a human rater the previous wording -- "divides up the work or sets
procedure, without doing the task" -- put 17 of 40 messages in `organize`,
including 7 the human called `propose`, and produced kappa = 0.072 with 20% raw
agreement (RESEARCH-LOG 4.13). The failure was not judge weakness but a collision
between the label and the task: this task *is* dividing up work, so a message
reading "J5 -> W4, J9 -> W5" satisfies the old definition literally while being
the plainest possible `propose`.

So the definition now names the distinction the taxonomy always meant --
coordinating *participants* rather than allocating *jobs* -- and says which way
the ambiguous case resolves. Any change to this string invalidates every
previously coded corpus, because the labels are only comparable under one
codebook; re-code rather than mix.
"""


#: Bumped whenever JUDGE_SYSTEM changes meaning. Stamped on every code so a
#: file coded under one codebook can never be silently compared with another --
#: the labels are only comparable within a version. v1 is the wording that
#: returned kappa = 0.072 against a human (RESEARCH-LOG 4.13); v2 disambiguates
#: `organize` from the puzzle's own job-to-worker allocation.
CODEBOOK_VERSION = 2


@dataclass(frozen=True, slots=True)
class MessageCode:
    """One coded message. Identity is attached here, never shown to the judge."""

    episode_id: str
    turn: int
    agent_id: str
    action: ActionType
    judge: str = ""
    codebook: int = CODEBOOK_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "turn": self.turn,
            "agent_id": self.agent_id,
            "action": self.action.value,
            "judge": self.judge,
            "codebook": self.codebook,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MessageCode:
        return cls(
            episode_id=d["episode_id"],
            turn=int(d["turn"]),
            agent_id=d["agent_id"],
            action=ActionType(d["action"]),
            judge=d.get("judge", ""),
            # Files written before the field existed are v1 by construction.
            codebook=int(d.get("codebook", 1)),
        )

    @property
    def key(self) -> tuple[str, int]:
        """Identifies the coded message, independent of which judge coded it.

        Two judges' files are aligned on this to compute kappa; aligning on
        position instead would silently compare different messages whenever one
        judge's run skipped or reordered anything.
        """
        return (self.episode_id, self.turn)


class JudgeUnavailable(RuntimeError):
    """The judge stopped answering, so coding must stop too.

    Raised rather than absorbed because the alternative is worse than a crash:
    a failed call becomes an `other` label, and a file full of `other` is
    indistinguishable from a corpus of genuinely uncategorizable messages once
    it is on disk. Every downstream statistic would be computed over fabricated
    observations.
    """


@dataclass(slots=True)
class CodingStats:
    coded: int = 0
    unparseable: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    last_error: str = ""

    def summary(self) -> str:
        return (
            f"coded {self.coded} | unparseable {self.unparseable} | "
            f"errors {self.errors}"
        )

    def record_error(self, message: str, *, tolerate: int = 8) -> None:
        """Count a failure, and give up once the judge is clearly gone.

        A handful of transient errors across thousands of messages is noise. A
        run of them means the key is bad, the quota is spent, or the endpoint is
        down -- and continuing would keep writing `other` for every remaining
        message in the corpus.
        """
        self.errors += 1
        self.consecutive_errors += 1
        self.last_error = message
        if self.consecutive_errors >= tolerate:
            raise JudgeUnavailable(
                f"{self.consecutive_errors} consecutive judge failures; "
                f"last error: {message}"
            )

    def record_success(self) -> None:
        self.consecutive_errors = 0


async def code_episode(
    *,
    backend: LLMBackend,
    record: EpisodeRecord,
    judge_name: str = "judge",
    max_chars: int = 4000,
    stats: CodingStats | None = None,
) -> list[MessageCode]:
    """Code every agent turn in one episode.

    Turns are coded concurrently: they are independent by construction, and the
    backend batches whatever is in flight.
    """
    stats = stats if stats is not None else CodingStats()
    turns = [m for m in record.messages if m.speaker is Speaker.AGENT]
    if not turns:
        return []

    results = await asyncio.gather(
        *(_code_one(backend, m, max_chars, stats) for m in turns)
    )
    return [
        MessageCode(
            episode_id=record.episode_id,
            turn=m.turn,
            agent_id=m.author,
            action=action,
            judge=judge_name,
        )
        for m, action in zip(turns, results)
    ]


async def _code_one(
    backend: LLMBackend,
    message: Message,
    max_chars: int,
    stats: CodingStats,
    system: str | None = None,
) -> ActionType:
    # `system` exists so the codebook sweep can vary the prompt without
    # reimplementing the call. Defaulting to JUDGE_SYSTEM keeps every existing
    # caller on the codebook the corpus was coded under -- a sweep that
    # accidentally became the default would silently mix codebook versions,
    # which is the thing CODEBOOK_VERSION is stamped to prevent.
    request = GenRequest(
        messages=[
            ChatMessage(role="system", content=system or JUDGE_SYSTEM),
            ChatMessage(role="user", content=_strip_identity(message.content)[:max_chars]),
        ],
        max_tokens=8,
        # Coding is a classification, not a sample. Temperature here would add
        # variance to the very labels whose reliability is being reported.
        temperature=0.0,
        top_p=1.0,
        seed=message.turn,
    )
    response = await backend.generate(request)
    if not response.ok:
        stats.record_error(response.error or "unknown judge error")
        return ActionType.OTHER

    action = parse_action(response.text)
    stats.record_success()
    if action is None:
        stats.unparseable += 1
        return ActionType.OTHER
    stats.coded += 1
    return action


def _strip_identity(text: str) -> str:
    """Remove the agent's self-label so the judge cannot track identity.

    The team brief asks agents to prefix turns with "(A2)". Leaving that in
    would hand the judge exactly the identity signal this design withholds.
    """
    return re.sub(r"\(A\d+\)", "", text).strip()


def parse_action(text: str) -> ActionType | None:
    """Recover a label from the judge's reply, which is not always just a label."""
    lowered = text.strip().lower()
    for action in ALL_ACTIONS:
        if re.search(rf"\b{action.value}\b", lowered):
            return action
    return None


# --------------------------------------------------------------------- metrics


def action_distributions(
    codes: Iterable[MessageCode],
) -> dict[str, dict[ActionType, float]]:
    """Per-agent distribution over action types, normalized within agent."""
    counts: dict[str, Counter] = {}
    for c in codes:
        counts.setdefault(c.agent_id, Counter())[c.action] += 1

    out: dict[str, dict[ActionType, float]] = {}
    for agent, counter in counts.items():
        total = sum(counter.values())
        out[agent] = {a: counter.get(a, 0) / total for a in ALL_ACTIONS} if total else {}
    return out


def js_divergence(p: dict[ActionType, float], q: dict[ActionType, float]) -> float:
    """Jensen-Shannon divergence between two action distributions, base 2.

    Bounded in [0, 1], symmetric, and finite when one distribution has zeros --
    all three of which KL lacks and all three of which matter when an agent
    simply never performs some action.
    """
    pv = np.array([p.get(a, 0.0) for a in ALL_ACTIONS], dtype=float)
    qv = np.array([q.get(a, 0.0) for a in ALL_ACTIONS], dtype=float)
    if pv.sum() == 0 or qv.sum() == 0:
        return 0.0
    pv, qv = pv / pv.sum(), qv / qv.sum()
    m = (pv + qv) / 2
    return float((_kl(pv, m) + _kl(qv, m)) / 2)


def _kl(a: np.ndarray, b: np.ndarray) -> float:
    mask = a > 0
    return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))


def mean_pairwise_divergence(
    distributions: dict[str, dict[ActionType, float]],
) -> float:
    """How differentiated the team is overall. Zero means interchangeable."""
    agents = sorted(distributions)
    pairs = [
        js_divergence(distributions[a], distributions[b])
        for i, a in enumerate(agents)
        for b in agents[i + 1 :]
    ]
    return float(np.mean(pairs)) if pairs else 0.0


def differentiation_vs_null(
    codes: Sequence[MessageCode],
    *,
    n_permutations: int = 500,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Observed differentiation, null mean, and a permutation p-value.

    The null shuffles which agent produced each message *within its episode*,
    which preserves how many turns each agent took and what the episode's action
    mix was, and destroys only the association between identity and action. Four
    agents drawing from one shared distribution still show nonzero pairwise
    divergence purely from finite samples -- comparing raw divergence to zero
    would report specialization in a team that has none, which is exactly the
    failure the selftest exists to catch.
    """
    observed = mean_pairwise_divergence(action_distributions(codes))
    if not codes:
        return 0.0, 0.0, 1.0

    rng = np.random.default_rng(seed)
    by_episode: dict[str, list[MessageCode]] = {}
    for c in codes:
        by_episode.setdefault(c.episode_id, []).append(c)

    null = np.empty(n_permutations)
    for i in range(n_permutations):
        shuffled: list[MessageCode] = []
        for group in by_episode.values():
            agents = [c.agent_id for c in group]
            rng.shuffle(agents)
            shuffled.extend(
                MessageCode(c.episode_id, c.turn, a, c.action, c.judge)
                for c, a in zip(group, agents)
            )
        null[i] = mean_pairwise_divergence(action_distributions(shuffled))

    # +1 in numerator and denominator: with finite permutations a p-value of
    # exactly zero is not a thing that was measured.
    p = float((np.sum(null >= observed) + 1) / (n_permutations + 1))
    return observed, float(null.mean()), p


def cohens_kappa(a: Sequence[ActionType], b: Sequence[ActionType]) -> float:
    """Inter-judge agreement corrected for chance.

    Raw agreement flatters any taxonomy with a dominant class: two judges that
    both label 80% of messages "propose" agree 80% of the time while sharing no
    real signal. Kappa removes that floor.
    """
    if len(a) != len(b):
        raise ValueError("judges must have coded the same messages")
    if not a:
        return 0.0

    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if expected >= 1.0:
        return 1.0 if observed >= 1.0 else 0.0
    return float((observed - expected) / (1 - expected))


def kappa_interval(
    a: Sequence[ActionType],
    b: Sequence[ActionType],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap 95% interval for Cohen's kappa.

    A point estimate of kappa carries no information about how many messages it
    rests on, and the sample available here is set by a judge's free quota
    rather than by design. Resampling message pairs with replacement gives the
    width directly, which is the difference between "the judges agree" and "we
    cannot tell whether the judges agree".
    """
    if len(a) != len(b):
        raise ValueError("judges must have coded the same messages")
    if len(a) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    n = len(a)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        estimates[i] = cohens_kappa([a[j] for j in idx], [b[j] for j in idx])
    return (
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def ownership_from_codes(
    codes: Iterable[MessageCode],
    *,
    components: Sequence[Component] = ALL_COMPONENTS,
) -> dict[Component, str]:
    """The transcript's claim about who owns each component.

    For each component, the agent with the largest share of the corresponding
    action. This is the object Phase 4 tests: `analyze_interaction` scores
    diagonal dominance against it, so a component whose owner is read off the
    transcript here must earn that label from the ablation matrix there.

    Share, not count: an agent that simply talks more would otherwise own every
    component, which would be a verbosity measure wearing a role's name.
    """
    dists = action_distributions(codes)
    owners: dict[Component, str] = {}
    for component in components:
        action = _component_action(component)
        if action is None:
            continue
        best = max(dists, key=lambda ag: dists[ag].get(action, 0.0), default=None)
        if best is not None and dists[best].get(action, 0.0) > 0:
            owners[component] = best
    return owners


def _component_action(component: Component) -> ActionType | None:
    for action, comp in ACTION_TO_COMPONENT.items():
        if comp is component:
            return action
    return None


def role_consistency(codes: Sequence[MessageCode]) -> float:
    """How stable an agent's action profile is across episodes.

    One minus the mean within-agent JS divergence between episodes. High means
    A2 behaves like A2 wherever it appears; low means the labels describe a
    momentary posture rather than a role, and calling it one would overstate
    what the transcript supports.
    """
    per_agent: dict[str, dict[str, list[MessageCode]]] = {}
    for c in codes:
        per_agent.setdefault(c.agent_id, {}).setdefault(c.episode_id, []).append(c)

    scores: list[float] = []
    for episodes in per_agent.values():
        if len(episodes) < 2:
            continue
        dists = [
            action_distributions(group)[agent]
            for group in episodes.values()
            for agent in [group[0].agent_id]
        ]
        pairs = [
            js_divergence(dists[i], dists[j])
            for i in range(len(dists))
            for j in range(i + 1, len(dists))
        ]
        if pairs:
            scores.append(1.0 - float(np.mean(pairs)))
    return float(np.mean(scores)) if scores else 0.0


@dataclass(slots=True)
class CodingReport:
    n_messages: int
    distributions: dict[str, dict[ActionType, float]] = field(default_factory=dict)
    differentiation: float = 0.0
    null_mean: float = 0.0
    p_value: float = 1.0
    consistency: float = 0.0
    ownership: dict[Component, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_messages": self.n_messages,
            "distributions": {
                agent: {a.value: round(v, 4) for a, v in dist.items()}
                for agent, dist in self.distributions.items()
            },
            "differentiation": self.differentiation,
            "null_mean": self.null_mean,
            "p_value": self.p_value,
            "consistency": self.consistency,
            "ownership": {c.value: a for c, a in self.ownership.items()},
        }


def summarize(codes: Sequence[MessageCode], *, n_permutations: int = 500) -> CodingReport:
    observed, null_mean, p = differentiation_vs_null(
        codes, n_permutations=n_permutations
    )
    return CodingReport(
        n_messages=len(codes),
        distributions=action_distributions(codes),
        differentiation=observed,
        null_mean=null_mean,
        p_value=p,
        consistency=role_consistency(codes),
        ownership=ownership_from_codes(codes),
    )
