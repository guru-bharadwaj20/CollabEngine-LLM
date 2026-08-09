"""Instance generation.

Generation is ground-truth-first: we draw a feasible assignment, then emit only
constraints that assignment already satisfies. Every instance is therefore
satisfiable by construction and scores 1.0 under the grader when solved
perfectly, which is what makes the Phase 1 difficulty curve interpretable.

Difficulty knobs are continuous so the calibration sweep can find the band where
the task is hard enough to reward collaboration but not so hard the model floors
out. If no such band exists, the task -- not the harness -- needs redesigning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from collabengine.tasks.schema import (
    Component,
    Constraint,
    Instance,
    Job,
    Worker,
)

SKILLS: tuple[str, ...] = ("alpha", "beta", "gamma", "delta", "epsilon")


@dataclass(frozen=True, slots=True)
class DifficultySpec:
    """Knobs controlling instance hardness.

    `name` is recorded on the instance so runs can be grouped without recomputing.
    """

    name: str
    n_jobs: int
    n_workers: int
    n_blocks: int
    skills_per_worker: int
    capacity_slack: float
    """Multiplier on the minimum feasible capacity. 1.0 == exactly tight."""
    n_exclusions: int
    n_synthesis: int
    n_planted_errors: int
    value_floor_ratio: float
    """Value threshold as a fraction of total available value."""


PRESETS: dict[str, DifficultySpec] = {
    "tiny": DifficultySpec(
        name="tiny",
        n_jobs=6,
        n_workers=3,
        n_blocks=2,
        skills_per_worker=3,
        capacity_slack=1.5,
        n_exclusions=1,
        n_synthesis=1,
        n_planted_errors=1,
        value_floor_ratio=0.5,
    ),
    "easy": DifficultySpec(
        name="easy",
        n_jobs=10,
        n_workers=4,
        n_blocks=2,
        skills_per_worker=3,
        capacity_slack=1.4,
        n_exclusions=2,
        n_synthesis=2,
        n_planted_errors=2,
        value_floor_ratio=0.55,
    ),
    "medium": DifficultySpec(
        name="medium",
        n_jobs=16,
        n_workers=5,
        n_blocks=3,
        skills_per_worker=2,
        capacity_slack=1.2,
        n_exclusions=4,
        n_synthesis=3,
        n_planted_errors=3,
        value_floor_ratio=0.65,
    ),
    "hard": DifficultySpec(
        name="hard",
        n_jobs=24,
        n_workers=6,
        n_blocks=4,
        skills_per_worker=2,
        capacity_slack=1.1,
        n_exclusions=6,
        n_synthesis=5,
        n_planted_errors=4,
        value_floor_ratio=0.72,
    ),
    # Scaled from `hard` along the same axes to test one prediction: a single
    # agent degrades as instances grow (0.879 -> 0.842 from medium to hard)
    # while four agents do not (0.874 -> 0.871), so the lines should cross
    # somewhere above `hard`. See docs/PREREG-xhard.md, written before any
    # episode at this tier was generated.
    #
    # The per-agent budget is deliberately left alone (max_tokens 1024, 3
    # rounds, 4 agents). That is the whole design: total work grows while
    # individual capacity does not, which is the first condition under which
    # dividing the work would pay rather than being redundant.
    "xhard": DifficultySpec(
        name="xhard",
        n_jobs=36,
        n_workers=8,
        n_blocks=5,
        skills_per_worker=2,
        capacity_slack=1.05,
        n_exclusions=9,
        n_synthesis=7,
        n_planted_errors=6,
        value_floor_ratio=0.78,
    ),
}


def generate(seed: int, difficulty: str = "medium") -> Instance:
    """Build one satisfiable instance.

    Deterministic in `seed`: the same seed and difficulty always yield an
    identical instance, so runs are reproducible from config alone and
    transcripts need not embed the instance to be replayable.
    """
    if difficulty not in PRESETS:
        raise ValueError(
            f"unknown difficulty {difficulty!r}; expected one of {sorted(PRESETS)}"
        )
    spec = PRESETS[difficulty]
    rng = random.Random(seed)

    workers = _make_workers(rng, spec)
    jobs = _make_jobs(rng, spec)
    ground_truth = _draw_feasible_assignment(rng, jobs, workers)
    if ground_truth is None:
        # Skill draw left some job unplaceable. Reseeding deterministically is
        # simpler and cheaper than backtracking, and keeps generation total.
        return generate(seed * 2654435761 % (2**31 - 1) + 1, difficulty)

    workers = _fit_capacities(jobs, workers, ground_truth, spec)
    constraints = _emit_constraints(rng, spec, jobs, workers, ground_truth)
    proposed, planted = _make_proposed_assignment(rng, spec, jobs, workers, ground_truth)

    return Instance(
        instance_id=f"{difficulty}-{seed}",
        seed=seed,
        difficulty=difficulty,
        jobs=jobs,
        workers=workers,
        constraints=constraints,
        proposed_assignment=proposed,
        planted_errors=planted,
        ground_truth=ground_truth,
    )


def _make_workers(rng: random.Random, spec: DifficultySpec) -> tuple[Worker, ...]:
    workers = []
    for i in range(spec.n_workers):
        k = min(spec.skills_per_worker, len(SKILLS))
        skills = tuple(sorted(rng.sample(SKILLS, k)))
        workers.append(Worker(wid=f"W{i + 1}", skills=skills, capacity=0))
    return tuple(workers)


def _make_jobs(rng: random.Random, spec: DifficultySpec) -> tuple[Job, ...]:
    jobs = []
    for i in range(spec.n_jobs):
        jobs.append(
            Job(
                jid=f"J{i + 1}",
                duration=rng.randint(1, 6),
                skill=rng.choice(SKILLS),
                value=rng.randint(5, 40),
                block=i % spec.n_blocks,
            )
        )
    return tuple(jobs)


def _draw_feasible_assignment(
    rng: random.Random, jobs: tuple[Job, ...], workers: tuple[Worker, ...]
) -> dict[str, str] | None:
    """Assign every job to a skill-matching worker, or fail."""
    assignment: dict[str, str] = {}
    for job in jobs:
        eligible = [w for w in workers if job.skill in w.skills]
        if not eligible:
            return None
        assignment[job.jid] = rng.choice(eligible).wid
    return assignment


def _fit_capacities(
    jobs: tuple[Job, ...],
    workers: tuple[Worker, ...],
    ground_truth: dict[str, str],
    spec: DifficultySpec,
) -> tuple[Worker, ...]:
    """Set each worker's capacity to its ground-truth load times the slack.

    Slack near 1.0 makes capacity the binding constraint and forces genuine
    arithmetic bookkeeping; larger slack makes it nearly free.
    """
    load: dict[str, int] = {w.wid: 0 for w in workers}
    for job in jobs:
        load[ground_truth[job.jid]] += job.duration
    return tuple(
        Worker(
            wid=w.wid,
            skills=w.skills,
            capacity=max(1, int(load[w.wid] * spec.capacity_slack)),
        )
        for w in workers
    )


def _emit_constraints(
    rng: random.Random,
    spec: DifficultySpec,
    jobs: tuple[Job, ...],
    workers: tuple[Worker, ...],
    ground_truth: dict[str, str],
) -> tuple[Constraint, ...]:
    """Emit only constraints the ground truth already satisfies."""
    out: list[Constraint] = []

    # --- ARITHMETIC: per-worker capacity, plus a global value floor. ---
    for w in workers:
        out.append(
            Constraint(
                cid=f"cap-{w.wid}",
                kind="capacity",
                component=Component.ARITHMETIC,
                params={"worker": w.wid, "limit": w.capacity},
            )
        )
    total_value = sum(j.value for j in jobs)
    out.append(
        Constraint(
            cid="value-floor",
            kind="value_floor",
            component=Component.ARITHMETIC,
            params={"threshold": int(total_value * spec.value_floor_ratio)},
        )
    )

    # --- SEARCH: every job needs a skill match; some carry extra exclusions. ---
    for job in jobs:
        out.append(
            Constraint(
                cid=f"skill-{job.jid}",
                kind="skill_match",
                component=Component.SEARCH,
                params={"job": job.jid, "skill": job.skill},
            )
        )
    for n, job in enumerate(rng.sample(list(jobs), min(spec.n_exclusions, len(jobs)))):
        assigned = ground_truth[job.jid]
        banned = [w.wid for w in workers if w.wid != assigned]
        if not banned:
            continue
        k = min(len(banned), max(1, len(workers) // 2))
        out.append(
            Constraint(
                cid=f"excl-{n + 1}",
                kind="exclusion",
                component=Component.SEARCH,
                params={"job": job.jid, "banned": sorted(rng.sample(banned, k))},
            )
        )

    # --- SYNTHESIS: couple jobs across different blocks. ---
    cross_pairs = [
        (a, b)
        for i, a in enumerate(jobs)
        for b in jobs[i + 1 :]
        if a.block != b.block
    ]
    rng.shuffle(cross_pairs)
    emitted = 0
    for a, b in cross_pairs:
        if emitted >= spec.n_synthesis:
            break
        same = ground_truth[a.jid] == ground_truth[b.jid]
        out.append(
            Constraint(
                cid=f"syn-{emitted + 1}",
                kind="co_assign" if same else "separate",
                component=Component.SYNTHESIS,
                params={"job_a": a.jid, "job_b": b.jid},
            )
        )
        emitted += 1

    return tuple(out)


def _make_proposed_assignment(
    rng: random.Random,
    spec: DifficultySpec,
    jobs: tuple[Job, ...],
    workers: tuple[Worker, ...],
    ground_truth: dict[str, str],
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Corrupt the ground truth at a few positions.

    Errors are always skill-violations, so they are objectively detectable from
    the instance alone rather than requiring the team to guess our ground truth.
    """
    proposed = dict(ground_truth)
    candidates = [
        j for j in jobs if any(j.skill not in w.skills for w in workers)
    ]
    rng.shuffle(candidates)

    planted: list[str] = []
    for job in candidates:
        if len(planted) >= spec.n_planted_errors:
            break
        wrong = [w.wid for w in workers if job.skill not in w.skills]
        if not wrong:
            continue
        proposed[job.jid] = rng.choice(wrong)
        planted.append(job.jid)

    return proposed, tuple(sorted(planted))
