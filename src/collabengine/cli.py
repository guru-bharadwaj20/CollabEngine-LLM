"""Command line entry point.

Subcommands map onto the phases in PLAN.md:

  calibrate  Phase 1 -- find the difficulty band where the task is hard enough
             to reward collaboration but not so hard the model floors out
  baseline   Phase 2 -- record un-ablated episodes
  ablate     Phase 3 -- run the ablation grid over a recorded baseline
  analyze    Phase 3/4 -- interaction report from recorded transcripts
  selftest   validate the measurement instrument against known-ground-truth
             mock worlds; no GPU required
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path

from collabengine.ablation import (
    capacity_control,
    frozen_excise,
    frozen_replay,
    live_ablation,
)
from collabengine.ablation.modes import (
    fungibility,
    propagation_index,
    random_message_control,
)
from collabengine.analysis import (
    AblationMatrix,
    MessageCode,
    analyze_interaction,
    code_episode,
    cohens_kappa,
    convergent_validity,
    summarize,
)
from collabengine.analysis.coding import CodingStats
from collabengine.backends.mock import MockBackend, MockMode
from collabengine.config import ExperimentConfig
from collabengine.orchestrator import run_episode
from collabengine.orchestrator.team import TeamConfig, build_team
from collabengine.runner import RunPlan, run_plan
from collabengine.tasks.generator import PRESETS
from collabengine.tasks.schema import ALL_COMPONENTS, Component
from collabengine.transcripts.store import TranscriptReader

BASELINE = "baseline.jsonl"
ABLATION = "ablation.jsonl"
_ALL_MODES = frozenset(
    {"live", "frozen_excise", "frozen_replay", "capacity", "random_message"}
)


def _run_id(condition: str, difficulty: str, seed: int) -> str:
    """The id `run_episode` will stamp on the record it produces.

    Resume compares plan ids against the ids already in the transcript, so a
    plan id that does not match its own record means the work is never skipped
    and the whole grid re-runs on restart. Deriving both from one function is
    what keeps them in step; `tests/test_pipeline_cli.py` asserts they agree for
    every mode.
    """
    return f"{condition}:{difficulty}:{seed}"


def _derived_id(mode: str, agent: str, source_episode_id: str) -> str:
    """The id the frozen-transcript modes stamp on their derived records."""
    return f"{mode}:{agent}:{source_episode_id}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="collabengine", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("calibrate", help="difficulty sweep (Phase 1)")
    p.add_argument("--config", type=Path)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument(
        "--difficulties",
        help="comma-separated subset of the presets (default: all)",
    )

    p = sub.add_parser("baseline", help="record un-ablated episodes (Phase 2)")
    p.add_argument("--config", type=Path, required=True)

    p = sub.add_parser("ablate", help="run the ablation grid (Phase 3)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--modes",
        default="live,frozen_excise,frozen_replay,capacity,random_message",
        help=(
            "comma-separated subset of: live, frozen_excise, frozen_replay, "
            "capacity, random_message"
        ),
    )

    p = sub.add_parser("analyze", help="interaction report from transcripts")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ownership", type=Path, help="JSON map of component -> agent id")

    p = sub.add_parser(
        "pipeline",
        help="run every GPU phase back to back in one process (keeps the card busy)",
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--episodes", type=int, help="override n_episodes")
    p.add_argument(
        "--phases",
        default="baseline,symmetry,c2,ablate",
        help="comma-separated subset of: baseline, symmetry, c2, ablate",
    )
    p.add_argument(
        "--auto-difficulty",
        action="store_true",
        help="calibrate first and pick the operating point, in the same process",
    )
    p.add_argument("--calibrate-episodes", type=int, default=6)

    p = sub.add_parser("code", help="behavioral coding of transcripts (Phase 2)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--transcript", type=Path, help=f"default: <run_dir>/{BASELINE}")
    p.add_argument("--out", type=Path, help="default: <run_dir>/codes.<judge>.jsonl")
    p.add_argument(
        "--judge",
        default="gemini",
        help="gemini | anthropic | self (the config's own backend) | mock",
    )
    p.add_argument("--judge-model", default=None, help="override the judge model id")
    p.add_argument(
        "--judge-rpm",
        type=float,
        default=5.0,
        help="requests/minute to pace at (free tier allows 5 per model)",
    )
    p.add_argument("--judge-name", default=None, help="label recorded on each code")
    p.add_argument(
        "--limit",
        type=int,
        help="code only the first N episodes -- use for the human-validation "
        "subsample before paying for the full corpus",
    )

    p = sub.add_parser("kappa", help="inter-judge agreement between two coding runs")
    p.add_argument("first", type=Path)
    p.add_argument("second", type=Path)

    p = sub.add_parser("converge", help="do labels predict contribution? (Phase 4)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--codes", type=Path, required=True)
    p.add_argument("--permutations", type=int, default=2000)

    p = sub.add_parser("selftest", help="validate the instrument on mock worlds")
    p.add_argument("--episodes", type=int, default=30)

    args = parser.parse_args(argv)
    return {
        "calibrate": cmd_calibrate,
        "baseline": cmd_baseline,
        "ablate": cmd_ablate,
        "analyze": cmd_analyze,
        "pipeline": cmd_pipeline,
        "code": cmd_code,
        "kappa": cmd_kappa,
        "converge": cmd_converge,
        "selftest": cmd_selftest,
    }[args.command](args)


def _load(path: Path | None) -> ExperimentConfig:
    return ExperimentConfig.load(path) if path else ExperimentConfig()


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Sweep difficulty at one agent and at N, and report the collaboration gap.

    Phase 1 is looking for a band, and the band is defined by two edges that a
    single team size cannot see:

      floor    the model scores near zero however many agents it has, so any
               ablation effect is measured against noise
      ceiling  one agent already saturates the task, so a team has nothing to
               divide and no role can pay for itself

    The operating point wants `team - solo` large and `team` short of 1.0. A
    preset where four agents match one agent is not a collaboration task, no
    matter how hard it looks -- and running Phase 2 on it would produce a null
    result about the harness rather than about emergence.
    """
    config = _load(args.config)
    backend = config.backend.build()
    print(f"backend: {backend.name}", file=sys.stderr)

    n_agents = config.team.n_agents
    difficulties = sorted(PRESETS, key=lambda d: PRESETS[d].n_jobs)
    if args.difficulties:
        wanted = {d.strip() for d in args.difficulties.split(",")}
        difficulties = [d for d in difficulties if d in wanted]

    print(
        f"{'difficulty':<10}{'solo':>8}{'sd':>7}"
        f"{f'{n_agents}-agent':>10}{'sd':>7}{'gap':>8}"
    )
    rows: list[tuple[str, float, float]] = []
    for difficulty in difficulties:
        base = {**config.team.to_dict(), "difficulty": difficulty}
        solo = asyncio.run(
            _score_many(
                backend, TeamConfig.from_dict({**base, "n_agents": 1}), args.episodes
            )
        )
        team = asyncio.run(
            _score_many(backend, TeamConfig.from_dict(base), args.episodes)
        )
        gap = statistics.mean(team) - statistics.mean(solo)
        rows.append((difficulty, statistics.mean(team), gap))
        print(
            f"{difficulty:<10}{statistics.mean(solo):>8.3f}"
            f"{statistics.pstdev(solo):>7.3f}"
            f"{statistics.mean(team):>10.3f}{statistics.pstdev(team):>7.3f}"
            f"{gap:>+8.3f}",
            flush=True,
        )

    _recommend(rows)
    return 0


def _recommend(rows: list[tuple[str, float, float]]) -> None:
    """Name an operating point, or say plainly that there is not one."""
    usable = [r for r in rows if 0.05 < r[1] < 0.95]
    if not usable:
        print(
            "\nNo preset sits off both the floor and the ceiling. Redesign the "
            "task before Phase 2 -- everything downstream depends on this band "
            "existing (PLAN.md 4, Phase 1).",
            file=sys.stderr,
        )
        return
    best = max(usable, key=lambda r: r[2])
    print(f"\nsuggested operating point: difficulty={best[0]} (gap {best[2]:+.3f})")
    if best[2] <= 0:
        print(
            "  WARNING: the team does not beat one agent anywhere in the band. "
            "Division of labor cannot pay off on this task as configured, and a "
            "null ablation result would say nothing about emergence.",
            file=sys.stderr,
        )


async def _score_many(backend, team: TeamConfig, n: int) -> list[float]:
    records = await asyncio.gather(
        *(
            run_episode(
                backend=backend, config=team, episode_seed=s, condition="calibrate"
            )
            for s in range(n)
        )
    )
    return [r.grade.overall for r in records]


def cmd_baseline(args: argparse.Namespace) -> int:
    config = ExperimentConfig.load(args.config)
    backend = config.backend.build()
    out = config.run_dir / BASELINE
    config.save(config.run_dir / "config.resolved.yaml")

    plans = [
        RunPlan(
            episode_id=f"baseline:{config.team.difficulty}:{seed}",
            factory=(
                lambda seed=seed: run_episode(
                    backend=backend,
                    config=config.team,
                    episode_seed=seed,
                    condition="baseline",
                )
            ),
        )
        for seed in config.seeds
    ]

    stats = asyncio.run(
        run_plan(
            plans,
            out_path=out,
            max_concurrency=config.max_concurrency,
            on_progress=lambda s: print(f"  {s.summary()}", file=sys.stderr),
        )
    )
    print(f"baseline -> {out}: {stats.summary()}")
    return 0 if stats.failed == 0 else 1


def cmd_ablate(args: argparse.Namespace) -> int:
    config = ExperimentConfig.load(args.config)
    backend = config.backend.build()
    baseline_path = config.run_dir / BASELINE

    if not baseline_path.exists():
        print(f"no baseline at {baseline_path}; run `baseline` first", file=sys.stderr)
        return 2

    records = [r for r in TranscriptReader(baseline_path) if r.condition == "baseline"]
    modes = {m.strip() for m in args.modes.split(",") if m.strip()}
    agents = [a.agent_id for a in build_team(config.team, config.seed_start)]
    plans = _ablation_plans(config, backend, records, modes)

    stats = asyncio.run(
        run_plan(
            plans,
            out_path=config.run_dir / ABLATION,
            max_concurrency=config.max_concurrency,
            on_progress=lambda s: print(f"  {s.summary()}", file=sys.stderr),
        )
    )
    print(f"ablation -> {config.run_dir / ABLATION}: {stats.summary()}")
    _report_propagation(records, agents)
    return 0 if stats.failed == 0 else 1


def _ablation_plans(config, backend, records, modes: set[str]) -> list[RunPlan]:
    """The Phase 3 condition grid over a recorded baseline.

    `frozen_excise` and `random_message` cost zero model calls, so they run over
    every episode rather than a sampled subset -- and the second is what makes
    the first interpretable, since excising k messages also shortens the context.
    """
    agents = [a.agent_id for a in build_team(config.team, config.seed_start)]
    plans: list[RunPlan] = []

    difficulty = config.team.difficulty
    capacity_size = max(1, config.team.n_agents - 1)

    for record in records:
        seed = record.instance_seed
        if "capacity" in modes:
            plans.append(
                RunPlan(
                    _run_id(f"capacity:{capacity_size}", difficulty, seed),
                    lambda seed=seed: capacity_control(
                        backend=backend, config=config.team, episode_seed=seed
                    ),
                )
            )
        for agent in agents:
            if "live" in modes:
                plans.append(
                    RunPlan(
                        _run_id(f"live:{agent}", difficulty, seed),
                        lambda seed=seed, agent=agent: live_ablation(
                            backend=backend,
                            config=config.team,
                            episode_seed=seed,
                            agent_id=agent,
                        ),
                    )
                )
            if "frozen_replay" in modes:
                plans.append(
                    RunPlan(
                        _derived_id("frozen_replay", agent, record.episode_id),
                        lambda rec=record, agent=agent: frozen_replay(
                            backend=backend,
                            record=rec,
                            config=config.team,
                            agent_id=agent,
                        ),
                    )
                )
            if "frozen_excise" in modes:
                # Free: no model calls, so it runs over every episode.
                plans.append(
                    RunPlan(
                        _derived_id("frozen_excise", agent, record.episode_id),
                        lambda rec=record, agent=agent: _immediate(
                            frozen_excise(rec, agent)
                        ),
                    )
                )
            if "random_message" in modes:
                # The volume-matched control for frozen_excise. Also free, and
                # excision drops are uninterpretable without it: dropping k
                # messages shortens the context whoever wrote them.
                plans.append(
                    RunPlan(
                        _derived_id("random_message", agent, record.episode_id),
                        lambda rec=record, agent=agent: _immediate(
                            random_message_control(rec, agent, seed=config.seed_start)
                        ),
                    )
                )
    return plans


async def _immediate(value):
    return value


def _report_propagation(records, agents: list[str]) -> None:
    """Warn when excision cannot be trusted on this corpus."""
    scores = [
        propagation_index(r, a) for r in records for a in agents if r.messages_by(a)
    ]
    if not scores:
        return
    mean = statistics.mean(scores)
    print(f"propagation index: {mean:.2f}")
    if mean > 0.3:
        print(
            "  WARNING: agents restate heavily, so frozen_excise will "
            "under-report. Prefer frozen_replay for the headline number.",
            file=sys.stderr,
        )


def cmd_analyze(args: argparse.Namespace) -> int:
    config = ExperimentConfig.load(args.config)
    baseline_path = config.run_dir / BASELINE
    ablation_path = config.run_dir / ABLATION

    for path in (baseline_path, ablation_path):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 2

    matrix = _live_matrix(baseline_path, ablation_path)
    if matrix is None:
        print("no live-ablation records found", file=sys.stderr)
        return 2

    ownership = None
    if args.ownership:
        raw = json.loads(Path(args.ownership).read_text(encoding="utf-8"))
        ownership = {Component(k): v for k, v in raw.items()}

    report = analyze_interaction(matrix, ownership)
    print(json.dumps(report.to_dict(), indent=2))
    print(
        f"\ninteraction strength {report.interaction_strength:.4f} | "
        f"diagonal dominance {report.diagonal_dominance:.2f} "
        f"(chance {report.chance_level:.2f})"
    )
    _report_modes(baseline_path, ablation_path)
    _report_mixed(ablation_path)
    return 0


def _report_mixed(ablation_path: Path) -> None:
    """The significance test for the interaction, with episode as random effect.

    Double-centering gives the effect size; this says whether it is
    distinguishable from noise once shared instance difficulty is absorbed.
    """
    from collabengine.analysis.mixed import fit_interaction

    report = fit_interaction(TranscriptReader(ablation_path))
    if report.interaction_p is None:
        print(f"\nmixed-effects interaction: not fitted ({report.note})")
        return

    print(
        f"\nmixed-effects interaction: chi2={report.interaction_chi2:.2f} "
        f"p={report.interaction_p:.4g} over {report.n_observations} observations "
        f"from {report.n_episodes} episodes"
    )
    if report.group_variance is not None:
        print(f"  episode random-intercept variance: {report.group_variance:.4f}")
    if not report.converged:
        print("  WARNING: the fit did not converge; treat the p-value as unreliable",
              file=sys.stderr)
    elif not report.significant:
        print(
            "  The interaction is not distinguishable from noise. Whatever the "
            "transcripts look like, ablation does not show agents damaging "
            "their own components differentially -- which is the claim."
        )


def _report_modes(baseline_path: Path, ablation_path: Path) -> None:
    """Per-mode scalar drops, and the fungibility metric they produce.

    `Delta(frozen_replay) - Delta(live)` is the headline of Phase 3: the gap
    between a compensation-blocked drop and a compensation-permitting one. Near
    zero means the contribution was irreplaceable; large means the role was real
    but any of them could have filled it. Neither number means anything alone,
    which is why they are printed together with their controls.
    """
    base = statistics.mean(
        [r.grade.overall for r in TranscriptReader(baseline_path) if r.condition == "baseline"]
        or [0.0]
    )
    by_mode: dict[str, dict[str, list[float]]] = {}
    for record in TranscriptReader(ablation_path):
        mode, _, agent = record.condition.partition(":")
        by_mode.setdefault(mode, {}).setdefault(agent, []).append(record.grade.overall)

    if not by_mode:
        return

    print(f"\nbaseline overall: {base:.3f}")
    print(f"{'mode':<16}{'mean drop':>11}{'per-agent drops':>20}")
    drops: dict[str, float] = {}
    for mode in sorted(by_mode):
        per_agent = {a: base - statistics.mean(v) for a, v in by_mode[mode].items()}
        mean_drop = statistics.mean(per_agent.values())
        drops[mode] = mean_drop
        detail = " ".join(f"{a}:{d:+.3f}" for a, d in sorted(per_agent.items()))
        print(f"{mode:<16}{mean_drop:>+11.3f}   {detail}")

    if "live" in drops and "frozen_replay" in drops:
        print(
            f"\nfungibility (frozen_replay - live): "
            f"{fungibility(drops['live'], drops['frozen_replay']):+.3f}"
        )
    if "frozen_excise" in drops and "random_message" in drops:
        excess = drops["frozen_excise"] - drops["random_message"]
        print(
            f"excision above its volume-matched control: {excess:+.3f}"
            + (
                "\n  At or below the control, excision is measuring lost context "
                "rather than this agent's contribution."
                if excess <= 0
                else ""
            )
        )


def _component_means(reader: TranscriptReader) -> dict[Component, float]:
    acc: dict[Component, list[float]] = {c: [] for c in ALL_COMPONENTS}
    for record in reader:
        for comp, val in record.grade.per_component.items():
            acc[comp].append(val)
    return {c: (statistics.mean(v) if v else 0.0) for c, v in acc.items()}


def cmd_pipeline(args: argparse.Namespace) -> int:
    """Run every GPU phase back to back in one process.

    Three things this does that running the subcommands in sequence does not,
    each of which is wall-clock rather than cosmetic:

    * **The weights load once.** Every separate invocation pays ~15 s to
      re-materialize 16 GB from disk, and the card sits idle throughout.
    * **Independent phases share one queue.** Baseline, the symmetry sweep and
      the fixed-order control do not depend on each other, so they are submitted
      as a single work list. A batching backend runs at its worst when a phase is
      draining -- the last few episodes of a hundred leave a batch of two on a
      card sized for sixteen. Merging removes three of those drains.
    * **No human in the loop between phases.** The ablation grid starts the
      instant the baseline it reads from is on disk.

    Ablation cannot merge into the first stage: it is defined over recorded
    transcripts, so it has to wait for them.
    """
    config = ExperimentConfig.load(args.config)
    if args.episodes:
        config.n_episodes = args.episodes
    backend = config.backend.build()
    phases = {p.strip() for p in args.phases.split(",") if p.strip()}

    from collabengine.backends.hf_local import cuda_report

    print(f"device: {cuda_report()}", file=sys.stderr)

    if args.auto_difficulty:
        # Calibrating in this process rather than a prior invocation is the
        # difference between one model load and two, and -- more to the point --
        # it leaves no window where the card sits idle waiting for someone to
        # read a table and pick a row.
        chosen = asyncio.run(
            _choose_difficulty(config, backend, args.calibrate_episodes)
        )
        if chosen is None:
            return 2
        config.team = replace(config.team, difficulty=chosen)

    config.save(config.run_dir / "config.resolved.yaml")
    print(
        f"run dir: {config.run_dir} | episodes: {config.n_episodes} "
        f"| difficulty: {config.team.difficulty}",
        file=sys.stderr,
    )
    return asyncio.run(_pipeline(config, backend, phases))


async def _choose_difficulty(config: ExperimentConfig, backend, episodes: int):
    """Run the Phase 1 sweep and return the operating point, or None to stop.

    Same criterion as `calibrate`: off the floor, off the ceiling, and the
    largest gap between a team and a single agent. Returning None is a real
    outcome -- if no preset rewards collaboration, running Phase 2 on it would
    produce a null result about the harness rather than about emergence, and
    the right move is to stop rather than to spend the card proving it.
    """
    print(f"\n== calibrating ({episodes} episodes/cell) ==", file=sys.stderr)

    # Every cell is submitted at once rather than one difficulty at a time. A
    # cell holds only `episodes` turns in flight, so running them sequentially
    # caps the batch at that number however much room the card has -- eight
    # cells of six episodes fills a batch that six alone leaves five-sixths
    # empty. The cells are independent, so there is nothing to serialize for.
    difficulties = sorted(PRESETS, key=lambda d: PRESETS[d].n_jobs)
    cells: list[tuple[str, int]] = [(d, n) for d in difficulties for n in (1, 0)]

    async def score(difficulty: str, solo: int) -> list[float]:
        base = {**config.team.to_dict(), "difficulty": difficulty}
        if solo:
            base["n_agents"] = 1
        return await _score_many(backend, TeamConfig.from_dict(base), episodes)

    results = await asyncio.gather(*(score(d, s) for d, s in cells))
    scored = dict(zip(cells, results))

    print(f"{'difficulty':<10}{'solo':>8}{'team':>8}{'gap':>9}", file=sys.stderr)
    rows: list[tuple[str, float, float]] = []
    for difficulty in difficulties:
        solo = statistics.mean(scored[(difficulty, 1)])
        team = statistics.mean(scored[(difficulty, 0)])
        rows.append((difficulty, team, team - solo))
        print(
            f"{difficulty:<10}{solo:>8.3f}{team:>8.3f}{team - solo:>+9.3f}",
            file=sys.stderr,
            flush=True,
        )

    usable = [r for r in rows if 0.05 < r[1] < 0.95]
    if not usable:
        print(
            "\nNo preset sits off both the floor and the ceiling. Redesign the "
            "task before Phase 2 (PLAN.md 4, Phase 1).",
            file=sys.stderr,
        )
        return None
    best = max(usable, key=lambda r: r[2])
    print(f"\noperating point: {best[0]} (gap {best[2]:+.3f})", file=sys.stderr)
    return best[0]


async def _pipeline(config: ExperimentConfig, backend, phases: set[str]) -> int:
    started = time.monotonic()
    baseline_path = config.run_dir / BASELINE
    failures = 0

    stage1: list[RunPlan] = []
    if "baseline" in phases:
        stage1 += _baseline_plans(config, backend)
    if "symmetry" in phases:
        stage1 += _symmetry_plans(config, backend)
    if "c2" in phases:
        stage1 += _fixed_order_plans(config, backend)

    if stage1:
        print(f"\n== stage 1: {len(stage1)} episodes ==", file=sys.stderr)
        stats = await run_plan(
            stage1,
            out_path=baseline_path,
            max_concurrency=config.max_concurrency,
            on_progress=lambda s: print(f"  {s.summary()}", file=sys.stderr),
        )
        failures += stats.failed
        print(f"stage 1 -> {baseline_path}: {stats.summary()}")

    if "ablate" in phases:
        if not baseline_path.exists():
            print("no baseline to ablate; include the baseline phase", file=sys.stderr)
            return 2
        records = [
            r for r in TranscriptReader(baseline_path) if r.condition == "baseline"
        ]
        plans = _ablation_plans(config, backend, records, _ALL_MODES)
        print(f"\n== stage 2: {len(plans)} ablation runs ==", file=sys.stderr)
        stats = await run_plan(
            plans,
            out_path=config.run_dir / ABLATION,
            max_concurrency=config.max_concurrency,
            on_progress=lambda s: print(f"  {s.summary()}", file=sys.stderr),
        )
        failures += stats.failed
        print(f"stage 2 -> {config.run_dir / ABLATION}: {stats.summary()}")
        agents = [a.agent_id for a in build_team(config.team, config.seed_start)]
        _report_propagation(records, agents)

    _report_throughput(config.run_dir, time.monotonic() - started)
    return 0 if failures == 0 else 1


def _baseline_plans(config: ExperimentConfig, backend) -> list[RunPlan]:
    return [
        RunPlan(
            episode_id=f"baseline:{config.team.difficulty}:{seed}",
            factory=(
                lambda seed=seed: run_episode(
                    backend=backend,
                    config=config.team,
                    episode_seed=seed,
                    condition="baseline",
                )
            ),
        )
        for seed in config.seeds
    ]


def _symmetry_plans(config: ExperimentConfig, backend) -> list[RunPlan]:
    """The C3 sweep: does minimal asymmetry amplify into stable roles?"""
    from collabengine.orchestrator.team import SymmetryBreaking

    plans: list[RunPlan] = []
    for level in SymmetryBreaking:
        if level is config.team.symmetry:
            continue  # already covered by the baseline condition
        team = replace(config.team, symmetry=level)
        plans += [
            RunPlan(
                episode_id=_run_id(
                    f"symmetry:{level.value}", config.team.difficulty, seed
                ),
                factory=(
                    lambda seed=seed, team=team, level=level: run_episode(
                        backend=backend,
                        config=team,
                        episode_seed=seed,
                        condition=f"symmetry:{level.value}",
                    )
                ),
            )
            for seed in config.seeds
        ]
    return plans


def _fixed_order_plans(config: ExperimentConfig, backend) -> list[RunPlan]:
    """The C2 control: with speaking order frozen, do roles track slot or identity?

    PLAN.md calls this the highest-information-per-GPU-hour test in the project,
    because a positional world scores 0.50 against a 0.25 baseline under a fixed
    order -- close enough to be mistaken for real specialization.
    """
    team = replace(config.team, randomize_turn_order=False)
    return [
        RunPlan(
            episode_id=_run_id("fixed_order", config.team.difficulty, seed),
            factory=(
                lambda seed=seed: run_episode(
                    backend=backend,
                    config=team,
                    episode_seed=seed,
                    condition="fixed_order",
                )
            ),
        )
        for seed in config.seeds
    ]


def _report_throughput(run_dir: Path, elapsed_s: float) -> None:
    """State the achieved rate, so the next run can be sized from data."""
    tokens = 0
    episodes = 0
    for name in (BASELINE, ABLATION):
        path = run_dir / name
        if not path.exists():
            continue
        for record in TranscriptReader(path):
            episodes += 1
            tokens += sum(
                int(m.meta.get("completion_tokens", 0) or 0) for m in record.messages
            )
    if elapsed_s <= 0:
        return
    print(
        f"\ncorpus: {episodes} episodes, {tokens:,} output tokens | "
        f"{elapsed_s / 60:.1f} min this run | {tokens / elapsed_s:.0f} tok/s aggregate"
    )


def cmd_code(args: argparse.Namespace) -> int:
    """Code a recorded corpus into per-message action labels.

    The judge is chosen separately from the agents' backend and defaults to a
    frontier model, because PLAN.md Phase 2 is explicit that a local 7-8B is not
    an adequate judge. `--judge self` reuses the config's backend anyway, which
    is the right call for a smoke test and the wrong one for a reported number.
    """
    config = ExperimentConfig.load(args.config)
    transcript = args.transcript or (config.run_dir / BASELINE)
    if not transcript.exists():
        print(f"no transcript at {transcript}; run `baseline` first", file=sys.stderr)
        return 2

    judge_name = args.judge_name or args.judge
    backend, warning = _build_judge(args, config)
    if backend is None:
        print(warning, file=sys.stderr)
        return 2
    if warning:
        print(warning, file=sys.stderr)

    records = list(TranscriptReader(transcript))
    if args.limit:
        records = records[: args.limit]
    if not records:
        print(f"{transcript} contains no episodes", file=sys.stderr)
        return 2

    out = args.out or (config.run_dir / f"codes.{judge_name}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    # A free-tier judge paces at a few requests a minute, so a corpus is hours
    # of wall clock. Anything that long has to survive being interrupted:
    # already-coded episodes are skipped and each one is flushed as it lands.
    done = {c.episode_id for c in _read_codes(out)} if out.exists() else set()
    todo = [r for r in records if r.episode_id not in done]
    if done:
        print(
            f"resuming: {len(done)} episodes already coded, {len(todo)} to go",
            file=sys.stderr,
        )
    if not todo:
        print(f"nothing to code; {out} is complete")
        codes = _read_codes(out)
    else:
        stats = CodingStats()
        codes = asyncio.run(_code_all(backend, todo, judge_name, stats, out))
        print(f"coded {len(todo)} episodes -> {out}: {stats.summary()}")
        codes = _read_codes(out)

    report = summarize(codes)
    print(json.dumps(report.to_dict(), indent=2))
    print(
        f"\ndifferentiation {report.differentiation:.4f} vs null "
        f"{report.null_mean:.4f} (p={report.p_value:.4f}) | "
        f"consistency {report.consistency:.3f}"
    )
    if report.p_value > 0.05:
        print(
            "  Differentiation is indistinguishable from the permutation null: "
            "on this corpus the transcripts do not show role structure, so a "
            "Phase 4 convergence result would have nothing to converge on.",
            file=sys.stderr,
        )
    if stats.unparseable > stats.coded * 0.05:
        print(
            f"  WARNING: {stats.unparseable} replies were unparseable and fell "
            "back to 'other'. Check the judge model and prompt before trusting "
            "these labels.",
            file=sys.stderr,
        )
    return 0


def _build_judge(args: argparse.Namespace, config: ExperimentConfig):
    """Returns (backend, warning). A None backend means refuse to run."""
    if args.judge == "self":
        return config.backend.build(), (
            "judge: reusing the experiment backend. PLAN.md Phase 2 rates a "
            "local 7-8B as an inadequate judge -- smoke tests only."
        )
    if args.judge == "mock":
        return MockBackend(mode=MockMode.SPECIALIZED), "judge: mock (labels are fake)"
    if args.judge == "anthropic":
        from collabengine.backends.anthropic_judge import (
            DEFAULT_JUDGE_MODEL,
            AnthropicJudgeBackend,
        )

        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None, (
                "no ANTHROPIC_API_KEY set, so the frontier judge cannot run.\n"
                "Set the key, or pass `--judge self` to code with the local "
                "model (adequate for a pipeline smoke test, not for a reported "
                "kappa or any Phase 4 number)."
            )
        return (
            AnthropicJudgeBackend(model=args.judge_model or DEFAULT_JUDGE_MODEL),
            None,
        )
    if args.judge == "gemini":
        from collabengine.backends.gemini_judge import (
            DEFAULT_JUDGE_MODEL as GEMINI_DEFAULT,
            GeminiJudgeBackend,
        )

        if not os.environ.get("GEMINI_API_KEY"):
            return None, (
                "no GEMINI_API_KEY set, so the frontier judge cannot run.\n"
                "Set the key, or pass `--judge self` to code with the local "
                "model (a pipeline smoke test, not a reportable number)."
            )
        return (
            GeminiJudgeBackend(
                model=args.judge_model or GEMINI_DEFAULT,
                requests_per_minute=args.judge_rpm,
            ),
            f"judge: {args.judge_model or GEMINI_DEFAULT} at {args.judge_rpm}/min "
            "-- free-tier pacing, so a large corpus takes hours. The run "
            "checkpoints after every episode and resumes where it stopped.",
        )
    return None, f"unknown judge {args.judge!r}; expected gemini|anthropic|self|mock"


async def _code_all(backend, records, judge_name: str, stats: CodingStats, out: Path):
    """Code episode by episode, flushing each one before starting the next.

    Per-episode rather than per-message so a resumed run never has to reason
    about a half-coded episode: an episode id is either absent from the file or
    complete in it.
    """
    codes: list = []
    with out.open("a", encoding="utf-8", newline="\n") as fh:
        for index, record in enumerate(records, start=1):
            batch = await code_episode(
                backend=backend, record=record, judge_name=judge_name, stats=stats
            )
            for code in batch:
                fh.write(json.dumps(code.to_dict()) + "\n")
            fh.flush()
            codes.extend(batch)
            if index % 5 == 0 or index == len(records):
                print(
                    f"  {index}/{len(records)} episodes | {stats.summary()}",
                    file=sys.stderr,
                    flush=True,
                )
    return codes


def cmd_kappa(args: argparse.Namespace) -> int:
    """Cohen's kappa between two coding runs over the same messages.

    2604.00026 reported 0.78. Below that, the labels are too noisy to carry a
    convergent-validity claim and the honest move is to report the kappa and
    say so rather than to report the correlation anyway.
    """
    first = {c.key: c for c in _read_codes(args.first)}
    second = {c.key: c for c in _read_codes(args.second)}
    shared = sorted(set(first) & set(second))

    if not shared:
        print("the two files share no coded messages", file=sys.stderr)
        return 2

    kappa = cohens_kappa(
        [first[k].action for k in shared], [second[k].action for k in shared]
    )
    raw = sum(1 for k in shared if first[k].action is second[k].action) / len(shared)

    print(f"messages compared: {len(shared)}")
    print(f"raw agreement:     {raw:.3f}")
    print(f"Cohen's kappa:     {kappa:.3f}")
    if kappa < 0.78:
        print(
            "  Below the 0.78 reported by 2604.00026. Report this number "
            "alongside any claim these labels support.",
            file=sys.stderr,
        )
    return 0


def _read_codes(path: Path) -> list[MessageCode]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [MessageCode.from_dict(json.loads(line)) for line in fh if line.strip()]


def cmd_converge(args: argparse.Namespace) -> int:
    """Phase 4: do the transcript labels predict causal contribution?

    Either answer is a result. A weak correlation is the stronger form of the
    project's thesis and independently replicates *Agents that Matter*' finding
    that introspective judgment diverges from ablation.
    """
    config = ExperimentConfig.load(args.config)
    baseline_path = config.run_dir / BASELINE
    ablation_path = config.run_dir / ABLATION

    for path in (baseline_path, ablation_path):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 2

    matrix = _live_matrix(baseline_path, ablation_path)
    if matrix is None:
        print("no live-ablation records found", file=sys.stderr)
        return 2

    codes = _read_codes(args.codes)
    report = convergent_validity(codes, matrix, n_permutations=args.permutations)

    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nr = {report.r:+.3f} (p={report.p_value:.4f}) -- {report.verdict}")
    return 0


def _live_matrix(baseline_path: Path, ablation_path: Path) -> AblationMatrix | None:
    base = _component_means(TranscriptReader(baseline_path))
    per_agent: dict[str, dict[Component, list[float]]] = {}
    for record in TranscriptReader(ablation_path):
        if not record.condition.startswith("live:"):
            continue
        agent = record.condition.split(":", 1)[1]
        bucket = per_agent.setdefault(agent, {c: [] for c in ALL_COMPONENTS})
        for comp, val in record.grade.per_component.items():
            bucket[comp].append(val)

    if not per_agent:
        return None
    ablated = {
        agent: {c: statistics.mean(v) for c, v in comps.items() if v}
        for agent, comps in per_agent.items()
    }
    return AblationMatrix.from_means(base, ablated)


def cmd_selftest(args: argparse.Namespace) -> int:
    """Reproduce the instrument-validation table without a GPU."""
    from collabengine.orchestrator.team import SymmetryBreaking

    agents = ("A1", "A2", "A3", "A4")
    ownership = {c: agents[i] for i, c in enumerate(ALL_COMPONENTS)}
    team = TeamConfig(
        n_agents=4,
        rounds=3,
        difficulty="hard",
        symmetry=SymmetryBreaking.NAME_SEED,
        randomize_turn_order=True,
    )

    print(f"{'world':<14}{'dominance':>11}{'strength':>11}   (chance 0.25)")
    for mode in (MockMode.SPECIALIZED, MockMode.POSITIONAL, MockMode.NULL):
        backend = MockBackend(mode=mode, competence=0.5, off_focus_competence=0.0)

        async def means(exclude: tuple[str, ...]) -> dict[Component, float]:
            recs = await asyncio.gather(
                *(
                    run_episode(
                        backend=backend,
                        config=team,
                        episode_seed=s,
                        exclude=exclude,
                        condition="selftest",
                    )
                    for s in range(args.episodes)
                )
            )
            return {
                c: statistics.mean([r.grade.per_component[c] for r in recs])
                for c in ALL_COMPONENTS
            }

        base = asyncio.run(means(()))
        ablated = {a: asyncio.run(means((a,))) for a in agents}
        report = analyze_interaction(
            AblationMatrix.from_means(base, ablated), ownership
        )
        print(
            f"{mode.value:<14}{report.diagonal_dominance:>11.2f}"
            f"{report.interaction_strength:>11.4f}"
        )

    print(
        "\nSpecialized should dominate at ~1.00; positional should sit at chance;\n"
        "null should be near zero. Anything else means the instrument is\n"
        "manufacturing structure and must be fixed before spending GPU time."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
