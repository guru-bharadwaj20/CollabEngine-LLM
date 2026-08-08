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
import statistics
import sys
from pathlib import Path

from collabengine.ablation import (
    capacity_control,
    frozen_excise,
    frozen_replay,
    live_ablation,
)
from collabengine.ablation.modes import propagation_index
from collabengine.analysis import AblationMatrix, analyze_interaction
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="collabengine", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("calibrate", help="difficulty sweep (Phase 1)")
    p.add_argument("--config", type=Path)
    p.add_argument("--episodes", type=int, default=20)

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

    p = sub.add_parser("selftest", help="validate the instrument on mock worlds")
    p.add_argument("--episodes", type=int, default=30)

    args = parser.parse_args(argv)
    return {
        "calibrate": cmd_calibrate,
        "baseline": cmd_baseline,
        "ablate": cmd_ablate,
        "analyze": cmd_analyze,
        "selftest": cmd_selftest,
    }[args.command](args)


def _load(path: Path | None) -> ExperimentConfig:
    return ExperimentConfig.load(path) if path else ExperimentConfig()


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Sweep difficulty and report the score band per preset.

    If no preset lands between the floor and the ceiling, the task needs
    redesigning before anything downstream is worth running.
    """
    config = _load(args.config)
    backend = config.backend.build()

    print(f"{'difficulty':<10} {'mean':>7} {'sd':>7} {'min':>6} {'max':>6}")
    for difficulty in sorted(PRESETS, key=lambda d: PRESETS[d].n_jobs):
        team = TeamConfig.from_dict({**config.team.to_dict(), "difficulty": difficulty})
        scores = asyncio.run(_score_many(backend, team, args.episodes))
        print(
            f"{difficulty:<10} {statistics.mean(scores):>7.3f} "
            f"{statistics.pstdev(scores):>7.3f} {min(scores):>6.2f} {max(scores):>6.2f}"
        )
    return 0


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

    records = {r.instance_seed: r for r in TranscriptReader(baseline_path)}
    modes = {m.strip() for m in args.modes.split(",") if m.strip()}
    agents = [a.agent_id for a in build_team(config.team, config.seed_start)]

    plans: list[RunPlan] = []
    for seed, record in records.items():
        if "capacity" in modes:
            plans.append(
                RunPlan(
                    f"capacity:{seed}",
                    lambda seed=seed: capacity_control(
                        backend=backend, config=config.team, episode_seed=seed
                    ),
                )
            )
        for agent in agents:
            if "live" in modes:
                plans.append(
                    RunPlan(
                        f"live:{agent}:{seed}",
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
                        f"frozen_replay:{agent}:{seed}",
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
                        f"frozen_excise:{agent}:{seed}",
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
                        f"random_message:{agent}:{seed}",
                        lambda rec=record, agent=agent: _immediate(
                            random_message_control(rec, agent, seed=config.seed_start)
                        ),
                    )
                )

    stats = asyncio.run(
        run_plan(
            plans,
            out_path=config.run_dir / ABLATION,
            max_concurrency=config.max_concurrency,
            on_progress=lambda s: print(f"  {s.summary()}", file=sys.stderr),
        )
    )
    print(f"ablation -> {config.run_dir / ABLATION}: {stats.summary()}")
    _report_propagation(records.values(), agents)
    return 0 if stats.failed == 0 else 1


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
        print("no live-ablation records found", file=sys.stderr)
        return 2

    ablated = {
        agent: {c: statistics.mean(v) for c, v in comps.items() if v}
        for agent, comps in per_agent.items()
    }
    matrix = AblationMatrix.from_means(base, ablated)

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
    return 0


def _component_means(reader: TranscriptReader) -> dict[Component, float]:
    acc: dict[Component, list[float]] = {c: [] for c in ALL_COMPONENTS}
    for record in reader:
        for comp, val in record.grade.per_component.items():
            acc[comp].append(val)
    return {c: (statistics.mean(v) if v else 0.0) for c, v in acc.items()}


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
