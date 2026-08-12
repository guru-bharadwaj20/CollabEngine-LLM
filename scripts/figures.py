"""Regenerate every figure in the README from the corpus on disk.

Figures are committed as PNGs because GitHub cannot run a plotting script, but
they are never hand-drawn: each one is produced from `runs/<name>/baseline.jsonl`
by this file, through the same integrity filter and the same scoring module the
analysis uses. A figure that disagrees with the tables would otherwise be
impossible to catch.

    python scripts/figures.py [--run-dir runs/llama31-8b-q4-hard]

The one exception is the throughput panel, whose numbers come from a batch-size
sweep that needs the GPU; those are transcribed from RESEARCH-LOG 4.3 and
marked as such below.
"""

from __future__ import annotations

import argparse
import random
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collabengine.analysis.integrity import (
    final_turn_truncated,
    is_instrument_failure,
)
from collabengine.analysis.scoring import METRICS, rescore
from collabengine.transcripts.store import TranscriptReader


def _run_dir(config_path: str) -> Path:
    """Read a tier's run directory from its config rather than hardcoding it.

    Hardcoded paths in this file have been wrong twice: once pointing at a
    stale solo arm, which overstated a gap by 0.031 in a published figure, and
    once at directories that no longer existed after the move to the served
    instrument. A figure that looks plausible does not get checked the way a
    table does, so it should not be the place a path is repeated.
    """
    from collabengine.config import ExperimentConfig

    return ExperimentConfig.load(config_path).run_dir

# One palette across every figure. Team arms share a hue so that the eye groups
# them against solo, which is the comparison every panel is making.
SOLO = "#c1440e"
TEAM = "#1f4e79"
TEAM_LIGHT = "#7ba7cc"
GREY = "#8a8a8a"
RULE = "#d0d0d0"

ARMS = [
    ("solo", "solo\n1 agent", SOLO),
    ("baseline", "baseline\n4 agents", TEAM),
    ("symmetry:name_seed_scratch", "symmetry\nname+seed", TEAM_LIGHT),
    ("fixed_order", "fixed order\ncontrol", TEAM_LIGHT),
]


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.labelcolor": "#222222",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.color": "#222222",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "figure.dpi": 150,
        }
    )


def load(run_dir: Path, name: str = "baseline.jsonl") -> dict[str, dict[str, list[float]]]:
    """metric -> condition -> scores, instrument failures excluded."""
    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in TranscriptReader(str(run_dir / name)):
        if is_instrument_failure(rec):
            continue
        scored = rescore(rec)
        for metric in METRICS:
            by[metric][rec.condition].append(scored.overall[metric])
    return by


def load_cut_flags(run_dir: Path, name: str = "baseline.jsonl") -> dict[str, list[bool]]:
    """condition -> per-episode "answer turn hit the cap", in `load` order.

    Kept in step with `load` by applying the same filter in the same pass order;
    the two are zipped in the figure, so a divergence would mislabel points
    rather than fail.
    """
    flags: dict[str, list[bool]] = defaultdict(list)
    for rec in TranscriptReader(str(run_dir / name)):
        if is_instrument_failure(rec):
            continue
        flags[rec.condition].append(final_turn_truncated(rec))
    return flags


def boot_ci(a: list[float], b: list[float], n: int = 10000, seed: int = 20260809):
    """Percentile bootstrap on mean(b) - mean(a)."""
    rng = random.Random(seed)
    gaps = []
    for _ in range(n):
        ra = [rng.choice(a) for _ in a]
        rb = [rng.choice(b) for _ in b]
        gaps.append(st.mean(rb) - st.mean(ra))
    gaps.sort()
    return gaps[int(0.025 * n)], gaps[int(0.975 * n)]


def perm_p(a: list[float], b: list[float], n: int = 10000, seed: int = 20260809) -> float:
    """Two-sided permutation test on the difference in means."""
    rng = random.Random(seed)
    obs = abs(st.mean(b) - st.mean(a))
    pool = a + b
    hits = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(st.mean(pool[len(a):]) - st.mean(pool[: len(a)])) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (n + 1)


def fig_gate(by, out: Path, cut_flags: dict[str, list[bool]] | None = None) -> None:
    """The headline: every episode, every arm, and the gap with its interval.

    Plotted as individual points rather than bars because the whole finding is
    that the distributions overlap. A bar chart of four means differing in the
    third decimal would imply a precision the corpus does not have.
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    rng = random.Random(7)
    means: list[tuple[int, float, str]] = []

    drawn: list[float] = []
    marked = False
    for i, (cond, label, colour) in enumerate(ARMS):
        vals = by["fraction"].get(cond, [])
        if not vals:
            continue
        drawn += vals
        cut = cut_flags.get(cond, []) if cut_flags else []
        xs = [i + rng.uniform(-0.13, 0.13) for _ in vals]
        # Episodes whose answer-bearing turn hit the cap are drawn hollow. They
        # are not a separate population to be read past: they are ordinary
        # episodes the instrument cut off, and on the served corpus they are
        # every zero in the solo arm (RESEARCH-LOG 4.9).
        for x, v, is_cut in zip(xs, vals, cut or [False] * len(vals)):
            if is_cut:
                marked = True
                ax1.scatter([x], [v], s=44, facecolor="none", edgecolor=colour,
                            linewidth=1.6, zorder=3)
            else:
                ax1.scatter([x], [v], s=34, color=colour, alpha=0.75, zorder=3,
                            edgecolor="white", linewidth=0.6)
        mean = st.mean(vals)
        ax1.plot([i - 0.30, i + 0.30], [mean, mean], color=colour, lw=2.6, zorder=4)
        means.append((i, mean, colour))

    # Limits from the data, not from the instrument that produced an earlier
    # corpus. The hardcoded 0.62 floor here predated the served run and would
    # have clipped all four of its zero-scoring solo episodes off the axis --
    # a figure that hides precisely the episodes the result turns on.
    lo_v, hi_v = (min(drawn), max(drawn)) if drawn else (0.0, 1.0)
    pad = max(0.04, 0.08 * (hi_v - lo_v))
    top = min(1.03, hi_v + pad)
    ax1.set_ylim(max(-0.03, lo_v - pad), top)
    # In points above the axis rather than in data units: the offset used to be
    # a data-space constant, which lands in a different place on every corpus
    # and ran the arm means into the title once the y-limits became data-driven.
    for i, mean, colour in means:
        ax1.annotate(f"{mean:.3f}", xy=(i, top), xytext=(0, 4),
                     textcoords="offset points", ha="center", va="bottom",
                     fontsize=9.5, color=colour, fontweight="bold",
                     annotation_clip=False)

    ax1.set_xticks(range(len(ARMS)))
    ax1.set_xticklabels([a[1] for a in ARMS])
    ax1.set_ylabel("score  (fraction of constraints satisfied)")
    ax1.set_title("Phase 1 gate: four agents do not beat one", pad=20)
    if marked:
        ax1.scatter([], [], s=44, facecolor="none", edgecolor=GREY, linewidth=1.6,
                    label="answer turn hit the token cap")
        ax1.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax1.grid(axis="y", color=RULE, lw=0.7)
    ax1.set_axisbelow(True)

    # Right panel: the gap against solo, per metric, with its interval.
    solo_by_metric = {m: by[m]["solo"] for m in METRICS}
    rows = []
    for metric in METRICS:
        team = by[metric]["baseline"]
        solo = solo_by_metric[metric]
        gap = st.mean(team) - st.mean(solo)
        lo, hi = boot_ci(solo, team)
        rows.append((metric, gap, lo, hi, perm_p(solo, team)))

    ys = range(len(rows))
    for y, (metric, gap, lo, hi, p) in zip(ys, rows):
        ax2.plot([lo, hi], [y, y], color=TEAM, lw=2.2, solid_capstyle="round")
        ax2.plot([gap], [y], "o", color=TEAM, ms=8, zorder=3)
        ax2.text(hi + 0.02, y, f"p = {p:.2f}", va="center", fontsize=9, color=GREY)

    ax2.axvline(0, color="#333333", lw=1.2, ls="--")
    ax2.set_yticks(list(ys))
    ax2.set_yticklabels([r[0] for r in rows])
    ax2.set_xlabel("team − solo  (95% bootstrap interval)")
    ax2.set_title("Every interval spans zero")
    span = max(abs(v) for r in rows for v in (r[2], r[3])) if rows else 0.5
    ax2.set_xlim(-span - 0.10, span + 0.22)  # room on the right for the p labels
    ax2.invert_yaxis()
    ax2.grid(axis="x", color=RULE, lw=0.7)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_throughput(out: Path) -> None:
    """Batch-size sweep. Numbers from RESEARCH-LOG 4.3 -- these need the GPU."""
    batch = [8, 16, 32, 48]
    tok_s = [72.6, 94.1, 108.1, 71.1]
    peak = [16.8, 18.3, 21.4, 24.4]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bars = ax.bar([str(b) for b in batch], tok_s,
                  color=[TEAM, TEAM, "#2e7d32", SOLO], width=0.62, zorder=3)
    for bar, val, gib in zip(bars, tok_s, peak):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2.0, f"{val:.0f}",
                ha="center", fontsize=10, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 4, f"{gib} GiB",
                ha="center", fontsize=8.5, color="white")

    ax.annotate(
        "paging to host RAM\n(nvidia-smi still reads 100%)",
        xy=(2.80, 76), xytext=(-0.38, 114),
        fontsize=9, color=SOLO, ha="left",
        arrowprops=dict(arrowstyle="->", color=SOLO, lw=1.4,
                        connectionstyle="arc3,rad=-0.2"),
    )
    ax.set_xlabel("batch size  (1600-token context)")
    ax.set_ylabel("aggregate tok/s")
    ax.set_title("Throughput collapses before the card reports a problem")
    ax.set_ylim(0, 125)
    ax.grid(axis="y", color=RULE, lw=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_selftest(out: Path) -> None:
    """Three synthetic worlds with known answers. Figures from `selftest`."""
    worlds = ["specialized", "positional", "null"]
    dominance = [1.00, 0.25, 0.25]
    interaction = [0.171, 0.015, 0.008]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    colours = ["#2e7d32", "#b8860b", GREY]

    ax1.bar(worlds, dominance, color=colours, width=0.55, zorder=3)
    ax1.axhline(0.25, color=SOLO, ls="--", lw=1.4, zorder=4)
    ax1.text(-0.42, 0.30, "chance = 1/n_agents", color=SOLO, fontsize=8.5,
             ha="left")
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("diagonal dominance")
    ax1.set_title("Separates specialized from chance")

    ax2.bar(worlds, interaction, color=colours, width=0.55, zorder=3)
    ax2.set_ylabel("interaction strength")
    ax2.set_title("Separates positional from null")

    for ax in (ax1, ax2):
        ax.grid(axis="y", color=RULE, lw=0.7)
        ax.set_axisbelow(True)

    fig.suptitle("Instrument validation: worlds whose answer is known in advance",
                 fontsize=11, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


TIER_LABELS = {
    "medium": "medium\n16 jobs, 5 workers",
    "hard": "hard\n24 jobs, 6 workers",
    "xhard": "xhard\n36 jobs, 8 workers",
}


def fig_curve(_unused, out: Path) -> bool:
    """The difficulty curve, and the reason it cannot be read on its own.

    Every tier comes from its own config-derived run directory. The previous
    version took one tier from `--run-dir` and plotted it as `hard` against
    `medium` read from a config -- so running the script against the medium
    directory silently plotted medium as both points. The right panel was worse:
    three transcribed bf16 constants that no longer described any corpus on
    disk. Both are the same mistake as the stale solo path, and a figure gets
    that mistake past review more easily than a table does.

    The right panel is now the methodological point, computed rather than
    transcribed: the headline gap against the gap with answer-turn truncation
    dropped. The headline trends upward across difficulty and the sensitivity
    does not, because the cap falls on solo's answer turn and not the team's
    (RESEARCH-LOG 4.10).
    """
    tiers = []
    for tier in ("medium", "hard", "xhard"):
        cfg = f"configs/llamacpp-{tier}.yaml"
        if not Path(cfg).exists():
            continue
        run_dir = _run_dir(cfg)
        if not (run_dir / "baseline.jsonl").exists():
            continue
        # A tier still generating has both arms present and neither finished,
        # and its means move for as long as it runs. Plotted anyway it is a
        # point on a published curve that nothing on disk will reproduce an
        # hour later -- caught here after a mid-run regeneration drew `xhard`
        # from 25 of its 48 episodes.
        if not _tier_complete(cfg, run_dir):
            print(f"{tier}: incomplete, excluded from curve.png")
            continue
        by = load(run_dir)
        if not (by["fraction"].get("solo") and by["fraction"].get("baseline")):
            continue
        tiers.append((tier, run_dir, by))

    if len(tiers) < 2:
        print("need at least two tiers with both arms; skipping curve.png")
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.0))
    xs = list(range(len(tiers)))
    solo = [st.mean(by["fraction"]["solo"]) for _, _, by in tiers]
    team = [st.mean(by["fraction"]["baseline"]) for _, _, by in tiers]

    ax1.plot(xs, solo, "o-", color=SOLO, lw=2.4, ms=9, label="solo (1 agent)")
    ax1.plot(xs, team, "s-", color=TEAM, lw=2.4, ms=9, label="team (4 agents)")
    for x, (a, b) in enumerate(zip(solo, team)):
        side = 34 if x == 0 else -34   # keep labels off the axis edges
        ax1.annotate(f"{a:.3f}", (x, a), textcoords="offset points",
                     xytext=(side, -12), ha="center", fontsize=9, color=SOLO)
        ax1.annotate(f"{b:.3f}", (x, b), textcoords="offset points",
                     xytext=(side, 8), ha="center", fontsize=9, color=TEAM)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([TIER_LABELS.get(t, t) for t, _, _ in tiers])
    ax1.set_ylabel("score (fraction)")
    lo, hi = min(solo + team), max(solo + team)
    pad = max(0.03, 0.15 * (hi - lo))
    ax1.set_ylim(max(0.0, lo - pad), min(1.0, hi + pad))
    # "Widens" was wrong once xhard landed: the headline gap runs +0.067,
    # +0.249, +0.188. It is not monotonic, because the preregistered filter
    # starts excluding solo's all-turns-truncated episodes at the hard end and
    # lifts solo's mean back up (RESEARCH-LOG 4.11). Describe the panel, not the
    # story the first two points suggested.
    ax1.set_title("As scored, solo falls away faster")
    ax1.legend(frameon=False, fontsize=9, loc="lower left")
    ax1.grid(axis="y", color=RULE, lw=0.7)
    ax1.set_axisbelow(True)

    head = [t - s for s, t in zip(solo, team)]
    sens = []
    for _, run_dir, _ in tiers:
        kept = _drop_cut(run_dir)
        sens.append(
            st.mean(kept["baseline"]) - st.mean(kept["solo"])
            if kept["solo"] and kept["baseline"] else float("nan")
        )

    width = 0.36
    ax2.bar([x - width / 2 for x in xs], head, width, color=TEAM, zorder=3,
            label="as scored")
    ax2.bar([x + width / 2 for x in xs], sens, width, color=GREY, zorder=3,
            label="answer-turn truncation dropped")
    for x, (h, s) in enumerate(zip(head, sens)):
        ax2.text(x - width / 2, h + 0.006, f"{h:+.3f}", ha="center", fontsize=8.5,
                 color=TEAM, fontweight="bold")
        if s == s:  # not NaN
            ax2.text(x + width / 2, s + 0.006, f"{s:+.3f}", ha="center",
                     fontsize=8.5, color="#5f5f5f")
    ax2.axhline(0, color="#333333", lw=1.1)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([t for t, _, _ in tiers])
    ax2.set_ylabel("team − solo (fraction)")
    ax2.set_title("Control the answer-turn cap and it goes")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2.grid(axis="y", color=RULE, lw=0.7)
    ax2.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return True


def _tier_complete(config_path: str, run_dir: Path, name: str = "baseline.jsonl") -> bool:
    """Both arms present at the episode count the config asks for.

    Counted before the integrity filter: a genuine instrument failure should not
    make a finished tier look unfinished, and it is already accounted for in the
    `unusable` column of every report.
    """
    from collabengine.config import ExperimentConfig

    want = ExperimentConfig.load(config_path).n_episodes
    seen: Counter[str] = Counter()
    for rec in TranscriptReader(str(run_dir / name)):
        seen[rec.condition] += 1
    return seen["solo"] >= want and seen["baseline"] >= want


def _drop_cut(run_dir: Path, name: str = "baseline.jsonl") -> dict[str, list[float]]:
    """`fraction` per condition with answer-turn-truncated episodes removed."""
    out: dict[str, list[float]] = defaultdict(list)
    for rec in TranscriptReader(str(run_dir / name)):
        if is_instrument_failure(rec) or final_turn_truncated(rec):
            continue
        out[rec.condition].append(rescore(rec).overall["fraction"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(_run_dir("configs/llamacpp-hard.yaml")))
    ap.add_argument("--out-dir", default="docs/figures")
    args = ap.parse_args()

    _style()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # The corpus-independent panels always regenerate. The two that need a
    # corpus are skipped with a note when the tier has not run yet, because
    # this script is run repeatedly while tiers are still landing and a
    # traceback there reads as "the figures are broken" rather than "the data
    # is not in yet".
    fig_throughput(out / "throughput.png")
    fig_selftest(out / "selftest.png")
    written = 2

    corpus = Path(args.run_dir) / "baseline.jsonl"
    if not corpus.exists():
        print(f"no corpus at {corpus}; skipping gate.png and curve.png")
        print(f"wrote {written} figures to {out}")
        return

    by = load(Path(args.run_dir))
    fig_gate(by, out / "gate.png", load_cut_flags(Path(args.run_dir)))
    written += 1
    if fig_curve(by, out / "curve.png"):
        written += 1

    for cond in sorted(by["fraction"]):
        vals = by["fraction"][cond]
        print(f"{cond:<28} n={len(vals):2d} mean={st.mean(vals):.3f}")
    print(f"wrote {written} figures to {out}")


if __name__ == "__main__":
    main()
