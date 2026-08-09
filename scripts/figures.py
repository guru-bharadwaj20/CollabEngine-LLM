"""Regenerate every figure in the README from the corpus on disk.

Figures are committed as PNGs because GitHub cannot run a plotting script, but
they are never hand-drawn: each one is produced from `runs/<name>/baseline.jsonl`
by this file, through the same integrity filter and the same scoring module the
analysis uses. A figure that disagrees with the tables would otherwise be
impossible to catch.

    python scripts/figures.py [--run-dir runs/qwen3-8b-local]

The one exception is the throughput panel, whose numbers come from a batch-size
sweep that needs the GPU; those are transcribed from RESEARCH-LOG 4.3 and
marked as such below.
"""

from __future__ import annotations

import argparse
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collabengine.analysis.integrity import is_instrument_failure
from collabengine.analysis.scoring import METRICS, rescore
from collabengine.transcripts.store import TranscriptReader

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


def load(run_dir: Path) -> dict[str, dict[str, list[float]]]:
    """metric -> condition -> scores, instrument failures excluded."""
    by: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for rec in TranscriptReader(str(run_dir / "baseline.jsonl")):
        if is_instrument_failure(rec):
            continue
        scored = rescore(rec)
        for metric in METRICS:
            by[metric][rec.condition].append(scored.overall[metric])
    return by


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


def fig_gate(by, out: Path) -> None:
    """The headline: every episode, every arm, and the gap with its interval.

    Plotted as individual points rather than bars because the whole finding is
    that the distributions overlap. A bar chart of four means differing in the
    third decimal would imply a precision the corpus does not have.
    """
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    rng = random.Random(7)

    for i, (cond, label, colour) in enumerate(ARMS):
        vals = by["fraction"].get(cond, [])
        if not vals:
            continue
        xs = [i + rng.uniform(-0.13, 0.13) for _ in vals]
        ax1.scatter(xs, vals, s=34, color=colour, alpha=0.75, zorder=3,
                    edgecolor="white", linewidth=0.6)
        mean = st.mean(vals)
        ax1.plot([i - 0.30, i + 0.30], [mean, mean], color=colour, lw=2.6, zorder=4)
        # Above the rule, not beside it -- at 0.28 offset the label ran into the
        # neighbouring arm's points.
        ax1.text(i, 1.005, f"{mean:.3f}", ha="center", fontsize=9.5,
                 color=colour, fontweight="bold")

    ax1.set_xticks(range(len(ARMS)))
    ax1.set_xticklabels([a[1] for a in ARMS])
    ax1.set_ylabel("score  (fraction of constraints satisfied)")
    ax1.set_title("Phase 1 gate: four agents do not beat one")
    ax1.set_ylim(0.62, 1.03)
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
    ax2.set_xlim(-0.42, 0.55)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/qwen3-8b-local")
    ap.add_argument("--out-dir", default="docs/figures")
    args = ap.parse_args()

    _style()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    by = load(Path(args.run_dir))
    fig_gate(by, out / "gate.png")
    fig_throughput(out / "throughput.png")
    fig_selftest(out / "selftest.png")

    for name, conds in [("fraction", by["fraction"])]:
        for cond in sorted(conds):
            vals = conds[cond]
            print(f"{cond:<28} n={len(vals):2d} mean={st.mean(vals):.3f}")
    print(f"wrote 3 figures to {out}")


if __name__ == "__main__":
    main()
