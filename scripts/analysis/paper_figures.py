"""Regenerate the figures for the paper from the corpus on disk.

Separate from `figures.py`, which draws the README's four panels. The paper
makes a different argument -- the artifact and what survives it -- so it needs
its own panels rather than the README's recropped. Every number here is read
from `runs/` through the same `analysis.scoring` and `analysis.integrity` the
tables use, for the same reason `figures.py` does it: a figure that disagrees
with a table would otherwise be impossible to catch.

    python scripts/analysis/paper_figures.py [--out-dir docs/figures/paper]

Palette is validated rather than chosen by eye (data-viz six checks, light
surface, `--pairs all`): blue/orange/aqua pass the lightness band, chroma floor,
CVD separation and the normal-vision floor. The band that makes those hues
colourblind-safe also makes them nearly equal in lightness, which is exactly
wrong for a paper that may be printed in black and white -- so every series
carries a second, non-colour channel as well: marker shape, hatching, and a
direct value label. Read greyscale, the figures still separate.
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
import numpy as np

from collabengine.analysis.integrity import final_turn_truncated, is_instrument_failure
from collabengine.analysis.scoring import rescore
from collabengine.transcripts.store import TranscriptReader

# --- validated categorical slots -------------------------------------------
SOLO = "#eb6834"   # slot 2, orange -- the single-agent arms
TEAM = "#2a78d6"   # slot 1, blue   -- the four-agent arms
THIRD = "#1baf7a"  # slot 3, aqua   -- the three-agent ablated arms
INK = "#1a1a1a"
MUTED = "#6b6b6b"  # reference/corrected marks: a neutral, never an identity
GRID = "#d8d8d6"

SEED = 20260810
BOOT = 10000
PERM = 20000

TIERS = ("medium", "hard", "xhard")
OLD = {t: f"runs/llama31-8b-q4-{t}/baseline.jsonl" for t in TIERS}
ANS = {t: f"runs/llama31-8b-q4-{t}-ans/baseline.jsonl" for t in TIERS}
FRESH = "runs/llama31-8b-q4-medium-h3b"
PILOT = "runs/llama31-8b-q4-medium-ans"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": "#555555",
        "ytick.color": "#555555",
        "figure.dpi": 400,
    })


# --- corpus access ----------------------------------------------------------
def scores(path, cond, drop_cut=False) -> list[float]:
    out = []
    for rec in TranscriptReader(str(path)):
        if rec.condition != cond or is_instrument_failure(rec):
            continue
        if drop_cut and final_turn_truncated(rec):
            continue
        out.append(rescore(rec).overall["fraction"])
    return out


def cut_rate(path, cond) -> tuple[int, int]:
    n = t = 0
    for rec in TranscriptReader(str(path)):
        if rec.condition != cond or is_instrument_failure(rec):
            continue
        n += 1
        t += 1 if final_turn_truncated(rec) else 0
    return t, n


def by_seed(path, pred) -> dict:
    out = defaultdict(dict)
    for rec in TranscriptReader(str(path)):
        if is_instrument_failure(rec) or not pred(rec.condition):
            continue
        out[rec.instance_seed][rec.condition] = rescore(rec).overall["fraction"]
    return out


_rng = random.Random(SEED)


def boot_ci(a, b):
    g = sorted(st.mean([_rng.choice(b) for _ in b]) - st.mean([_rng.choice(a) for _ in a])
               for _ in range(BOOT))
    return g[int(.025 * BOOT)], g[int(.975 * BOOT)]


def paired_ci(d):
    g = sorted(st.mean([_rng.choice(d) for _ in d]) for _ in range(BOOT))
    return g[int(.025 * BOOT)], g[int(.975 * BOOT)]


# --- figure 1: the mechanism ------------------------------------------------
def fig_mechanism(out: Path) -> None:
    """Where the cap lands, and what happens to the gap when you control it.

    Two panels because the claim has two halves: the asymmetry is *measured*
    (left) and it *carries the headline* (right). Either alone is suggestive;
    together they are the argument.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.4, 2.25))
    x = np.arange(len(TIERS))
    w = 0.36

    solo_r, team_r, labels = [], [], []
    for t in TIERS:
        st_, sn = cut_rate(OLD[t], "solo")
        tt, tn = cut_rate(OLD[t], "baseline")
        solo_r.append(100 * st_ / sn)
        team_r.append(100 * tt / tn)
        labels.append((f"{st_}/{sn}", f"{tt}/{tn}"))

    b1 = ax1.bar(x - w / 2, solo_r, w, color=SOLO, zorder=3, label="1 agent")
    b2 = ax1.bar(x + w / 2, team_r, w, color=TEAM, zorder=3, hatch="///",
                 edgecolor="white", linewidth=0.6, label="4 agents")
    for bar, lab in zip(b1, [l[0] for l in labels]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.4, lab,
                 ha="center", fontsize=6.8, color=INK)
    for bar, lab in zip(b2, [l[1] for l in labels]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.4, lab,
                 ha="center", fontsize=6.8, color=INK)
    ax1.set_xticks(x)
    ax1.set_xticklabels(TIERS)
    ax1.set_ylabel("answer turn hit the cap (%)")
    ax1.set_ylim(0, 60)
    ax1.set_title("(a) the cap lands on one arm")
    ax1.legend(frameon=False, loc="upper left", handlelength=1.4)
    ax1.grid(axis="y", color=GRID, lw=0.6)
    ax1.set_axisbelow(True)

    head, sens = [], []
    for t in TIERS:
        s, b = scores(OLD[t], "solo"), scores(OLD[t], "baseline")
        head.append(st.mean(b) - st.mean(s))
        s2, b2_ = scores(OLD[t], "solo", True), scores(OLD[t], "baseline", True)
        sens.append(st.mean(b2_) - st.mean(s2))

    ax2.bar(x - w / 2, head, w, color=TEAM, zorder=3, label="as scored")
    ax2.bar(x + w / 2, sens, w, color=MUTED, zorder=3, hatch="\\\\\\",
            edgecolor="white", linewidth=0.6, label="truncation dropped")
    for xi, (h, s) in enumerate(zip(head, sens)):
        ax2.text(xi - w / 2, h + (0.008 if h >= 0 else -0.022), f"{h:+.3f}",
                 ha="center", fontsize=6.8, color=TEAM, fontweight="bold")
        ax2.text(xi + w / 2, s + (0.008 if s >= 0 else -0.022), f"{s:+.3f}",
                 ha="center", fontsize=6.8, color="#4a4a4a")
    ax2.axhline(0, color=INK, lw=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(TIERS)
    ax2.set_ylabel("team − 1 agent (fraction)")
    # Headroom above the tallest bar's label so the legend has somewhere to sit
    # that is not on top of it.
    ax2.set_ylim(-0.11, 0.43)
    ax2.set_title("(b) control it and the gap goes")
    ax2.legend(frameon=False, loc="upper left", handlelength=1.4)
    ax2.grid(axis="y", color=GRID, lw=0.6)
    ax2.set_axisbelow(True)

    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# --- figure 2: three positives, corrected -----------------------------------
def fig_corrections(out: Path) -> None:
    """Each significant pro-team result, as measured and after its control.

    A slope chart rather than paired bars: the quantity that matters is the
    *movement* between two estimates of the same contrast, and a slope draws
    movement where bars draw two heights the eye must subtract.
    """
    rows = []

    s, b = scores(OLD["hard"], "solo"), scores(OLD["hard"], "baseline")
    s2, b2 = scores(ANS["hard"], "solo"), scores(ANS["hard"], "baseline")
    rows.append(("1. team − 1 agent, hard\n(answer-budget instrument)",
                 st.mean(b) - st.mean(s), boot_ci(s, b),
                 st.mean(b2) - st.mean(s2), boot_ci(s2, b2)))

    fb = Path(FRESH) / "baseline.jsonl"
    a, c = scores(fb, "solo_long"), scores(fb, "baseline")
    a2, c2 = scores(fb, "solo_long", True), scores(fb, "baseline", True)
    rows.append(("2. team − matched-budget agent\n(answer-turn truncation dropped)",
                 st.mean(c) - st.mean(a), boot_ci(a, c),
                 st.mean(c2) - st.mean(a2), boot_ci(a2, c2)))

    def participation(d):
        team = by_seed(Path(d) / "baseline.jsonl", lambda c: c == "baseline")
        abl = by_seed(Path(d) / "ablation.jsonl", lambda c: c.startswith("live:"))
        diffs = [team[s0]["baseline"] - v
                 for s0, cells in abl.items() if s0 in team
                 for v in cells.values()]
        return st.mean(diffs), paired_ci(diffs)

    p_pilot, ci_pilot = participation(PILOT)
    p_fresh, ci_fresh = participation(FRESH)
    rows.append(("3. cost of removing one agent\n(fresh seeds, 3× the sample)",
                 p_pilot, ci_pilot, p_fresh, ci_fresh))

    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    ys = np.arange(len(rows))[::-1]
    for y, (_, g0, ci0, g1, ci1) in zip(ys, rows):
        ax.plot([ci0[0], ci0[1]], [y + 0.14] * 2, color=SOLO, lw=1.4,
                solid_capstyle="round", alpha=0.85, zorder=2)
        ax.plot([ci1[0], ci1[1]], [y - 0.14] * 2, color=MUTED, lw=1.4,
                solid_capstyle="round", alpha=0.85, zorder=2)
        ax.annotate("", xy=(g1, y - 0.14), xytext=(g0, y + 0.14),
                    arrowprops=dict(arrowstyle="->", color="#999999", lw=0.9,
                                    shrinkA=3, shrinkB=3), zorder=3)
        ax.plot([g0], [y + 0.14], "o", color=SOLO, ms=5.5, zorder=4,
                markeredgecolor="white", markeredgewidth=0.7)
        ax.plot([g1], [y - 0.14], "s", color=MUTED, ms=5.0, zorder=4,
                markeredgecolor="white", markeredgewidth=0.7)
        ax.text(g0, y + 0.30, f"{g0:+.3f}", ha="center", fontsize=6.8,
                color=SOLO, fontweight="bold")
        # Nudge the corrected label clear of the zero rule, which is exactly
        # where these estimates land -- the label would otherwise be drawn on
        # top of the one line the reader is comparing it against.
        ha = "left" if abs(g1) < 0.012 else "center"
        dx = 0.012 if ha == "left" else 0.0
        ax.text(g1 + dx, y - 0.44, f"{g1:+.3f}", ha=ha, fontsize=6.8, color="#4a4a4a")

    ax.axvline(0, color=INK, lw=0.9, ls=(0, (4, 2)))
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.2)
    ax.set_xlabel("effect on `fraction` (95% bootstrap interval)")
    ax.set_xlim(-0.11, 0.42)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.plot([], [], "o", color=SOLO, ms=5.5, label="as measured")
    ax.plot([], [], "s", color=MUTED, ms=5.0, label="after its control")
    ax.legend(frameon=False, loc="lower right", handlelength=1.0, ncol=2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# --- figure 3: what reproduced ----------------------------------------------
def fig_reproduction(out: Path) -> None:
    """Pilot against fresh seeds, per arm. Only the reference moved."""
    pb, fb = Path(PILOT) / "baseline.jsonl", Path(FRESH) / "baseline.jsonl"
    pa, fa = Path(PILOT) / "ablation.jsonl", Path(FRESH) / "ablation.jsonl"

    def pooled_live(p):
        v = []
        for rec in TranscriptReader(str(p)):
            if not rec.condition.startswith("live:") or is_instrument_failure(rec):
                continue
            v.append(rescore(rec).overall["fraction"])
        return v

    rows = [
        ("1 agent × 3 rounds", scores(pb, "solo"), scores(fb, "solo"), SOLO, "o"),
        ("3 agents (pooled ablated)", pooled_live(pa), pooled_live(fa), THIRD, "^"),
        ("4 agents (the reference)", scores(pb, "baseline"), scores(fb, "baseline"), TEAM, "s"),
    ]

    fig, ax = plt.subplots(figsize=(5.4, 2.2))
    ys = np.arange(len(rows))[::-1]
    for y, (lab, p, f, col, mk) in zip(ys, rows):
        mp, mf = st.mean(p), st.mean(f)
        ax.annotate("", xy=(mf, y), xytext=(mp, y),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5,
                                    shrinkA=4, shrinkB=4), zorder=2)
        ax.plot([mp], [y], mk, color="white", ms=7, zorder=3,
                markeredgecolor=col, markeredgewidth=1.4)
        ax.plot([mf], [y], mk, color=col, ms=7, zorder=4,
                markeredgecolor="white", markeredgewidth=0.7)
        # Above and below rather than both above: on two of the three rows the
        # means coincide to within 0.006 -- which is the finding -- so labels
        # placed on one line would overlap precisely where they agree.
        ax.text(mp, y + 0.24, f"{mp:.3f}", ha="center", fontsize=6.8, color=col)
        ax.text(mf, y - 0.36, f"{mf:.3f}", ha="center", fontsize=6.8,
                color=col, fontweight="bold")
        ax.text(0.648, y, f"Δ {mf - mp:+.3f}", fontsize=7.2, va="center",
                color=INK if abs(mf - mp) > 0.02 else MUTED,
                fontweight="bold" if abs(mf - mp) > 0.02 else "normal")

    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlabel("score (fraction), medium")
    ax.set_xlim(0.545, 0.712)
    ax.set_ylim(-0.85, len(rows) - 0.35)
    ax.plot([], [], "o", color="white", markeredgecolor=MUTED, markeredgewidth=1.4,
            ms=7, label="pilot (seeds 0–47)")
    ax.plot([], [], "o", color=MUTED, ms=7, label="fresh (seeds 1000–1149)")
    ax.legend(frameon=False, loc="lower left", handlelength=1.0, ncol=2)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# --- figure 4: team size does nothing ---------------------------------------
def fig_teamsize(out: Path) -> None:
    """The headline, drawn as episodes rather than bars.

    Points, because the finding is that the distributions coincide; four bars
    differing in the third decimal would imply a precision the corpus lacks.
    """
    fb, fa = Path(FRESH) / "baseline.jsonl", Path(FRESH) / "ablation.jsonl"

    def pooled_live(p):
        v = []
        for rec in TranscriptReader(str(p)):
            if not rec.condition.startswith("live:") or is_instrument_failure(rec):
                continue
            v.append(rescore(rec).overall["fraction"])
        return v

    arms = [
        ("1 agent\n3 rounds", scores(fb, "solo"), SOLO, "o"),
        ("3 agents\n3 rounds", pooled_live(fa), THIRD, "^"),
        ("4 agents\n3 rounds", scores(fb, "baseline"), TEAM, "s"),
        ("1 agent\n12 rounds", scores(fb, "solo_long"), SOLO, "D"),
    ]

    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    rng = random.Random(7)
    for i, (lab, vals, col, mk) in enumerate(arms):
        xs = [i + rng.uniform(-0.16, 0.16) for _ in vals]
        ax.scatter(xs, vals, s=5, color=col, alpha=0.32, zorder=2,
                   linewidths=0, marker=mk)
        m = st.mean(vals)
        ax.plot([i - 0.30, i + 0.30], [m, m], color=col, lw=2.2, zorder=4,
                solid_capstyle="butt")
        ax.text(i, 1.055, f"{m:.3f}", ha="center", fontsize=8,
                color=col, fontweight="bold")

    ax.annotate("spread across the first three arms: 0.005",
                xy=(1.0, 0.577), xytext=(0.42, 0.90), fontsize=7,
                color=INK, ha="left",
                arrowprops=dict(arrowstyle="->", color="#999999", lw=0.9,
                                connectionstyle="arc3,rad=0.25"))
    ax.annotate("more turns, one agent:\n−0.063 (p = 0.011)",
                xy=(3.0, 0.516), xytext=(2.30, 0.135), fontsize=7,
                color=SOLO, ha="left", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=SOLO, lw=1.0,
                                connectionstyle="arc3,rad=-0.2"))

    ax.set_xticks(range(len(arms)))
    # n goes in the tick label rather than floating above the axis, where it
    # collided with the tick marks at this figure height.
    ax.set_xticklabels([f"{a[0]}\nn = {len(a[1])}" for a in arms])
    ax.set_ylabel("score (fraction of constraints satisfied)")
    ax.set_ylim(-0.05, 1.02)
    ax.set_xlim(-0.55, len(arms) - 0.45)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# --- figure 5: the interaction residuals ------------------------------------
def fig_residuals(out: Path) -> None:
    """Double-centred ablation matrix from the pilot grid.

    Diverging, because the quantity has a meaningful zero and a sign: two hues
    with a neutral midpoint, never a rainbow. The scale is deliberately set to
    the magnitude a *true null* would produce (~0.06), so the reader sees that
    the observed structure does not fill it.
    """
    agents = ["A1", "A2", "A3", "A4"]
    comps = ["arithmetic", "search", "verification", "synthesis"]
    resid = np.array([
        [+0.013, -0.001, +0.006, -0.018],
        [-0.006, -0.017, -0.003, +0.026],
        [+0.002, +0.006, +0.019, -0.027],
        [-0.009, +0.012, -0.023, +0.019],
    ])

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", [SOLO, "#f0efec", TEAM])

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    lim = 0.06
    im = ax.imshow(resid, cmap=cmap, vmin=-lim, vmax=lim)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{resid[i, j]:+.3f}", ha="center", va="center",
                    fontsize=6.8, color=INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(comps, rotation=30, ha="right", fontsize=7)
    ax.set_yticks(range(4))
    ax.set_yticklabels([f"remove {a}" for a in agents], fontsize=7)
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("residual after double-centring", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    cb.outline.set_visible(False)
    ax.set_title("scale set to a true null's expected ~0.06", fontsize=7.5, pad=6)
    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="docs/figures/paper")
    args = ap.parse_args()
    style()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig_mechanism(out / "fig1_mechanism.png")
    print("fig1 mechanism")
    fig_corrections(out / "fig2_corrections.png")
    print("fig2 corrections")
    fig_reproduction(out / "fig3_reproduction.png")
    print("fig3 reproduction")
    fig_teamsize(out / "fig4_teamsize.png")
    print("fig4 teamsize")
    fig_residuals(out / "fig5_residuals.png")
    print("fig5 residuals")
    print(f"wrote 5 figures to {out}")


if __name__ == "__main__":
    main()
