"""What each arm could see, computed before it runs rather than after it.

This project reported *post-hoc* power in its prereg postscripts. Post-hoc power
is a deterministic function of the observed *p*-value and carries no information
beyond it; computing it after a null result and finding it low is arithmetic,
not diagnosis. What those paragraphs were reaching for is one of two things,
both of which are here and neither of which needs the result:

  MDE     the smallest effect an arm of size n could detect, from a PRIOR sd
  n_for   the episodes per arm needed to detect an effect of interest

**The number that would have prevented this project's largest error.** The
withdrawn participation finding was +0.055 read off a 48-episode arm with
`sd` ~= 0.15. The MDE there is 0.086. One line of this script, run while the
grid was being sized rather than after it failed to replicate, would have said
that the arm could not resolve the effect it was about to report
(RESEARCH-LOG 4.22).

    python scripts/analysis/power_report.py                 # the whole table
    python scripts/analysis/power_report.py --sd 0.12       # a different prior
    python scripts/analysis/power_report.py --run-dir runs/x # realised sd instead

With `--run-dir` the prior is replaced by the realised per-arm `sd` measured
from that corpus, which is what a *later* study should use. Without it the
script uses the published prior, so it works with no corpus on disk -- which is
the current state of this repository (Final Sweep 1.0.a).
"""

from __future__ import annotations

import argparse
import statistics as st
from collections import defaultdict
from pathlib import Path

from collabengine.analysis.inference import mde, n_for, plan

#: Realised spread on `fraction`, four-agent arm, from the fresh-seed corpus.
#: Published in RESEARCH-LOG 4.22 as the quantity whose absence produced a
#: headline. Anyone sizing a study like this one should start here rather than
#: guess, and guessing it low is exactly how a 48-episode arm carries a result.
SD_PRIOR = 0.15

#: Every arm size this project has actually run, the effect measured on it, and
#: whether that effect was *reported as a finding*. The last flag is what makes
#: the table readable: an arm smaller than its own MDE is exactly what should
#: produce a null, and is only damning when a positive was read off it.
ARMS: tuple[tuple[str, int, float, bool], ...] = (
    ("Phase 1 gate, per tier", 24, 0.249, True),
    ("gate at n=48", 48, 0.045, True),
    ("ablation pilot baseline", 48, 0.055, True),
    ("C5 matched budget", 150, 0.060, True),
    ("fresh-seed gate", 150, 0.003, False),
    ("pooled 3-agent arms", 599, 0.002, False),
    ("all arms, fresh seeds", 899, 0.005, False),
)

#: The runs Final Sweep section 2 proposes, sized before they are run. This is
#: the half of the discipline that was missing: a planned arm gets its MDE
#: written into the preregistration, not discovered afterwards.
PLANNED: tuple[tuple[str, int], ...] = (
    ("14B grid, per arm", 150),
    ("second model family, per arm", 150),
    ("precision ladder, per rung", 150),
    ("second task family, per arm", 150),
    ("dose-response, per cap value", 100),
)

#: Effects worth being able to detect, and what each costs in episodes per arm.
TARGETS: tuple[tuple[str, float], ...] = (
    ("one constraint (the equivalence margin)", 0.050),
    ("the withdrawn participation effect", 0.055),
    ("half a constraint", 0.025),
    ("the observed spread across team sizes", 0.005),
)


def realised_sd(run_dir: Path) -> dict[str, tuple[int, float]]:
    """Per-condition (n, sd) on `fraction` from a corpus, if one exists."""
    from collabengine.analysis.integrity import is_instrument_failure
    from collabengine.analysis.scoring import rescore
    from collabengine.transcripts.store import TranscriptReader

    by_cond: dict[str, list[float]] = defaultdict(list)
    for path in sorted(run_dir.glob("*.jsonl")):
        for rec in TranscriptReader(str(path)):
            if is_instrument_failure(rec):
                continue
            by_cond[rec.condition].append(rescore(rec).overall["fraction"])
    return {
        cond: (len(v), st.stdev(v) if len(v) > 1 else 0.0)
        for cond, v in sorted(by_cond.items())
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd", type=float, default=SD_PRIOR)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.8)
    ap.add_argument("--run-dir", help="measure sd from this corpus instead of the prior")
    args = ap.parse_args()

    sd = args.sd
    print(f"{'=' * 74}\nA-PRIORI POWER  (alpha = {args.alpha}, "
          f"power = {args.power:.0%}, sd = {sd:.3f})\n{'=' * 74}")

    if args.run_dir:
        d = Path(args.run_dir)
        if d.exists():
            print("\n  realised spread per arm -- the prior a later study should use")
            print(f"  {'condition':<24}{'n':>6}{'sd':>8}{'MDE':>8}")
            for cond, (n, s) in realised_sd(d).items():
                m = mde(n, s, args.alpha, args.power) if n > 1 and s > 0 else float("nan")
                print(f"  {cond:<24}{n:>6}{s:>8.3f}{m:>8.3f}")
        else:
            print(f"\n  no corpus at {d} -- using the published prior instead")

    print("\n  arms this project ran, against what each was used to claim")
    print(f"  {'arm':<28}{'n/arm':>7}{'MDE':>8}{'effect':>9}   verdict")
    for label, n, effect, reported in ARMS:
        m = mde(n, sd, args.alpha, args.power)
        if effect >= m:
            verdict = "resolvable"
        elif reported:
            verdict = "REPORTED BELOW ITS OWN MDE"
        else:
            verdict = "null, correctly (see PREREG-equivalence for the bound)"
        print(f"  {label:<28}{n:>7}{m:>8.3f}{effect:>+9.3f}   {verdict}")
    print("\n  An arm below its own MDE is exactly what should produce a null, so")
    print("  the bottom three rows are not a criticism -- they are the result,")
    print("  and PREREG-equivalence turns each into a bound. The middle rows are")
    print("  the criticism: a positive finding read off an arm that could not")
    print("  have resolved it. All of those were later withdrawn.")

    print("\n  runs proposed in Final Sweep section 2, sized before they run")
    print(f"  {'run':<32}{'n/arm':>7}{'MDE':>8}")
    for label, n in PLANNED:
        print(f"  {plan(label, n, sd, args.alpha, args.power).label:<32}"
              f"{n:>7}{mde(n, sd, args.alpha, args.power):>8.3f}")

    print("\n  what it costs to be able to see an effect of a given size")
    print(f"  {'effect':<42}{'delta':>7}{'n/arm':>8}")
    for label, delta in TARGETS:
        print(f"  {label:<42}{delta:>7.3f}{n_for(delta, sd, args.alpha, args.power):>8}")
    print("\n  Read the last row before proposing to 'run more episodes'. Resolving")
    print("  a difference the size of the one actually observed across team sizes")
    print("  would take corpora this card cannot produce, which is a reason to")
    print("  report a bound (PREREG-equivalence) rather than to keep sampling.")


if __name__ == "__main__":
    main()
