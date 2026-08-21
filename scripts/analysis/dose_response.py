"""The token-cap artifact as a curve: apparent team advantage vs verbosity ratio.

The paper currently characterises the artifact at two points -- a cap of 1024
where the team appeared to beat one agent by d = 1.09 at `hard`, and a cap of
3072 where the same contrast is +0.003 (RESEARCH-LOG 4.9-4.11). Two points
support "we found a bug". They do not support "we characterise when the bug
appears and how large it gets", and the second claim is the one that makes an
audit of somebody else's system possible: given their cap and the verbosity
ratio between their arms, this curve estimates how much of their reported
advantage is instrument.

**The x-axis is the ratio, not the cap**, and that is the substantive claim
being tested. A cap is not comparable across systems -- it interacts with the
tokenizer, the task, the brief. What should transfer is the relationship between
the cap and *what the arms actually write*: the artifact needs the cap to fall
between the two arms' answer lengths, so the arm that writes more loses score
the other keeps. The cap column is printed beside it because this study's rungs
are caps, but the row that generalises is the ratio one.

Reads the corpora `scripts/experiments/cap-sweep.sh` writes and reports, per
rung: the verbosity ratio, the apparent advantage, its 95% bootstrap CI, the
answer-turn truncation counts that are the proposed mechanism, and the same
advantage with truncated episodes dropped. The last two columns are what
separate "there is a curve" from "there is a curve *because of truncation*" --
a rung whose advantage survives dropping every truncated episode was not
produced by the cap, whatever the shape says.

Silent about rungs that have not run. A missing rung is "not measured", never
"no effect", and an interpolated curve through the gap would be neither.

    python scripts/analysis/dose_response.py
    python scripts/analysis/dose_response.py --runs-dir runs --metric fraction
"""

from __future__ import annotations

import argparse
import random
import re
import statistics as st
from pathlib import Path

from collabengine.analysis.inference import mde, smallest_equivalence_bound
from collabengine.analysis.integrity import (
    final_turn_truncated,
    is_instrument_failure,
)
from collabengine.analysis.scoring import rescore
from collabengine.transcripts.store import TranscriptReader

BOOTSTRAPS = 10000
SEED = 20260821

#: Run directories are named `...-capNNNN`, zero-padded so that the directory
#: listing sorts into cap order. The number in the name is the cap; it is not
#: re-read from the config on purpose, so a config edited after the run cannot
#: silently relabel a corpus that was generated under a different budget.
RUN_PATTERN = re.compile(r"-cap(\d{4})$")

#: The prior registered in docs/PREREG-cap-sweep.md, pooled across the two arms
#: of the fresh-seed `medium` corpus (baseline sd 0.098, solo sd 0.252). Used
#: only for the MDE column, which says what each rung could have seen and is
#: therefore the column that keeps a small gap from being read as a small
#: effect.
SD_PRIOR = 0.191


def arm(path: Path, condition: str, metric: str, drop_truncated: bool = False):
    """(scores, answer-turn chars, whole-episode chars, answer-turn truncations).

    **Two verbosity measures, and the answer-turn one is the primary.** The cap
    is a per-turn budget, so what it can truncate is one turn: the last
    agent-authored message, which is the one carrying the submitted answer. The
    whole-episode figure is the quantity RESEARCH-LOG 4.18 reports (1.87x for
    C5 against the team) and it is dominated by turn count -- a 3-round solo arm
    writes three turns against the team's twelve, so its episode total is
    *smaller* while its answer is longer. Ranking rungs on the episode total
    would therefore put the arms in the wrong order for the one comparison the
    cap actually makes. It is printed beside the answer-turn ratio rather than
    dropped, because a reader who has 4.18's number in hand needs to see that
    this is a different one and why.
    """
    scores: list[float] = []
    answer_chars: list[float] = []
    all_chars: list[float] = []
    truncated = 0
    for rec in TranscriptReader(str(path)):
        if rec.condition != condition or is_instrument_failure(rec):
            continue
        cut = final_turn_truncated(rec)
        truncated += cut
        if drop_truncated and cut:
            continue
        body = [m for m in rec.messages if m.is_ablatable()]
        if not body:
            continue
        scores.append(rescore(rec).overall[metric])
        answer_chars.append(len(body[-1].content))
        all_chars.append(sum(len(m.content) for m in body))
    return scores, answer_chars, all_chars, truncated


def boot_ci(a: list[float], b: list[float], rng: random.Random) -> tuple[float, float]:
    """95% percentile bootstrap on mean(b) - mean(a), gate_report's method.

    Same procedure as scripts/analysis/gate_report.py rather than a normal
    interval, because the solo arm's spread is roughly 2.5x the team's and grows
    further as the cap shortens -- which is the assumption a t-interval makes and
    this design deliberately violates.
    """
    gaps = sorted(
        st.mean([rng.choice(b) for _ in b]) - st.mean([rng.choice(a) for _ in a])
        for _ in range(BOOTSTRAPS)
    )
    return gaps[int(0.025 * BOOTSTRAPS)], gaps[int(0.975 * BOOTSTRAPS)]


def rungs(runs_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for d in sorted(runs_dir.glob("*-cap[0-9][0-9][0-9][0-9]")):
        m = RUN_PATTERN.search(d.name)
        if m and (d / "baseline.jsonl").exists():
            out.append((int(m.group(1)), d / "baseline.jsonl"))
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--metric", default="fraction",
                    help="scored metric for the advantage column")
    ap.add_argument("--team", default="baseline", help="the multi-agent arm")
    ap.add_argument("--solo", default="solo", help="the single-agent arm")
    args = ap.parse_args()

    found = rungs(Path(args.runs_dir))
    if not found:
        print(f"no cap-sweep corpora in {args.runs_dir}/. Nothing has run yet:\n"
              f"    CTX=64512 PARALLEL=3 scripts/ops/serve.sh --detach\n"
              f"    bash scripts/experiments/cap-sweep.sh")
        return

    rng = random.Random(SEED)
    print(f"{'=' * 96}\nDOSE-RESPONSE: apparent team advantage on `{args.metric}` "
          f"as a function of the cap\n{'=' * 96}")
    print(f"  {'cap':>6}{'n team':>8}{'n solo':>8}{'answer s/t':>12}{'episode s/t':>13}"
          f"{'advantage':>11}{'95% CI':>18}{'MDE':>7}{'trunc t/s':>11}{'cut adv':>9}")

    rows = []
    for cap, path in found:
        team, team_ans, team_all, team_cut = arm(path, args.team, args.metric)
        solo, solo_ans, solo_all, solo_cut = arm(path, args.solo, args.metric)
        if min(len(team), len(solo)) < 5:
            print(f"  {cap:>6}   too few usable episodes to read "
                  f"(team {len(team)}, solo {len(solo)})")
            continue
        ratio = st.mean(solo_ans) / st.mean(team_ans) if st.mean(team_ans) else float("nan")
        ratio_all = st.mean(solo_all) / st.mean(team_all) if st.mean(team_all) else float("nan")
        adv = st.mean(team) - st.mean(solo)
        lo, hi = boot_ci(solo, team, rng)
        m = mde(min(len(team), len(solo)), SD_PRIOR)

        # The same contrast with every answer-turn-truncated episode dropped.
        # If this column tracks the advantage column, the gap is not the cap's
        # doing and the mechanism registered in the prereg is wrong.
        cut_team, _, _, _ = arm(path, args.team, args.metric, drop_truncated=True)
        cut_solo, _, _, _ = arm(path, args.solo, args.metric, drop_truncated=True)
        cut_adv = (
            st.mean(cut_team) - st.mean(cut_solo)
            if min(len(cut_team), len(cut_solo)) >= 5 else float("nan")
        )

        print(f"  {cap:>6}{len(team):>8}{len(solo):>8}{ratio:>12.2f}{ratio_all:>13.2f}"
              f"{adv:>+11.3f}{f'[{lo:+.3f}, {hi:+.3f}]':>18}{m:>7.3f}"
              f"{f'{team_cut}/{solo_cut}':>11}{cut_adv:>+9.3f}")
        rows.append((cap, ratio, adv, lo, hi, team_cut, solo_cut, cut_adv,
                     solo, team))

    if not rows:
        return

    print("\n  `advantage` is team minus one agent, so positive means the team")
    print("  looks better. `answer s/t` is the solo-to-team ratio of ANSWER-turn")
    print("  characters -- the quantity a per-turn cap can actually cut -- while")
    print("  `episode s/t` is the whole-episode ratio of RESEARCH-LOG 4.18, which")
    print("  runs the other way because a 3-round solo arm writes three turns")
    print("  against the team's twelve. `trunc t/s` counts answer-turn truncations")
    print("  per arm; `cut adv` is the advantage with those episodes dropped. An")
    print("  advantage that survives `cut adv` was not produced by the cap.")

    # The row that generalises. Ordered by ratio rather than by cap, because a
    # cap is an instrument setting and a ratio is a property of the arms -- and
    # only the second is something another study can look up for itself.
    print(f"\n{'=' * 96}\nTHE TRANSFERABLE CURVE: advantage against verbosity ratio"
          f"\n{'=' * 96}")
    print(f"  {'answer chars s/t':>18}{'cap':>7}{'advantage':>11}{'95% CI':>18}"
          f"{'excludes 0':>12}{'equiv bound':>13}")
    for cap, ratio, adv, lo, hi, _, _, _, solo, team in sorted(rows, key=lambda r: r[1]):
        bound = smallest_equivalence_bound(solo, team)
        print(f"  {ratio:>18.2f}{cap:>7}{adv:>+11.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>18}"
              f"{('yes' if lo > 0 or hi < 0 else 'no'):>12}{bound:>13.3f}")
    print("\n  `equiv bound` is the smallest delta at which this rung's two arms")
    print("  are equivalent (PREREG-equivalence). It is the number to report for")
    print("  a rung whose interval contains zero: 'effects larger than this are")
    print("  excluded', which is a result, where 'not significant' is not.")

    # Registered in PREREG-cap-sweep section 4 as the confirming pattern. Printed
    # as an arithmetic check rather than a verdict: it says whether the shape
    # matches, and the prereg says what that is allowed to mean.
    short = [r for r in rows if r[0] <= 1024]
    long_ = [r for r in rows if r[0] >= 3072]
    if short and long_:
        drop = max(r[2] for r in short) - max(r[2] for r in long_)
        print(f"\n  registered contrast (PREREG-cap-sweep H-C1): largest advantage")
        print(f"  at cap <= 1024 minus largest at cap >= 3072 = {drop:+.3f}")
        print(f"  Threshold registered before the run: >= +0.10, with the short-cap")
        print(f"  interval excluding zero and its truncation counts asymmetric.")


if __name__ == "__main__":
    main()
