"""The Phase 1 gate, reported the same way every time.

Every gate number in this project has so far been produced by a one-off script
written at the moment it was needed. That is exactly how the retracted result
in RESEARCH-LOG 4.1 happened: an ad-hoc script filtered on `malformed` only,
missed the errored turns the integrity module already knew about, and published
a mean computed over episodes the pipeline itself would have rejected. *A
result computed outside the pipeline's own validity checks is not a result.*

So this file is the one path. It applies `analysis.integrity`, scores through
`analysis.scoring`, and prints the integrity audit **before** the means, so a
censored arm cannot be read without seeing that it was censored.

    python scripts/gate_report.py                     # every corpus it can find
    python scripts/gate_report.py --run-dir runs/llama31-8b-q4-hard
"""

from __future__ import annotations

import argparse
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

from collabengine.analysis.integrity import (
    audit,
    final_turn_truncated,
    is_instrument_failure,
)
from collabengine.analysis.scoring import METRICS, rescore
from collabengine.transcripts.store import TranscriptReader

PERMUTATIONS = 20000
BOOTSTRAPS = 10000
SEED = 20260810

# (label, team corpus, solo corpus). Both arms of every point now live in the
# same run directory.
#
# `medium` briefly did not: its solo arm was reused from `runs/medium-corpus`
# while only the team arm had been regenerated. That was correct while it was
# true and became wrong the moment 24 fresh solo episodes were generated
# alongside the team arm -- the report went on reading the stale 12-episode
# file and printed `solo n=12, team n=21`, an arm mismatch it had no way to
# flag because both files were valid. Keep both arms in one directory.
#: Tier -> config. The run directory is read from the config rather than
#: written here, because a hardcoded path is exactly what went wrong twice: it
#: silently pointed at a stale arm (above), and then at directories that no
#: longer existed at all after the move to the served instrument. The config is
#: the one place the run directory is already declared.
TIER_CONFIGS = {
    "medium": "configs/llamacpp-medium.yaml",
    "hard": "configs/llamacpp-hard.yaml",
    "xhard": "configs/llamacpp-xhard.yaml",
}


def corpora() -> list[tuple[str, str, str]]:
    """(label, team path, solo path) for every tier whose config resolves.

    Both arms come from one run directory. They always should — the one time
    they did not, the report read a 12-episode solo arm against a 21-episode
    team arm and could not flag it, because both files were valid.
    """
    from collabengine.config import ExperimentConfig

    out = []
    for tier, cfg_path in TIER_CONFIGS.items():
        if not Path(cfg_path).exists():
            continue
        path = str(ExperimentConfig.load(cfg_path).run_dir / "baseline.jsonl")
        out.append((tier, path, path))
    return out


def scores(
    path: Path, condition: str, drop_final_truncated: bool = False
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    if not path.exists():
        return out
    for rec in TranscriptReader(str(path)):
        if rec.condition != condition or is_instrument_failure(rec):
            continue
        if drop_final_truncated and final_turn_truncated(rec):
            continue
        scored = rescore(rec)
        for metric in METRICS:
            out[metric].append(scored.overall[metric])
    return out


def generation(path: Path, condition: str) -> tuple[float, float] | None:
    """(agent turns, characters generated) per episode, averaged over usable ones.

    Exists because this report asserted for three sections that the matched-budget
    arms "spend the same generation", which is false and was never measured. They
    are matched on turn count and on the per-turn cap. They are not matched on
    what gets written: one agent restates the whole working solution every round,
    four agents each add to a shared transcript, so the single agent emits ~1.7x
    the team's text on the same nominal budget (RESEARCH-LOG 4.18).

    Characters, not tokens, and deliberately: it is reported as a ratio between
    two arms of one instrument, and tokenising the corpus to confirm a 1.7x would
    not change how the gap reads.
    """
    turns, chars = [], []
    for rec in TranscriptReader(str(path)):
        if rec.condition != condition or is_instrument_failure(rec):
            continue
        body = [m for m in rec.messages if m.is_ablatable()]
        turns.append(len(body))
        chars.append(sum(len(m.content) for m in body))
    return (st.mean(turns), st.mean(chars)) if turns else None


def perm_p(a: list[float], b: list[float], rng: random.Random) -> float:
    """Two-sided permutation test on the difference in means."""
    obs = abs(st.mean(b) - st.mean(a))
    pool = a + b
    hits = 0
    for _ in range(PERMUTATIONS):
        rng.shuffle(pool)
        if abs(st.mean(pool[len(a):]) - st.mean(pool[: len(a)])) >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (PERMUTATIONS + 1)


def boot_ci(a: list[float], b: list[float], rng: random.Random) -> tuple[float, float]:
    gaps = sorted(
        st.mean([rng.choice(b) for _ in b]) - st.mean([rng.choice(a) for _ in a])
        for _ in range(BOOTSTRAPS)
    )
    return gaps[int(0.025 * BOOTSTRAPS)], gaps[int(0.975 * BOOTSTRAPS)]


def report(label: str, team_path: Path, solo_path: Path, rng: random.Random) -> dict | None:
    if not team_path.exists():
        return None
    print(f"\n{'=' * 74}\n{label.upper()}\n{'=' * 74}")

    # Integrity first, always. A mean printed above its instrument-failure count
    # gets quoted without it.
    for line in audit(list(TranscriptReader(str(team_path)))).lines():
        print("  " + line)

    team = scores(team_path, "baseline")
    solo = scores(solo_path, "solo")
    if not team.get("fraction") or not solo.get("fraction"):
        print("  incomplete: need both arms")
        return None

    n_t, n_s = len(team["fraction"]), len(solo["fraction"])
    print(f"\n  usable: solo n={n_s}, team n={n_t}")
    if min(n_t, n_s) < 5:
        print("  VERDICT: none. Too few usable episodes per arm to read.")
        return None

    print(f"\n  {'metric':<11}{'solo':>8}{'team':>8}{'gap':>9}{'d':>7}"
          f"{'perm p':>9}{'95% CI':>22}")
    out = {}
    for metric in METRICS:
        a, b = solo[metric], team[metric]
        gap = st.mean(b) - st.mean(a)
        pooled = ((st.pstdev(a) ** 2 + st.pstdev(b) ** 2) / 2) ** 0.5
        d = gap / pooled if pooled else 0.0
        lo, hi = boot_ci(a, b, rng)
        p = perm_p(a, b, rng)
        out[metric] = (st.mean(a), st.mean(b), gap, d, p)
        print(f"  {metric:<11}{st.mean(a):>8.3f}{st.mean(b):>8.3f}{gap:>+9.3f}"
              f"{d:>+7.2f}{p:>9.3f}   [{lo:+.3f}, {hi:+.3f}]")

    best = max(abs(v[3]) for v in out.values())
    sig = [m for m, v in out.items() if v[4] < 0.05]
    print(f"\n  largest |d| = {best:.2f}; metrics significant at 0.05: "
          f"{', '.join(sig) if sig else 'none'}")

    _sensitivity(team_path, solo_path, out, rng)
    for condition, tag, brief in _MATCHED_ARMS:
        _matched_budget(team_path, out, rng, condition, tag, brief)
    _brief_effect(team_path, rng)
    return out


def _brief_effect(team_path: Path, rng: random.Random) -> None:
    """C5 minus C4: the same agent, the same budget, a different brief.

    This is the only place the brief is isolated. Both arms are one agent over
    twelve turns on one instrument, so every difference between them is the
    wording -- no phantom co-workers, no instruction that the group's last
    message is the answer of record, and an explicit licence not to restate the
    whole assignment every turn (RESEARCH-LOG 4.12).

    Reported as its own comparison rather than inferred from the two gaps to the
    team: those share the team arm, so differencing them would reuse one sample
    on both sides and understate the interval.
    """
    old, new = scores(team_path, "solo_budget"), scores(team_path, "solo_long")
    if not (old.get("fraction") and new.get("fraction")):
        return
    n_o, n_n = len(old["fraction"]), len(new["fraction"])
    print(f"\n  -- the brief alone: C5 - C4 (solo_long n={n_n}, solo_budget n={n_o}) --")
    if min(n_o, n_n) < 5:
        print("     too few usable episodes to read.")
        return
    print(f"     {'metric':<11}{'C4':>9}{'C5':>8}{'gap':>9}{'perm p':>9}")
    for metric in METRICS:
        a, b = old[metric], new[metric]
        print(f"     {metric:<11}{st.mean(a):>9.3f}{st.mean(b):>8.3f}"
              f"{st.mean(b) - st.mean(a):>+9.3f}{perm_p(a, b, rng):>9.3f}")


#: The two matched-budget arms, both one agent spending the team's whole turn
#: budget and differing only in which brief they were given. They go through the
#: same function rather than two similar ones on purpose: C4 and C5 are meant to
#: be read against each other, and two implementations of "the same comparison"
#: is how the difference between them stops being the brief.
_MATCHED_ARMS: tuple[tuple[str, str, str], ...] = (
    ("solo_budget", "C4", "TEAM_BRIEF, the brief written for a group"),
    ("solo_long", "C5", "SOLO_BRIEF, written for one agent"),
)


def _matched_budget(
    team_path: Path,
    headline: dict,
    rng: random.Random,
    condition: str = "solo_budget",
    tag: str = "C4",
    brief: str = "TEAM_BRIEF, the brief written for a group",
) -> None:
    """Team vs one agent holding the team's whole turn budget.

    Silent when the arm has not been generated, because it is off by default in
    the pipeline -- absence here means "not run", never "no difference".
    """
    budget = scores(team_path, condition)
    if not budget.get("fraction"):
        return
    team = scores(team_path, "baseline")
    n_b, n_t = len(budget["fraction"]), len(team["fraction"])

    print(f"\n  -- {tag}: matched turn budget, {brief}")
    print(f"     ({condition} n={n_b}, team n={n_t}) --")
    if min(n_b, n_t) < 5:
        print("     too few usable episodes to read.")
        return
    print(f"     {'metric':<11}{'1 agent':>9}{'team':>8}{'gap':>9}{'perm p':>9}"
          f"{'  vs vs-solo':>13}")
    for metric in METRICS:
        a, b = budget[metric], team[metric]
        gap = st.mean(b) - st.mean(a)
        p = perm_p(a, b, rng)
        print(f"     {metric:<11}{st.mean(a):>9.3f}{st.mean(b):>8.3f}{gap:>+9.3f}"
              f"{p:>9.3f}{gap - headline[metric][2]:>+13.3f}")
    gen_b, gen_t = generation(team_path, condition), generation(team_path, "baseline")
    if gen_b and gen_t and gen_t[1]:
        print(f"     Turns are matched: {gen_b[0]:.0f} agent turns against "
              f"{gen_t[0]:.0f}. Generation is")
        print(f"     not: {gen_b[1]:,.0f} chars against {gen_t[1]:,.0f}, a factor of "
              f"{gen_b[1] / gen_t[1]:.2f}x. One agent")
        print("     restates the working solution every round; four agents each add")
        print("     to a shared transcript. So read this as team against one agent")
        print("     at equal TURNS -- not at equal tokens, which it is not (4.18).")

    # C4 equalises the budget, which is not the same as removing the cap. Both
    # arms here run twelve turns, so if `solo_budget` still spends its last turn
    # writing the answer while the team spends its last turn restating one, the
    # asymmetry of 4.10 survives the fix and has to be reported the same way.
    cut_b = scores(team_path, condition, drop_final_truncated=True)
    cut_t = scores(team_path, "baseline", drop_final_truncated=True)
    if not (cut_b.get("fraction") and cut_t.get("fraction")):
        return
    dropped_b, dropped_t = n_b - len(cut_b["fraction"]), n_t - len(cut_t["fraction"])
    if not (dropped_b or dropped_t):
        print("     No answer-turn truncation in either arm: the budget fix")
        print("     removed the asymmetry rather than merely balancing it.")
        return
    print(f"     ...same, answer-turn truncation dropped "
          f"(1 agent -{dropped_b}, team -{dropped_t}): ", end="")
    if min(len(cut_b["fraction"]), len(cut_t["fraction"])) < 5:
        print("too few left to read.")
        return
    g = st.mean(cut_t["fraction"]) - st.mean(cut_b["fraction"])
    print(f"fraction {g:+.3f} (p={perm_p(cut_b['fraction'], cut_t['fraction'], rng):.3f})")


def _sensitivity(
    team_path: Path, solo_path: Path, headline: dict, rng: random.Random
) -> None:
    """Re-report with answer-bearing truncated turns dropped from both arms.

    Printed always, not on a flag, and printed under the headline rather than
    instead of it. The preregistered exclusion rule is the one above; this is
    the check on whether that rule's known asymmetry is carrying the result.
    Making it opt-in would mean the one number nobody runs is the one that
    could overturn the other.
    """
    team = scores(team_path, "baseline", drop_final_truncated=True)
    solo = scores(solo_path, "solo", drop_final_truncated=True)
    if not team.get("fraction") or not solo.get("fraction"):
        return
    n_t, n_s = len(team["fraction"]), len(solo["fraction"])
    dropped_t = len(headline_n(team_path, "baseline")) - n_t
    dropped_s = len(headline_n(solo_path, "solo")) - n_s
    if not (dropped_t or dropped_s):
        print("\n  sensitivity (drop final-turn truncation): nothing to drop.")
        return

    print(f"\n  -- sensitivity: final-turn truncation dropped "
          f"(solo -{dropped_s}, team -{dropped_t}) --")
    if min(n_t, n_s) < 5:
        print(f"     solo n={n_s}, team n={n_t}: too few left to read.")
        return
    print(f"     {'metric':<11}{'solo':>8}{'team':>8}{'gap':>9}{'perm p':>9}"
          f"{'  vs headline':>14}")
    for metric in METRICS:
        a, b = solo[metric], team[metric]
        gap = st.mean(b) - st.mean(a)
        p = perm_p(a, b, rng)
        moved = gap - headline[metric][2]
        print(f"     {metric:<11}{st.mean(a):>8.3f}{st.mean(b):>8.3f}{gap:>+9.3f}"
              f"{p:>9.3f}{moved:>+14.3f}")
    print("     A gap that changes sign or loses its ordering here was being")
    print("     carried by the cap, not by the condition. Neither row is the")
    print("     unbiased one: truncation is post-treatment, so this is a")
    print("     complete-case analysis and the two bracket rather than settle.")


def headline_n(path: Path, condition: str) -> list[float]:
    return scores(path, condition)["fraction"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", help="report a single run directory instead of all")
    args = ap.parse_args()

    rng = random.Random(SEED)
    sources = corpora()
    if args.run_dir:
        p = str(Path(args.run_dir) / "baseline.jsonl")
        sources = [(Path(args.run_dir).name, p, p)]
    if not sources:
        print("no corpora found; has anything run yet?")
        return

    results = {}
    paths = {}
    for label, team_p, solo_p in sources:
        got = report(label, Path(team_p), Path(solo_p), rng)
        if got:
            results[label] = got
            paths[label] = Path(team_p)

    # The difficulty curve, which is the actual hypothesis: does the team-solo
    # gap grow with instance size? Printed rather than tested -- with two or
    # three points and n<=24 per arm, a slope test would be theatre.
    if len(results) > 1:
        print(f"\n{'=' * 74}\nDIFFICULTY CURVE (fraction)\n{'=' * 74}")
        # cut@end is printed *inside* the curve, not left in the per-tier audits
        # above, because the trend in this column has the same shape as the
        # trend the hypothesis predicts. Harder instances need longer answers,
        # so solo's single answer-bearing turn hits the cap more often while the
        # team's summary turn does not -- an instrument that manufactures a
        # growing team advantage. A curve read without this column beside it
        # cannot be distinguished from that.
        print(f"  {'point':<10}{'solo':>8}{'team':>8}{'gap':>9}"
              f"{'cut@end s/t':>14}{'sens. gap':>11}")
        for label in ("medium", "hard", "xhard"):
            if label not in results:
                continue
            s, t, gap, _, _ = results[label]["fraction"]
            rec = list(TranscriptReader(str(paths[label])))
            cells = audit(rec).by_condition
            cut_s = cells.get("solo", None)
            cut_t = cells.get("baseline", None)
            cut = (f"{cut_s.final_truncated if cut_s else 0}"
                   f"/{cut_t.final_truncated if cut_t else 0}")
            sens_solo = scores(paths[label], "solo", drop_final_truncated=True)
            sens_team = scores(paths[label], "baseline", drop_final_truncated=True)
            if sens_solo.get("fraction") and sens_team.get("fraction"):
                sens = (f"{st.mean(sens_team['fraction']) - st.mean(sens_solo['fraction']):+.3f}")
            else:
                sens = "n/a"
            print(f"  {label:<10}{s:>8.3f}{t:>8.3f}{gap:>+9.3f}{cut:>14}{sens:>11}")
        print("\n  `sens. gap` is the same gap with answer-turn truncation")
        print("  dropped from both arms. If the headline gaps trend upward and")
        print("  the sensitivity gaps do not, the trend is the cap.")
        print("\n  Read the `unusable` column above beside every mean. On the")
        print("  bf16 instrument the prefill ceiling censored 28.9% of team")
        print("  turns at `hard` and 0% of solo turns, and the apparent gap")
        print("  tracked that censoring rather than any effect (RESEARCH-LOG")
        print("  4.1e, 4.6). The served instrument removes that failure mode")
        print("  but not the need to check: context overflow here is")
        print("  deterministic in the prompt, so it lands on the same long")
        print("  team turns, only labelled rather than silent.")


if __name__ == "__main__":
    main()
