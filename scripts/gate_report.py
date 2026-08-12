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

from collabengine.analysis.integrity import audit, is_instrument_failure
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


def scores(path: Path, condition: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    if not path.exists():
        return out
    for rec in TranscriptReader(str(path)):
        if rec.condition != condition or is_instrument_failure(rec):
            continue
        scored = rescore(rec)
        for metric in METRICS:
            out[metric].append(scored.overall[metric])
    return out


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
    return out


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
    for label, team_p, solo_p in sources:
        got = report(label, Path(team_p), Path(solo_p), rng)
        if got:
            results[label] = got

    # The difficulty curve, which is the actual hypothesis: does the team-solo
    # gap grow with instance size? Printed rather than tested -- with two or
    # three points and n<=24 per arm, a slope test would be theatre.
    if len(results) > 1:
        print(f"\n{'=' * 74}\nDIFFICULTY CURVE (fraction)\n{'=' * 74}")
        print(f"  {'point':<10}{'solo':>8}{'team':>8}{'gap':>9}")
        for label in ("medium", "hard", "xhard"):
            if label in results:
                s, t, gap, _, _ = results[label]["fraction"]
                print(f"  {label:<10}{s:>8.3f}{t:>8.3f}{gap:>+9.3f}")
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
