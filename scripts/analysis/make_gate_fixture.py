"""Cut a committable slice of the corpus, and the golden numbers it produces.

`tests/test_gate_golden.py` needs a corpus small enough to live in git and real
enough to exercise the whole analysis path. This produces both halves: the
slice, and the numbers it currently yields.

**Deterministic, and by seed rather than by sampling.** The slice takes the
lowest-numbered instance seeds present in each arm, so re-running this on a
larger corpus keeps the same episodes rather than silently rotating them. A
fixture that changes every time it is regenerated makes its own golden file
meaningless.

    python scripts/analysis/make_gate_fixture.py
    python scripts/analysis/make_gate_fixture.py --n 30 --run-dir runs/other

Commit both outputs. A change in the golden file is a change in the analysis and
belongs in the commit that causes it, with a reason.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

OUT_DIR = Path("tests/fixtures")
SLICE = OUT_DIR / "gate_slice.jsonl"
GOLDEN = OUT_DIR / "gate_slice.golden.json"

#: Arms the gate is read from. `solo_long` is deliberately excluded: it makes
#: the fixture half as big again and the golden numbers it would add are not
#: ones any test asserts.
ARMS = ("baseline", "solo")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/llama31-8b-q4-medium-h3b")
    ap.add_argument("--n", type=int, default=20, help="episodes per arm")
    args = ap.parse_args()

    from collabengine.analysis.inference import smallest_equivalence_bound
    from collabengine.analysis.integrity import is_instrument_failure
    from collabengine.analysis.scoring import METRICS, rescore
    from collabengine.transcripts.store import TranscriptReader

    run_dir = Path(args.run_dir)
    src = run_dir / "baseline.jsonl"
    if not src.exists():
        print(f"no corpus at {src}")
        return 2

    by_arm: dict[str, list] = {a: [] for a in ARMS}
    for rec in TranscriptReader(str(src)):
        if rec.condition in by_arm and not is_instrument_failure(rec):
            by_arm[rec.condition].append(rec)

    kept = []
    for arm in ARMS:
        chosen = sorted(by_arm[arm], key=lambda r: r.instance_seed)[: args.n]
        if len(chosen) < args.n:
            print(f"only {len(chosen)} usable episodes in {arm}; "
                  f"the corpus is still generating?")
            return 2
        kept.extend(chosen)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SLICE.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in kept:
            fh.write(json.dumps(rec.to_dict(), sort_keys=True) + "\n")

    arms: dict[str, dict[str, list[float]]] = {}
    for rec in kept:
        scored = rescore(rec)
        bucket = arms.setdefault(rec.condition, {m: [] for m in METRICS})
        for metric in METRICS:
            bucket[metric].append(scored.overall[metric])

    golden = {
        "source": str(run_dir),
        "task_family": (kept[0].config or {}).get("task", "scheduling"),
        "arms": {
            name: {
                "n": len(vals["fraction"]),
                "mean": {m: st.mean(vals[m]) for m in METRICS},
            }
            for name, vals in arms.items()
        },
        "gap": {
            m: st.mean(arms["baseline"][m]) - st.mean(arms["solo"][m])
            for m in METRICS
        },
        "equivalence_bound_fraction": smallest_equivalence_bound(
            arms["solo"]["fraction"], arms["baseline"]["fraction"]
        ),
    }
    GOLDEN.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")

    print(f"wrote {SLICE} ({SLICE.stat().st_size / 1024:.0f} KiB, "
          f"{len(kept)} episodes)")
    print(f"wrote {GOLDEN}")
    for name, info in golden["arms"].items():
        print(f"  {name:<10} n={info['n']:<4} "
              f"fraction={info['mean']['fraction']:.4f}")
    print(f"  gap        {golden['gap']['fraction']:+.4f}")
    print(f"  bound      {golden['equivalence_bound_fraction']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
