"""Regenerate instrument-failure episodes at low concurrency.

Dropping a failed episode is not neutral. Failures land on the longest
contexts, the longest contexts are team episodes, and at `medium` the episodes
lost to OOM scored 0.775 against surviving episodes' 0.907 with none of them
feasible (RESEARCH-LOG 4.1d). Dropping them therefore *inflates* the team mean
and manufactures a benefit. The n=24 `hard` arm is the worked example: unusable
team episodes went from 1 in 12 to 5 in 24 and the gap appeared to grow.

So they are regenerated rather than dropped, and regenerated at
`max_concurrency: 2` where a long prefill has the card to itself. The
exhaustions are lost races against sibling batches holding memory, not a fixed
ceiling -- a 10,221-token turn needs ~7 GiB of transient logits memory on top
of 15.3 GiB of weights, which fits alone and does not fit in a crowd.

    python scripts/repair.py --config configs/local-gpu-medium-serial.yaml

Always backs up the transcript before removing anything. Idempotent: an episode
that fails again is left failed rather than retried forever, and the script
reports what it could not fix.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from collabengine.analysis.integrity import is_instrument_failure
from collabengine.config import ExperimentConfig
from collabengine.transcripts.store import TranscriptReader


#: `collabengine baseline` plans episodes for this condition and no other, so
#: this is the only condition whose episodes can be deleted and recreated. The
#: first version of this script dropped *every* instrument failure and then ran
#: that command, which would have permanently deleted the `fixed_order` and
#: `symmetry` failures -- data no longer reachable by any command in the CLI.
#: Caught by a dry run against the hard corpus, which listed
#: `fixed_order:hard:3` and `fixed_order:hard:6` among the episodes it proposed
#: to remove.
REGENERABLE = "baseline"


def failures(path: Path, condition: str = REGENERABLE) -> tuple[list[str], list[str]]:
    """(repairable, unrepairable) instrument-failure episode ids."""
    repairable: list[str] = []
    other: list[str] = []
    for rec in TranscriptReader(str(path)):
        if not is_instrument_failure(rec):
            continue
        (repairable if rec.condition == condition else other).append(rec.episode_id)
    return repairable, other


def drop(path: Path, ids: set[str]) -> int:
    kept = [
        line
        for line in path.open(encoding="utf-8")
        if json.loads(line)["episode_id"] not in ids
    ]
    path.write_text("".join(kept), encoding="utf-8")
    return len(kept)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    path = cfg.run_dir / "baseline.jsonl"
    if not path.exists():
        print(f"no transcript at {path}", file=sys.stderr)
        return 2

    for attempt in range(1, args.passes + 1):
        bad, untouched = failures(path)
        if untouched:
            print(f"leaving {len(untouched)} failure(s) in other conditions alone "
                  f"({', '.join(untouched)}): `baseline` cannot regenerate them, "
                  "so removing them would delete data permanently")
        if not bad:
            print(f"pass {attempt}: no repairable instrument failures")
            return 0

        print(f"pass {attempt}: {len(bad)} repairable -> {', '.join(bad)}")
        if args.dry_run:
            return 0

        # Back up before every destructive step, not once at the start. A second
        # pass would otherwise overwrite the only copy of the first pass's data.
        stamp = datetime.now().strftime("%H%M%S")
        backup = path.with_suffix(f".before-repair-{stamp}.jsonl")
        shutil.copy2(path, backup)
        remaining = drop(path, set(bad))
        print(f"  backed up to {backup.name}; {remaining} episodes kept")

        # Resume regenerates exactly the missing episode ids.
        rc = subprocess.call(
            [sys.executable, "-u", "-m", "collabengine.cli", "baseline",
             "--config", args.config]
        )
        print(f"  regeneration exited {rc}")

    left, _ = failures(path)
    if left:
        print(f"\nstill failing after {args.passes} passes: {', '.join(left)}")
        print("These are genuinely too large for this card. Report the corpus")
        print("with its censoring rate stated beside every mean -- do not")
        print("quietly analyse the survivors.")
        return 1
    print("\nall episodes repaired")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
