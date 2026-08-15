#!/usr/bin/env bash
# The night's queue.
#
# `xhard` is NOT here. It was the plan, and it is not runnable on this card:
# a single 5223-token prefill OOMs with 15.3 GiB of bf16 weights resident,
# measured directly (RESEARCH-LOG 3.10). xhard prompts *start* at ~5200 tokens,
# so the tier is infeasible at this precision on 24 GB, and no amount of batch
# tuning changes that -- the failing allocation is per prompt token and
# independent of batch size.
#
# What is left is worth more than a third operating point anyway: both existing
# points are at n=12 per arm, which is why every interval in the project spans
# zero. Doubling them is the difference between "we could not detect a benefit"
# and "we can bound the benefit". A tighter null is a stronger result.
#
#   1. hard   -> 24 episodes per arm
#   2. medium -> 24 episodes per arm (team; solo already n=12, extended here)
#
# Both resume: episodes on disk are skipped, so re-running this is safe and a
# stage that dies costs only its unfinished episodes.
set -u

LOG=runs/overnight.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== 1/2  hard -> 24 episodes per arm ==="
python -u -m collabengine.cli pipeline --config configs/hf-local/hard.yaml \
  --phases baseline,solo --episodes 24 2>&1 | tee -a "$LOG"

say "=== 2/2  medium -> 24 episodes per arm ==="
python -u -m collabengine.cli pipeline --config configs/hf-local/medium.yaml \
  --phases baseline,solo --episodes 24 2>&1 | tee -a "$LOG"

say "ALL DONE"
