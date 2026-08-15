#!/usr/bin/env bash
# The whole difficulty curve, on the served 4-bit instrument.
#
# Replaces scripts/overnight.sh, whose header explains at length why `xhard`
# could not be in it. It can be here: llama.cpp prefills in micro-batches and
# materialises logits only for the sampled position, so the per-prompt-token
# allocation that made the tier infeasible at bf16 does not exist
# (RESEARCH-LOG 3.11, PREREG-xhard Amendment 2).
#
# All three tiers, because H1 is a trend across them and a trend with one point
# measured on other weights measures the weights. 6 arms, n=24 each.
#
# Start the server first -- docs/LLAMACPP-SETUP.md has the arithmetic:
#
#   llama-server -m models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf \
#       --host 127.0.0.1 --port 8000 -ngl 99 -c 73728 --parallel 4 \
#       -b 2048 -ub 512 --flash-attn
#
# Order is `medium` first, and it is not alphabetical. Medium is the cheapest
# tier and its solo arm is H4's anchor: if solo does not degrade across the
# three tiers on this model, the instance size is not biting, H1 is untestable
# on this run rather than false, and the rest of the card time should go
# somewhere else.
#
# Every tier resumes: episodes already on disk are skipped, so re-running this
# is safe and a stage that dies costs only its unfinished episodes.
# pipefail, unlike overnight.sh: every command here is piped through tee, and
# without it the `if !` below tests tee's exit status, which is always 0. The
# preflight gate would then pass unconditionally -- a check that always passes
# is worse than no check, because it is also a reason not to look.
set -uo pipefail

LOG=runs/served-run.log
mkdir -p runs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

i=0
for tier in medium hard xhard; do
  i=$((i + 1))
  CONFIG="configs/llamacpp/$tier.yaml"

  say "=== $i/3  preflight $tier ==="
  # Not a formality, and not skippable. Three corpora here were lost to
  # conditions true before the first episode ran. A failure aborts the whole
  # queue rather than the tier, because every failure this checks for -- an
  # undersized slot, the wrong weights, a dead server -- is global to the
  # server and would fail the next tier identically, only eight hours later.
  if ! python -u scripts/preflight.py --config "$CONFIG" 2>&1 | tee -a "$LOG"; then
    say "PREFLIGHT FAILED for $tier -- stopping. Nothing after this would run clean."
    exit 1
  fi

  say "=== $i/3  $tier -> 24 episodes per arm ==="
  python -u -m collabengine.cli pipeline --config "$CONFIG" \
    --phases baseline,solo --episodes 24 2>&1 | tee -a "$LOG"
done

say "ALL DONE -- now: scripts/gatecheck.sh, then scripts/followup.sh per tier"
