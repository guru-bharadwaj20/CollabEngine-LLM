#!/usr/bin/env bash
# H2 on fresh seeds: does the participation effect replicate?
#
# The pilot put the per-agent live drop at +0.055 pooled, measured against a
# baseline of 0.631 drawn from seeds 0-47. On seeds 1000-1149 that same
# baseline is 0.577, and A4's drop falls from +0.071 to +0.016 (4.20b). Either
# the pilot drew a favourable 48-seed baseline or the effect is not there.
#
# A4 is already run (300 episodes, 08:39 today). This adds A1, A2 and A3 on the
# same fresh seeds so H2 can be pooled across all four WITHOUT blending the
# pilot and confirmatory baselines -- which is the whole point of the fresh-seed
# rule in PREREG-phase3.
#
# `live` only. No frozen_replay, no random_message, no further capacity cells:
# the rest of the grid stays paused until H2 has an honest answer, because
# every one of those arms is expressed as a drop from a baseline H2 is
# currently questioning.
#
# Resumes on plan id, so a death costs only the unfinished episodes.
set -uo pipefail

LOG=runs/h2-run.log
CONFIG=configs/llamacpp/medium-h3b.yaml
AGENTS=${AGENTS:-A1,A2,A3}

mkdir -p runs
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "H2 fresh-seed replication: live ablation for $AGENTS on seeds 1000-1149"

if ! python -u scripts/ops/preflight.py --config "$CONFIG" 2>&1 | tee -a "$LOG"; then
  say "PREFLIGHT FAILED -- nothing after this runs clean"
  exit 1
fi

say "=== 1/2 live:$AGENTS, 150 episodes each ==="
if ! python -u -m collabengine.cli ablate --config "$CONFIG" \
     --modes live --agents "$AGENTS" 2>&1 | tee -a "$LOG"; then
  say "FAILED live arms"
  exit 1
fi
say "H2 ARMS COMPLETE -- pooled replication is readable now"

# The baseline stopped at 149/150 when the server died this morning. One
# episode, and H2's reference should not be short a seed.
say "=== 2/2 top up the baseline to 150 ==="
python -u -m collabengine.cli pipeline --config "$CONFIG" \
  --phases baseline --episodes 150 2>&1 | tee -a "$LOG"

say "ALL DONE"
