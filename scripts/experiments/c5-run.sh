#!/usr/bin/env bash
# Re-measure the last positive finding on fresh seeds.
#
# 4.22 showed the pilot's four-agent baseline (0.631, n=48) was a high draw:
# the three-agent arms reproduce to within 0.002 across 791 episodes while the
# baseline moved -0.054. Every claim resting on that corpus inherits the doubt,
# and the biggest is 4.16b -- `medium` was selected as the one operating point
# where the team beats a matched-budget single agent, C5 +0.126 at p < 0.001.
#
# That contrast is team-vs-solo_long, and the team side of it is the arm that
# moved. So it is re-measured here on seeds 1000-1149, where the baseline is
# already recorded at n=150.
#
# Both single-agent arms, not just C5:
#
#   solo       1 agent x  3 rounds   the preregistered gate baseline, and per
#                                    4.18 the BEST single-agent configuration
#   solo_long  1 agent x 12 rounds   C5, the matched-budget contrast
#
# Running only C5 would re-measure the comparison that flatters the team while
# leaving the one that does not, which is the asymmetry 4.18 exists to name.
#
# solo is cheap (3 turns/episode) and runs first so the gate lands early.
set -uo pipefail

LOG=runs/c5-run.log
CONFIG=configs/llamacpp/medium-h3b.yaml
EPISODES=${EPISODES:-150}

mkdir -p runs
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "C5 + gate re-measurement on fresh seeds 1000-$((1000 + EPISODES - 1))"

if ! python -u scripts/ops/preflight.py --config "$CONFIG" 2>&1 | tee -a "$LOG"; then
  say "PREFLIGHT FAILED"
  exit 1
fi

say "=== solo + solo_long, $EPISODES episodes each ==="
if ! python -u -m collabengine.cli pipeline --config "$CONFIG" \
     --phases solo,solo_long --episodes "$EPISODES" 2>&1 | tee -a "$LOG"; then
  say "FAILED"
  exit 1
fi

say "ALL DONE -- python scripts/analysis/gate_report.py --run-dir runs/llama31-8b-q4-medium-h3b"
