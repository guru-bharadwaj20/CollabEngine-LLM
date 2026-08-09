#!/usr/bin/env bash
# The night's queue, chained so no card time is lost between stages.
#
# Runs only what a hypothesis depends on, in the order that the preregistration
# (docs/PREREG-xhard.md) needs it:
#
#   1. xhard to 24 episodes per arm   -- H1/H2, the newest and least certain point
#   2. hard  to 24 episodes per arm   -- H1's trend has three points; this is the
#                                        middle one and it was the thinnest
#   3. behavioural coding on xhard    -- differentiation at the largest instance
#
# Not run: the symmetry sweep and fixed-order control at xhard. No hypothesis
# mentions them, and card time spent on arms nobody predicted anything about is
# how a study ends up with results it has to explain rather than test.
#
# Each stage resumes. Episodes already on disk are skipped, so a stage that dies
# costs only its unfinished episodes, and re-running this script is safe.
set -u

LOG=runs/overnight.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

wait_for_pipeline() {
  # Absent five times running before believing it, as in followup.sh: a restart
  # leaves a gap of seconds, and firing into it starts a second 15 GiB model
  # against the first. Both then page instead of failing.
  local gone=0
  while true; do
    alive=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*collabengine.cli*' } | Measure-Object).Count" | tr -d '\r ')
    if [ "${alive:-0}" = "0" ]; then
      gone=$((gone + 1))
      [ "$gone" -ge 5 ] && return 0
    else
      gone=0
    fi
    sleep 30
  done
}

say "waiting for the in-flight xhard run to finish"
wait_for_pipeline
say "card is idle; starting the queue"

say "=== 1/3  xhard -> 24 episodes per arm ==="
python -u -m collabengine.cli pipeline --config configs/local-gpu-xhard.yaml \
  --phases baseline,solo --episodes 24 2>&1 | tee -a "$LOG"

say "=== 2/3  hard -> 24 episodes per arm ==="
python -u -m collabengine.cli pipeline --config configs/local-gpu.yaml \
  --phases baseline,solo --episodes 24 2>&1 | tee -a "$LOG"

say "=== 3/3  behavioural coding on the xhard corpus ==="
python -u -m collabengine.cli code --config configs/local-gpu-xhard.yaml \
  --judge self --judge-name local8b --episode-concurrency 8 2>&1 | tee -a "$LOG"

say "ALL DONE"
