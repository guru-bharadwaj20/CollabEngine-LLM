#!/usr/bin/env bash
# The re-measurement on an instrument where the answer turn has its own budget,
# plus the honest single-agent baseline. Tasks 2 and 1 of RESEARCH-LOG 7.
#
# Why both in one run: they share every other setting, and running them
# separately would mean the C5 baseline is measured on a different instrument
# from the gate it is meant to inform -- the mistake 4.1c cost a write-up for.
#
# Four arms per tier, 24 episodes each:
#
#   baseline      4 agents x 3 rounds        the team
#   solo          1 agent  x 3 rounds        the preregistered gate baseline
#   solo_budget   1 agent  x 12 rounds       C4, TEAM_BRIEF -- the defective arm
#   solo_long     1 agent  x 12 rounds       C5, SOLO_BRIEF -- the honest one
#
# solo_budget is re-run here rather than reused from runs/llama31-8b-q4-*
# because those episodes were generated under the shared cap. Comparing C5
# against them would confound the brief with the answer budget, and the whole
# point of C5 is to isolate the brief.
#
# Order is medium, hard, xhard: cheapest first, and every tier resumes, so a
# death costs only its unfinished episodes and the earlier tiers still report.
#
# Start the server first (scripts/serve.sh). The slot is unchanged at 18,432 --
# this instrument moves budget from the prompt to the answer rather than asking
# the card for more.
set -uo pipefail

LOG=runs/answer-run.log
TIERS=${TIERS:-"medium hard xhard"}
PHASES=${PHASES:-"baseline,solo,solo_budget,solo_long"}
mkdir -p runs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "answer-budget re-measurement + C5 baseline on [$TIERS], phases: $PHASES"

i=0
for tier in $TIERS; do
  i=$((i + 1))
  CONFIG="configs/llamacpp-$tier-ans.yaml"

  if [ ! -f "$CONFIG" ]; then
    say "no $CONFIG -- skipping $tier"
    continue
  fi

  # Preflight matters more here than anywhere: this instrument spends the slot
  # differently (max_model_len 15360 + answer 3072 = 18432), so an arithmetic
  # slip shows up as rejected answer turns in exactly the arm being fixed.
  say "=== $i  preflight $tier ==="
  if ! python -u scripts/preflight.py --config "$CONFIG" 2>&1 | tee -a "$LOG"; then
    say "PREFLIGHT FAILED for $tier -- stopping. Nothing after this runs clean."
    exit 1
  fi

  say "=== $i  $tier -> $PHASES, 24 episodes per arm ==="
  python -u -m collabengine.cli pipeline --config "$CONFIG" \
    --phases "$PHASES" --episodes 24 2>&1 | tee -a "$LOG"

  say "=== $i  $tier done -- gate below ==="
  python -u scripts/gate_report.py --run-dir "runs/llama31-8b-q4-$tier-ans" \
    2>&1 | tee -a "$LOG"
done

say "ALL DONE -- python scripts/gate_report.py, then scripts/figures.py"
