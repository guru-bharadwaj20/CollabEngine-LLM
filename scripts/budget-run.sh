#!/usr/bin/env bash
# The C4 arm: one agent given the team's whole turn budget, all three tiers.
#
# RESEARCH-LOG 4.9 measured what the preregistered gate actually compares. The
# team arm takes 12 agent turns to solo's 3 and emits 4,443 output tokens to
# solo's 2,055 -- 2.16x the generation across 4x the forward passes. So "team
# beats solo" has never been separable from "more tokens beat fewer", and the
# design's three named confounds (LOG 1.3) do not cover it.
#
# `solo_budget` is 1 agent x (rounds x n_agents) at the same per-turn cap: 1x12
# against the team's 4x3. Team vs solo_budget removes the budget difference and
# leaves the multi-agent structure. It does not replace the gate -- team vs solo
# stays exactly as preregistered -- it decides which reading of it survives.
#
# Runs after scripts/served-run.sh. Same server, same configs, same seeds, so
# the new arm is paired with the two already on disk rather than merely matched.
#
#   scripts/budget-run.sh              # all three tiers
#   TIERS="medium" scripts/budget-run.sh
#
# Resumes like every other stage: episodes already on disk are skipped, so this
# is safe to re-run and a death costs only the unfinished episodes.
#
# pipefail for the reason served-run.sh documents: every command is piped
# through tee, and without it the preflight `if !` tests tee's status and the
# gate passes unconditionally.
set -uo pipefail

LOG=runs/budget-run.log
TIERS=${TIERS:-"medium hard xhard"}
mkdir -p runs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# This arm costs what the team arm costs -- it is the same total generation by
# construction, which is the entire point. Say so before spending it.
say "C4: solo_budget on [$TIERS]. Each tier is one team-arm's worth of card time."

i=0
for tier in $TIERS; do
  i=$((i + 1))
  CONFIG="configs/llamacpp/$tier.yaml"

  if [ ! -f "$CONFIG" ]; then
    say "no $CONFIG -- skipping $tier"
    continue
  fi

  # The same global preconditions served-run.sh checks, for the same reason: an
  # undersized slot or the wrong weights fails every tier identically, and
  # finding out on the third one wastes the first two.
  say "=== $i  preflight $tier ==="
  if ! python -u scripts/preflight.py --config "$CONFIG" 2>&1 | tee -a "$LOG"; then
    say "PREFLIGHT FAILED for $tier -- stopping."
    exit 1
  fi

  say "=== $i  $tier -> solo_budget, 24 episodes ==="
  python -u -m collabengine.cli pipeline --config "$CONFIG" \
    --phases solo_budget --episodes 24 2>&1 | tee -a "$LOG"
done

say "C4 DONE -- python scripts/gate_report.py reads the new arm from the same corpus"
