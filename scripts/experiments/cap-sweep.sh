#!/usr/bin/env bash
# Dose-response: the token-cap artifact as a curve, not as an on/off switch.
#
# Final Sweep 7.4, and the item that section says to keep if the schedule
# collapses. What the paper currently has is a before/after: at
# `answer_max_tokens` = 1024 the team appeared to beat one agent by d = 1.09 at
# `hard` (p < 0.0001, RESEARCH-LOG 4.9-4.11), and at 3072 the same contrast is
# +0.003. Two points. Two points support "we found a bug"; they do not support
# "we characterise when the bug appears and how large it gets", and the second
# claim is the one that makes 7.1's audit predictive -- given a published
# system's cap and the verbosity ratio between its arms, the curve estimates how
# much of its reported advantage is artifact.
#
# So: five rungs, 512 / 1024 / 2048 / 3072 / 6144, one tier (`medium`), both
# arms (`baseline` = 4 agents x 3 rounds, `solo` = 1 agent x 3 rounds), 100
# episodes per arm per rung. 1,000 episodes, ~3-4 card-nights.
#
# **Why the mechanism is a ratio and not a cap.** The cap does not truncate
# arms equally; it truncates the arm that writes more. One agent restates the
# whole working solution in its final turn, four agents commit an answer the
# transcript already holds, and the measured generation ratio between them is
# 1.87x at `medium` (RESEARCH-LOG 4.18). A cap above both arms' answers costs
# nothing; a cap between them costs one arm its score. That predicts a
# threshold, not a slope, and the threshold is where the curve should sit --
# which is exactly what five rungs can falsify and two cannot.
# docs/PREREG-cap-sweep.md registers the predicted shape before the run.
#
# **The prompt window is held fixed and the slot grows with the cap.** The -ans
# instrument spends a fixed 18,432-token slot as 15360 + 3072. Holding the slot
# instead would mean the prompt window shrank as the cap grew, so a rung's
# result would confound the answer budget with the context available to produce
# it -- two knobs, one curve, nothing separable. Every config here therefore
# pins max_model_len at 15360 and the server is started at the LARGEST rung's
# slot, once, for the whole sweep:
#
#     3 slots x (15360 + 6144) = 3 x 21504 = 64512
#
# Three slots, not four: ~12.5 GiB against the default preset's ~13.6, because
# this runs on a shared card and a job that asks for less than its predecessor
# cannot be the one that pushes the card into paging (RESEARCH-LOG 3.4).
#
# **This script does not kill anything.** If a server is already up on :8000 it
# is inspected, not replaced: a wrong-geometry server is a reason to stop and
# say so, never a reason to restart something that may belong to another run.
# Use scripts/ops/queue-tier2.sh to launch this behind a job that is still
# going; that is where the wait-for-the-card gate lives.
#
# Resumes on plan id, so a death costs only the unfinished episodes of the rung
# it died in, and every completed rung stays readable.
#
#   CTX=64512 PARALLEL=3 scripts/ops/serve.sh --detach
#   bash scripts/experiments/cap-sweep.sh
#   python scripts/analysis/dose_response.py
set -uo pipefail

LOG=runs/cap-sweep.log
CAPS=${CAPS:-"512 1024 2048 3072 6144"}
PHASES=${PHASES:-"baseline,solo"}
# Per arm per rung. Raising it resumes: the pipeline compares plan ids against
# the ids already in the transcript, so EPISODES=150 adds seeds 2100..2149 and
# re-runs nothing. 100 is what PREREG-cap-sweep registered and what
# power_report.py sized; changing it after the fact is an amendment, not a knob.
EPISODES=${EPISODES:-100}
# PY comes from _python.sh, which refuses the Microsoft Store stub. See its header.
. scripts/ops/_python.sh

# The geometry every rung must be served at. Not the default preset: the 6144
# rung needs 21,504 tokens in a slot and would otherwise have its answer turns
# rejected -- in the one arm the sweep exists to measure.
NEED_SLOT=$((15360 + 6144))

mkdir -p runs
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "dose-response cap sweep: caps [$CAPS], phases $PHASES, $EPISODES episodes/arm"

if ! curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
  say "no server on :8000. Start it at the sweep's geometry first:"
  say "    CTX=$((3 * NEED_SLOT)) PARALLEL=3 scripts/ops/serve.sh --detach"
  exit 1
fi

# Preflight is the check that matters here and it is per rung, not per sweep.
# Check 2 sizes the worst-case request against a single slot, which is the
# arithmetic that differs between rungs -- and the failure it catches is silent:
# a rejected answer turn scores as a bad episode, not as an error, in whichever
# arm writes the longest answers. That is the arm whose truncation this whole
# sweep is measuring, so an unpreflighted rung would manufacture the result.
say "=== 0  preflighting all rungs before spending anything ==="
for cap in $CAPS; do
  CONFIG="configs/llamacpp/cap-${cap}.yaml"
  if [ ! -f "$CONFIG" ]; then
    say "no $CONFIG -- the sweep is not complete; stopping rather than reporting a gap"
    exit 1
  fi
  if ! $PY -u scripts/ops/preflight.py --config "$CONFIG" 2>&1 | tee -a "$LOG"; then
    say "PREFLIGHT FAILED for cap $cap. If check 2 is what failed, the server is"
    say "not at the sweep geometry: stop it and restart with"
    say "    CTX=$((3 * NEED_SLOT)) PARALLEL=3 scripts/ops/serve.sh --detach"
    exit 1
  fi
done
say "all rungs preflight clean on one server; nothing restarts mid-sweep"

# Ascending, and the order is not arbitrary. The short caps are where the
# artifact is predicted to be largest, so a sweep that dies halfway has still
# measured the interesting half -- and if 512 shows no gap at all, the
# mechanism is wrong and the remaining three card-nights should not be spent.
i=0
n=$(echo "$CAPS" | wc -w)
for cap in $CAPS; do
  i=$((i + 1))
  CONFIG="configs/llamacpp/cap-${cap}.yaml"
  RUN_DIR="runs/llama31-8b-q4-cap$(printf '%04d' "$cap")"

  say "=== $i/$n  cap $cap -> $PHASES, $EPISODES episodes per arm ==="
  if ! $PY -u -m collabengine.cli pipeline --config "$CONFIG" \
       --phases "$PHASES" --episodes "$EPISODES" 2>&1 | tee -a "$LOG"; then
    say "FAILED at cap $cap; rungs already on disk stay readable. Re-run this"
    say "script to resume -- completed episodes are not regenerated."
    exit 1
  fi

  # Printed per rung rather than only at the end, because the truncation counts
  # in the integrity block are the mechanism itself. If they are zero in both
  # arms at a short cap, the curve about to be drawn has no cause behind it and
  # that is worth seeing on the night it happens, not three nights later.
  say "=== $i/$n  cap $cap done -- gate below ==="
  $PY -u scripts/analysis/gate_report.py --run-dir "$RUN_DIR" 2>&1 | tee -a "$LOG"
done

say "=== the curve ==="
$PY -u scripts/analysis/dose_response.py 2>&1 | tee -a "$LOG"

say "ALL DONE -- read it against docs/PREREG-cap-sweep.md section 4 before writing anything"
