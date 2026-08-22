#!/usr/bin/env bash
# The Tier-1 arms, reordered by what each one buys per card-hour.
#
# This replaces stages 1-6 of `queue-all.sh`. It exists as a separate file
# rather than an edit because queue-all.sh was mid-sweep when the order changed,
# and bash reads a script incrementally: editing a running one changes what it
# executes next. Launch this only after the sweep has finished and queue-all is
# stopped.
#
#   stage                 ETA     value        why it sits here
#   ------------------------------------------------------------------------
#   1  code family        7 h     highest      one task family is the study's
#                                              biggest weakness; if the artifact
#                                              appears in an unrelated domain
#                                              with an unrelated grader it stops
#                                              being a bug in one harness
#   2  qwen 7B gate       2 h     moderate     cheapest arm here, and it answers
#                                              a question reviewers do ask
#   3  mistral gate       3 h     moderate     the same question, second family;
#                                              largely redundant with stage 2
#   4  llama Q8_0 gate    4 h     low          no reviewer has asked this
#   5  llama f16 gate     7.5 h   low          as above, and three times longer
#   6  14B grid          15 h     conditional  longest item, and confounded
#
# **Why the most valuable item is first rather than the shortest.** Ordering
# purely by ETA would put the two 7B gates ahead of the code family and finish
# the cheap questions first. That optimises for stages completed, which is not
# the quantity that matters: if the card is lost on Thursday, the paper is in a
# materially better position having the second task family and nothing else than
# having four gates and no second task family. Everything after stage 1 is
# ascending ETA, so the cheap arms still land early and the two long ones cannot
# block anything.
#
# **Stage 6 is last on merit, not only on length.** The 14B changes scale and
# model family in the same step -- Qwen2.5-14B against Llama-3.1-8B -- so it
# cannot cleanly answer "is 8B too weak to collaborate", only "does this other
# model at this other size behave differently". docs/PREREG-14b.md registers
# that confound rather than hiding it. It pays off in one branch: a team
# advantage that appears here and not at 8B turns the paper's negative result
# into a capability-threshold finding. That branch is worth fifteen hours only
# once everything cheaper has been bought.
#
# **Every stage is independent.** No stage consumes another's output, each one
# gates on the card, serves its own geometry, preflights, and resumes from disk.
# A failure is recorded and the next stage starts anyway, because the arms
# answer unrelated questions and a card-night is worth more than a tidy exit
# code. Re-running this script resumes whatever did not finish; a run directory
# that is already complete is skipped rather than regenerated.
#
# **Nothing is analysed while it runs.** Stages write episodes and stop. Read
# each against its registered threshold afterwards.
#
#   nohup bash scripts/ops/queue-tier1.sh >/dev/null 2>&1 &
#
# Launch detached: a session exit kills its children.
set -uo pipefail

LOG=runs/queue-tier1.log
LOCK=runs/queue-tier1.lock
STOP_SERVER=${STOP_SERVER:-1}
STABLE=${STABLE:-3}            # consecutive clear checks (x60s) before believing it
# PY comes from _python.sh, which refuses the Microsoft Store stub. See its header.
. scripts/ops/_python.sh

mkdir -p runs
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another queue holds $LOCK; not starting a second" >&2
  exit 3
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM

# ---------------------------------------------------------------------------
# Helpers -- verbatim from queue-all.sh, which took them from queue-judge.sh.
# ---------------------------------------------------------------------------
pipeline_alive() {
  powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*collabengine.cli pipeline*' -or \$_.CommandLine -like '*collabengine.cli ablate*' } | Measure-Object).Count" \
    | tr -d '\r '
}

# Do not rewrite this and do not drop the `tr -d ' \r'`. nvidia-smi on Windows
# ends its lines with CRLF; a surviving CR makes the arithmetic a syntax error,
# which leaves $free unset and lets the gate fall open. That is how a run was
# once launched into a card holding 17.3 GiB of another account's job.
wait_for_card() {
  local need=$1 clear_count=0 total used free holder
  say "waiting for ${need} MiB free, stable for ${STABLE} checks"
  while true; do
    read -r total used < <(nvidia-smi --query-gpu=memory.total,memory.used \
      --format=csv,noheader,nounits | tr -d ' \r' | tr ',' ' ')
    case "${total:-}${used:-}" in
      *[!0-9]*|"") say "unparseable nvidia-smi output; treating the card as busy"
                   clear_count=0; sleep 60; continue ;;
    esac
    free=$(( total - used ))
    if [ "$free" -ge "$need" ]; then
      clear_count=$((clear_count + 1))
      say "free ${free} MiB (${clear_count}/${STABLE})"
      [ "$clear_count" -ge "$STABLE" ] && break
    else
      if [ "$clear_count" -ne 0 ]; then say "card reclaimed; restarting the count"; fi
      clear_count=0
      holder=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' \r' | paste -sd, -)
      say "free ${free} MiB, need ${need}; compute pids: ${holder:-none}"
    fi
    sleep 60
  done
}

# The pid here is a WINDOWS pid; the MSYS job pid serve.sh prints is a different
# number and means nothing to Stop-Process.
stop_our_server() {
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 || return 0
  if [ "$STOP_SERVER" != "1" ]; then
    say "a server is up on :8000 at the wrong geometry and STOP_SERVER=0."
    return 0
  fi
  local alive
  alive=$(pipeline_alive)
  if [ "${alive:-1}" != "0" ]; then
    say "refusing to stop the server: a collabengine pipeline is still running"
    return 1
  fi
  say "stopping our idle llama-server on :8000 (windows pid, via Win32_Process)"
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" 2>&1 | tee -a "$LOG"
  for _ in $(seq 1 30); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 || break
    sleep 5
  done
}

serve_preset() {
  local preset=$1
  say "starting llama-server at preset '${preset}'"
  scripts/ops/serve.sh --model "$preset" --detach 2>&1 | tee -a "$LOG"
  for _ in $(seq 1 120); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && return 0
    sleep 5
  done
  say "server did not come up at preset '${preset}'"
  return 1
}

# $1 label  $2 config  $3 preset  $4 MiB needed  $5 run name  $6 grid|gate
run_arm() {
  local label=$1 config=$2 preset=$3 need=$4 name=$5 kind=$6
  local dir="runs/${name}"

  if [ "$kind" = "grid" ] && [ -s "${dir}/ablation.jsonl" ]; then
    say "=== ${label}: skipped, ${dir}/ablation.jsonl already exists ==="
    return 0
  fi
  if [ "$kind" = "gate" ] && [ -s "${dir}/baseline.jsonl" ] && \
     [ "$(wc -l < "${dir}/baseline.jsonl")" -ge "${COMPLETE_AT:-450}" ]; then
    say "=== ${label}: skipped, ${dir}/baseline.jsonl is complete ==="
    return 0
  fi

  say "=== ${label} (${config}, preset ${preset}, ${kind}) ==="
  stop_our_server
  wait_for_card "$need"
  serve_preset "$preset" || return 1

  # verify_model in preflight is the guard that matters when six models share
  # one machine and one port.
  if ! $PY -u scripts/ops/preflight.py --config "$config" 2>&1 | tee -a "$LOG"; then
    say "${label}: PREFLIGHT FAILED -- not spending the card on it"
    return 1
  fi

  if ! $PY -u -m collabengine.cli pipeline --config "$config" \
       --phases baseline,solo,solo_long 2>&1 | tee -a "$LOG"; then
    say "${label}: the headline arms failed; completed episodes are on disk"
    return 1
  fi
  say "${label}: GATE IS READABLE"

  if [ "$kind" = "grid" ]; then
    if ! $PY -u -m collabengine.cli ablate --config "$config" \
         --modes live --agents A1,A2,A3,A4 2>&1 | tee -a "$LOG"; then
      say "${label}: ablation failed; the gate above is still good"
      return 1
    fi
  fi

  say "=== ${label}: COMPLETE ==="
  return 0
}

# ---------------------------------------------------------------------------
# The arms, in value-then-ETA order.
# ---------------------------------------------------------------------------
# MiB figures come from serve.sh's KV table plus headroom, never from what fits
# exactly: Windows pages instead of raising OOM, so an allocation that only just
# fits runs at PCIe speed while every dashboard reads 100% (RESEARCH-LOG 3.4).
FAILED=()

run_arm "1/6 code family (1.1)"     configs/llamacpp/code-medium.yaml \
        q4       15000 llama31-8b-q4-code-medium    grid || FAILED+=("code family")

run_arm "2/6 qwen 7B gate (1.2)"    configs/llamacpp/medium-qwen.yaml \
        qwen      9500 qwen25-7b-q4-medium          gate || FAILED+=("qwen 7B")

run_arm "3/6 mistral 7B gate (1.2)" configs/llamacpp/medium-mistral.yaml \
        mistral  14700 mistral-7b-q4-medium         gate || FAILED+=("mistral 7B")

run_arm "4/6 llama Q8_0 gate (1.4)" configs/llamacpp/medium-q8.yaml \
        q8       18500 llama31-8b-q8-medium         gate || FAILED+=("llama Q8_0")

run_arm "5/6 llama f16 gate (1.4)"  configs/llamacpp/medium-f16.yaml \
        f16      21000 llama31-8b-f16-medium        gate || FAILED+=("llama f16")

run_arm "6/6 14B grid (1.3)"        configs/llamacpp/medium-14b.yaml \
        qwen-14b 20500 qwen25-14b-q4-medium         grid || FAILED+=("14B grid")

# ---------------------------------------------------------------------------
say ""
if [ ${#FAILED[@]} -eq 0 ]; then
  say "ALL DONE -- every Tier-1 arm is measured."
else
  say "DONE with ${#FAILED[@]} stage(s) unfinished: ${FAILED[*]}"
  say "Completed episodes are on disk. Re-run to resume; nothing regenerates."
fi
say ""
say "Read each against its registered threshold before writing anything:"
say "  14B          docs/PREREG-14b.md section 3"
say "  the rest     docs/PREREG-equivalence.md, at the registered margin"
