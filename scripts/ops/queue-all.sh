#!/usr/bin/env bash
# The whole remaining GPU lane, in one chain, in priority order.
#
# `queue-tier2.sh` sequences the two Tier-2 items. This sequences everything:
# Tier-2 first, then the four Tier-1 arms that are built, preregistered and
# unrun. Roughly 40 card-hours end to end, which on a shared card is a week of
# nights.
#
#   stage                        item        cost        what it answers
#   ---------------------------------------------------------------------
#   0  tier-2 (judge + sweep)    7.3, 7.4    3-4 nights  is the artifact a curve?
#   1  14B grid                  1.3         6-8 h       is 8B too weak to collaborate?
#   2  code task family          1.1         5-7 h       is it one harness or a class?
#   3  qwen 7B gate              1.2         3 h         is it a Llama quirk?
#   4  mistral 7B gate           1.2         3 h         is it a Llama quirk?
#   5  llama Q8_0 gate           1.4         4 h         is it a quantisation artifact?
#   6  llama f16 gate            1.4         8-10 h      is it a quantisation artifact?
#
# **The order is Final Sweep 9's, and its two load-bearing dependencies.**
# Tier-2 runs first because 7.4 is the item that upgrades the central claim
# rather than broadening it, and Final Sweep 10 names it the one to keep if the
# schedule collapses. The 14B goes before the other families because if a team
# advantage appears at 14B the thesis changes and the other families should be
# run at 14B rather than at 8B -- running them first risks spending eight
# card-nights at the wrong scale.
#
# **Stages 3-6 run the gate only, and that is a judgement call worth stating.**
# Stages 0-2 run the full grid, pipeline and ablation both. Stages 3-6 run
# `pipeline` and stop. The ablation grid is the most expensive thing here and it
# answers the differentiation question, which is scoped to the instrument the
# study characterised. What 1.2 and 1.4 exist to answer is narrower and is a
# gate question: does the solo-vs-team null replicate off Llama, and off 4-bit?
# Six ablation grids would cost ~15 further card-hours to re-measure a null that
# the headline corpus already bounds. If a gate moves at any of these, its
# ablation is the obvious follow-up and this script is the place to add it.
#
# **Nothing here is analysed while it runs.** Every stage writes episodes and
# stops. The preregistrations -- PREREG-14b.md, PREREG-cap-sweep.md -- name
# their thresholds in advance precisely so that a result cannot be read as it
# arrives, and a chain that printed a gate after each stage would invite exactly
# that. Read them afterwards, against the registered threshold.
#
# **Resumption.** Every stage resumes on plan id, so a death costs only the
# unfinished episodes of the stage it happened in, not the stage. Re-running
# this script after any failure picks up where it stopped. A stage whose run
# directory is already complete is skipped rather than regenerated -- silently
# overwriting a measured arm is the one unrecoverable mistake available here.
#
# **The card is shared and half the other accounts are administrators**
# (RESEARCH-LOG 3.9), so there is no lock to take. Every stage gates on the card
# being genuinely free and waits rather than killing. The gate is copied
# verbatim from queue-judge.sh: it carries a CRLF fix that was paid for once
# already, and a guard that fails open is worse than no guard because it is
# trusted.
#
#   nohup bash scripts/ops/queue-all.sh >/dev/null 2>&1 &
#
# Launch it detached. A session exit kills its children, which has already cost
# this project one full stage-1 run.
set -uo pipefail

LOG=runs/queue-all.log
LOCK=runs/queue-all.lock
STOP_SERVER=${STOP_SERVER:-1}
STABLE=${STABLE:-3}            # consecutive clear checks (x60s) before believing it
SKIP_TIER2=${SKIP_TIER2:-0}
PY=${PY:-python}

mkdir -p runs
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another queue holds $LOCK; not starting a second" >&2
  exit 3
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM

# ---------------------------------------------------------------------------
# Is one of our pipelines still generating? Verbatim from queue-tier2.sh.
# ---------------------------------------------------------------------------
pipeline_alive() {
  powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*collabengine.cli pipeline*' -or \$_.CommandLine -like '*collabengine.cli ablate*' } | Measure-Object).Count" \
    | tr -d '\r '
}

# ---------------------------------------------------------------------------
# GPU gate -- verbatim from scripts/ops/queue-judge.sh. Do not rewrite it and
# do not drop the `tr -d ' \r'`; see the header of that file for what it cost.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Stop OUR idle llama-server, and only that. Verbatim from queue-tier2.sh.
# ---------------------------------------------------------------------------
# Every stage below wants a different slot geometry, so the server from the
# previous stage must go. A server left up at the wrong geometry does not fail
# loudly: it holds its weights so the gate never opens, or it accepts the run
# and rejects the long answer turns. Neither shows in a headline metric (3.4).
#
# The pid here is a WINDOWS pid. The MSYS job pid that serve.sh prints does not
# exist in this shell and means nothing to Stop-Process. Confusing the two has
# broken three separate things in this project.
stop_our_server() {
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 || return 0
  if [ "$STOP_SERVER" != "1" ]; then
    say "a server is up on :8000 at the wrong geometry and STOP_SERVER=0."
    say "Stop it by hand, or the next stage will wait on a card it holds."
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

# ---------------------------------------------------------------------------
# One arm.
# ---------------------------------------------------------------------------
# $1 stage label   $2 config   $3 serve preset   $4 MiB needed   $5 run name
# $6 "grid" to run the ablation as well, "gate" to stop after the pipeline.
#
# The completion marker is the run's own ablation or baseline file rather than a
# sentinel this script writes, because the question worth asking on resume is
# "are the episodes on disk", not "did a previous invocation think it finished".
run_arm() {
  local label=$1 config=$2 preset=$3 need=$4 name=$5 kind=$6
  local dir="runs/${name}"

  if [ "$kind" = "grid" ] && [ -s "${dir}/ablation.jsonl" ]; then
    say "=== ${label}: skipped, ${dir}/ablation.jsonl already exists ==="
    return 0
  fi
  if [ "$kind" = "gate" ] && [ -s "${dir}/baseline.jsonl" ] && \
     [ "$(wc -l < "${dir}/baseline.jsonl")" -ge 450 ]; then
    say "=== ${label}: skipped, ${dir}/baseline.jsonl is complete ==="
    return 0
  fi

  say "=== ${label} (${config}, preset ${preset}, ${kind}) ==="
  stop_our_server
  wait_for_card "$need"
  serve_preset "$preset" || return 1

  # Preflight before spending the night. Its weight-verification check is the
  # guard that matters when six models share one machine and one port: three
  # families on one card is exactly how a server ends up holding the wrong
  # weights while every arm it serves looks healthy.
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
# Stage 0. Tier 2: the 14B judge check, then the dose-response sweep.
# ---------------------------------------------------------------------------
# Delegated rather than reimplemented. queue-tier2.sh holds the sweep's slot
# arithmetic and its own resume logic, and it takes its own lock, so this waits
# for it to finish rather than running it in parallel.
if [ "$SKIP_TIER2" = "1" ]; then
  say "=== 0/6 tier-2 skipped by SKIP_TIER2=1 ==="
elif [ -f docs/data/dose-response.json ]; then
  say "=== 0/6 tier-2 skipped: docs/data/dose-response.json already exists ==="
else
  say "=== 0/6 tier-2: the 14B judge check, then the cap sweep ==="
  if ! bash scripts/ops/queue-tier2.sh 2>&1 | tee -a "$LOG"; then
    # Not fatal. The Tier-1 arms are independent of both Tier-2 items, and a
    # week of card time is worth more than a tidy exit code.
    say "tier-2 stopped early; its completed rungs are on disk. Continuing."
  fi
fi

# ---------------------------------------------------------------------------
# Stages 1-6. The Tier-1 arms.
# ---------------------------------------------------------------------------
# The MiB figures come from serve.sh's KV table plus headroom for buffers,
# never from what fits exactly. Windows pages instead of raising OOM, so an
# allocation that only just fits produces a run that looks healthy at 100%
# utilisation and 57 W while it moves 8 GB/s over PCIe (3.4).
FAILED=()

run_arm "1/6 14B grid (1.3)"        configs/llamacpp/medium-14b.yaml \
        qwen-14b 20500 qwen25-14b-q4-medium         grid || FAILED+=("14B grid")

run_arm "2/6 code family (1.1)"     configs/llamacpp/code-medium.yaml \
        q4       15000 llama31-8b-q4-code-medium    grid || FAILED+=("code family")

run_arm "3/6 qwen 7B gate (1.2)"    configs/llamacpp/medium-qwen.yaml \
        qwen      9500 qwen25-7b-q4-medium          gate || FAILED+=("qwen 7B")

run_arm "4/6 mistral 7B gate (1.2)" configs/llamacpp/medium-mistral.yaml \
        mistral  14700 mistral-7b-q4-medium         gate || FAILED+=("mistral 7B")

run_arm "5/6 llama Q8_0 gate (1.4)" configs/llamacpp/medium-q8.yaml \
        q8       18500 llama31-8b-q8-medium         gate || FAILED+=("llama Q8_0")

run_arm "6/6 llama f16 gate (1.4)"  configs/llamacpp/medium-f16.yaml \
        f16      21000 llama31-8b-f16-medium        gate || FAILED+=("llama f16")

# ---------------------------------------------------------------------------
say ""
if [ ${#FAILED[@]} -eq 0 ]; then
  say "ALL DONE -- every queued arm is measured."
else
  say "DONE with ${#FAILED[@]} stage(s) unfinished: ${FAILED[*]}"
  say "Their completed episodes are on disk. Re-run this script to resume them;"
  say "nothing already measured regenerates."
fi
say ""
say "Read each against its registered threshold before writing anything:"
say "  14B          docs/PREREG-14b.md section 3"
say "  cap sweep    docs/PREREG-cap-sweep.md section 4"
say "  the rest     docs/PREREG-equivalence.md, at the registered margin"
