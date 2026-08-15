#!/usr/bin/env bash
# Phase 2 behavioural coding, queued behind whoever else is using the card.
#
# This is followup.sh's tail end with a different gate on the front. followup.sh
# waits for *our* pipeline to exit; this waits for the GPU to be free of anyone,
# because this workstation is shared and a second account runs jobs on the same
# device.
#
# Why it exists: a coding run launched at 10:03:38 into a card that a foreign
# job (profile_e8_efficiency.py, another project, 17.3 GiB) had claimed at
# 10:03:18. Qwen3-8B in bf16 needs 15.3 GiB for weights alone, so the two do not
# fit in 24 GiB. The run did not fail -- Windows WDDM pages the working set to
# host RAM over PCIe instead of raising OOM, so nvidia-smi showed 100%
# utilisation while the card drew 60 W of 210 W and moved 8 GB/s in and 11 GB/s
# out over the bus. See RESEARCH-LOG 3.4. The lesson repeated here is that on
# this machine "the GPU is busy" and "the GPU is thrashing" look identical in
# every headline metric, so the only safe policy is not to start until the card
# is actually empty.
#
# It waits rather than kills. The foreign job writes to
# experiments/results/evals/FINAL/ and was launched with --resume; killing it to
# reclaim the card is not ours to do.
set -u

CONFIG=configs/hf-local/hard.yaml
RUN_DIR=runs/qwen3-8b-local
LOG=runs/queue-judge.log
NEED_MIB=${NEED_MIB:-18000}   # free VRAM before Qwen3-8B is worth starting
STABLE=${STABLE:-3}           # consecutive clear checks (x60s) before believing it

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for ${NEED_MIB} MiB free, stable for ${STABLE} checks"
clear_count=0
while true; do
  # Strip CR as well as spaces. nvidia-smi on Windows ends lines with CRLF, and
  # a trailing CR survives `tr -d ' '` invisibly: $(( 24570 - 17348<CR> )) is an
  # arithmetic syntax error, which leaves $free unset and lets the gate fall
  # open. The first version of this script did exactly that and launched the
  # judge into a card holding 17.3 GiB of someone else's job -- the precise
  # outcome the gate exists to prevent. A guard that fails open is worse than
  # no guard, because it is trusted.
  read -r total used < <(nvidia-smi --query-gpu=memory.total,memory.used \
    --format=csv,noheader,nounits | tr -d ' \r' | tr ',' ' ')
  case "${total:-}${used:-}" in
    *[!0-9]*|"") say "unparseable nvidia-smi output; treating the card as busy"
                 clear_count=0; sleep 60; continue ;;
  esac
  free=$(( total - used ))
  if [ "$free" -ge "$NEED_MIB" ]; then
    clear_count=$((clear_count + 1))
    say "free ${free} MiB (${clear_count}/${STABLE})"
    [ "$clear_count" -ge "$STABLE" ] && break
  else
    # Name the holder. A job we could stop and a job we must not are
    # indistinguishable from the free figure alone.
    if [ "$clear_count" -ne 0 ]; then say "card reclaimed; restarting the count"; fi
    clear_count=0
    holder=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' \r' | paste -sd, -)
    say "free ${free} MiB, need ${NEED_MIB}; compute pids: ${holder:-none}"
  fi
  sleep 60
done

say "=== behavioural coding (local 8B judge) ==="
python -u -m collabengine.cli code --config "$CONFIG" \
  --judge self --judge-name local8b --episode-concurrency 8 2>&1 | tee -a "$LOG"

# Confirm the run was not silently paging the whole time. A coding pass that
# took three times as long as it should have is still a valid corpus, but the
# throughput number in the log would be a lie about the hardware.
say "post-run PCIe sample (sustained >2000 MB/s means it was paging):"
nvidia-smi dmon -s put -c 3 2>&1 | tee -a "$LOG"

if [ -n "${GEMINI_API_KEY:-}" ]; then
  say "=== frontier subsample for kappa ==="
  python -u -m collabengine.cli code --config "$CONFIG" \
    --judge gemini --judge-name gemini --condition baseline --limit 1 2>&1 | tee -a "$LOG"

  if [ -f "$RUN_DIR/codes.local8b.jsonl" ] && [ -f "$RUN_DIR/codes.gemini.jsonl" ]; then
    say "=== kappa ==="
    python -u -m collabengine.cli kappa \
      "$RUN_DIR/codes.local8b.jsonl" "$RUN_DIR/codes.gemini.jsonl" 2>&1 | tee -a "$LOG"
  fi
else
  say "GEMINI_API_KEY unset; skipping the frontier subsample and kappa"
fi

say "=== convergent validity ==="
python -u -m collabengine.cli converge --config "$CONFIG" \
  --codes "$RUN_DIR/codes.local8b.jsonl" --permutations 2000 2>&1 | tee -a "$LOG"

say "ALL DONE"
