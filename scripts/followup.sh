#!/usr/bin/env bash
# Everything that runs once the corpus exists, chained so no GPU time is lost
# waiting for a human to notice the pipeline finished.
#
# Launch this detached (Start-Process on Windows) rather than from an agent
# session: a session exit kills its child processes, which has already cost this
# project one full stage-1 run.
set -u

CONFIG=configs/hf-local/hard.yaml
RUN_DIR=runs/qwen3-8b-local
LOG=runs/followup.log
GONE_LIMIT=${GONE_LIMIT:-5}   # consecutive misses (x60s) before believing it
gone=0

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "waiting for the pipeline to finish"
while true; do
  if tr '\r' '\n' < runs/pipeline.err 2>/dev/null | grep -qa "^corpus:"; then break; fi
  if tr '\r' '\n' < runs/pipeline.log 2>/dev/null | grep -qa "^corpus:"; then break; fi
  # A dead pipeline is also a reason to stop waiting; analyse what landed.
  #
  # Match the command line, not the image name. Counting `python` processes
  # treats a test run or a one-off probe as the pipeline still being alive, and
  # -- worse -- treats the moment between two of them as the pipeline having
  # died, which starts the judge while the card is still busy.
  alive=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*collabengine.cli pipeline*' } | Measure-Object).Count" | tr -d '\r ')
  if [ "${alive:-0}" = "0" ]; then
    # Absent once is not the same as finished. A restart leaves a gap of a few
    # seconds, and firing into it starts the judge -- a second 15 GiB model --
    # while the relaunched pipeline is loading its own. Both then fit only by
    # paging to host memory, which does not fail, it just runs the card at a
    # third speed with no error anywhere. Observed: judge at 01:01:00, pipeline
    # at 01:01:05, 66 W of a 210 W cap.
    gone=$((gone + 1))
    say "pipeline not found (${gone}/${GONE_LIMIT} consecutive)"
    if [ "$gone" -ge "$GONE_LIMIT" ]; then
      say "pipeline gone for ${GONE_LIMIT} checks; proceeding with what is on disk"
      break
    fi
  else
    gone=0
  fi
  sleep 60
done

# The judge loads a second copy of the weights, so it must not start while the
# pipeline still holds the card -- two 16 GiB models do not fit in 24 GiB.
say "waiting for the GPU to drain"
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [ "${used:-99999}" -lt 2000 ] && break
  sleep 20
done
say "gpu free (${used:-unknown} MiB)"

say "=== analyze ==="
python -u -m collabengine.cli analyze --config "$CONFIG" 2>&1 | tee -a "$LOG"

say "=== behavioural coding (local 8B judge) ==="
python -u -m collabengine.cli code --config "$CONFIG" \
  --judge self --judge-name local8b --episode-concurrency 8 2>&1 | tee -a "$LOG"

# The free Gemini tier is metered per day, so this can only ever be a subsample.
# It exists to put a number on how far the local judge can be trusted.
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
