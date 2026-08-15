#!/usr/bin/env bash
# Wait for midnight, take the card the moment it is genuinely free, and spend
# it on the work RESEARCH-LOG 4.19 leaves queued.
#
# Why stability rather than a single reading. This card is shared with student
# accounts and a second project of the owner's, half those accounts are
# administrators, and no lock is possible -- so nothing here may evict anything
# and a single free reading is not evidence the card is free. It is evidence of
# one moment, and the moment between two phases of somebody else's job looks
# exactly like an idle card. scripts/queue-judge.sh already established the
# house pattern (N GiB free, stable across three checks); this follows it.
#
# It signals exactly one process: the llama-server it started itself, by the
# Windows pid it looked up after starting it. A pid here is meaningless without
# saying whose -- serve.sh --detach prints an MSYS pid, which is a different
# number in a different namespace, and killing by it is how the wrong thing
# gets killed on this box.
#
# The queue, in order, and why this order:
#
#   1. frozen_replay   4 agents x 48 -- the fungibility metric, D(frozen) - D(live).
#                      The primary open item. Runs first and alone so a failure
#                      later cannot obscure whether it completed.
#   2. capacity        48 episodes -- a 3-agent team that never had agent i.
#   3. random_message  free, zero model calls -- volume-matched excision.
#
# 2 and 3 are what separate "this agent mattered" from "one fewer worker" and
# from "less text in context", which is 4.19's recommendation. They reuse the
# recorded seeds, so they interpret the pilot rather than testing a hypothesis
# the pilot generated -- that distinction is why the confirmatory run is NOT
# queued here. It needs fresh seeds and a preregistration first.
#
# The card is released when the queue finishes, the same way guard.sh releases
# it on ALL DONE. An idle server holding 21 GB overnight is an eviction of the
# other accounts by another name.
#
#   nohup bash scripts/overnight-watch.sh >/dev/null 2>&1 &
#   tail -f runs/overnight-watch.log
set -uo pipefail

LOG=runs/overnight-watch.log
CONFIG=${CONFIG:-configs/llamacpp/medium-ans.yaml}
NEED_FREE=${NEED_FREE:-22000}      # MiB. The -ans server's resident set is ~21 GB.
STABLE=${STABLE:-3}                # consecutive passing checks before acquiring
POLL=${POLL:-120}                  # seconds between checks
WAKE_DATE=${WAKE_DATE:-20260814}   # start looking when the date flips to this
HEARTBEAT=${HEARTBEAT:-15}         # log a still-waiting line every N checks

# The answer-budget geometry: 129024 / 7 = 18432 tokens per slot, and
# max_concurrency 7 in the config is matched to --parallel. serve.sh defaults
# to 73728/4, which is the same slot size but only four of them -- starting
# with the defaults would queue three of every seven requests behind a slot.
export CTX=129024 PARALLEL=7

SERVER_PID=""
mkdir -p runs

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

free_mib() {
  local used total
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "${used:-}" ] && [ -n "${total:-}" ]; then echo $((total - used)); else echo 0; fi
}

release() {
  if [ -n "$SERVER_PID" ]; then
    taskkill //PID "$SERVER_PID" //F >/dev/null 2>&1
    say "RELEASED llama-server win pid $SERVER_PID stopped; card free"
    SERVER_PID=""
  fi
}
trap release EXIT INT TERM

say "watch armed: waiting for $WAKE_DATE, then >= ${NEED_FREE} MiB free stable across ${STABLE} checks every ${POLL}s"

while [ "$(date +%Y%m%d)" -lt "$WAKE_DATE" ]; do sleep 60; done
say "MIDNIGHT reached -- now watching the card"

streak=0
ticks=0
while :; do
  f=$(free_mib)
  ticks=$((ticks + 1))
  if [ "$f" -ge "$NEED_FREE" ]; then
    streak=$((streak + 1))
    say "candidate ${streak}/${STABLE}: ${f} MiB free"
    [ "$streak" -ge "$STABLE" ] && break
  else
    if [ "$streak" -gt 0 ]; then say "reset: ${f} MiB free, below ${NEED_FREE}"; fi
    streak=0
    if [ $((ticks % HEARTBEAT)) -eq 0 ]; then
      say "heartbeat: ${f} MiB free, still waiting (check ${ticks})"
    fi
  fi
  sleep "$POLL"
done

say "ACQUIRED ${f} MiB free, stable across ${STABLE} checks"

bash scripts/serve.sh --detach >>"$LOG" 2>&1
for _ in $(seq 1 60); do
  curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break
  sleep 5
done
if ! curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
  say "FAILED llama-server did not answer /health within 300s -- nothing started"
  exit 1
fi
SERVER_PID=$(tasklist //FI "IMAGENAME eq llama-server.exe" //NH //FO CSV 2>/dev/null \
  | head -1 | cut -d, -f2 | tr -d '"')
say "SERVER-UP win pid ${SERVER_PID:-unknown}, $((CTX / PARALLEL)) tokens/slot"

if ! python -u scripts/preflight.py --config "$CONFIG" >>"$LOG" 2>&1; then
  say "FAILED preflight rejected the run -- nothing after this would be clean"
  exit 1
fi
say "preflight ok"

say "STARTED 1/2 frozen_replay -- 4 agents x 48 episodes"
if python -u -m collabengine.cli ablate --config "$CONFIG" --modes frozen_replay >>"$LOG" 2>&1; then
  say "DONE 1/2 frozen_replay complete"
else
  say "FAILED 1/2 frozen_replay returned nonzero -- see $LOG"
  exit 1
fi

say "STARTED 2/2 capacity + random_message controls"
if python -u -m collabengine.cli ablate --config "$CONFIG" \
     --modes capacity,random_message >>"$LOG" 2>&1; then
  say "DONE 2/2 controls complete"
else
  say "FAILED 2/2 controls returned nonzero -- frozen_replay above is unaffected"
fi

say "ALL DONE -- releasing the card"
