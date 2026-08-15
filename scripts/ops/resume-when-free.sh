#!/usr/bin/env bash
# Wait for the card, then bring the whole answer-budget run back up unattended.
#
# 2026-08-12 22:45: a foreign Ollama process (another account) holds ~21.8 GiB of
# the 24 GiB card with llama3.1:8b at a 128k context. Our server's 15 GiB was
# therefore almost entirely paged to host RAM, which is RESEARCH-LOG 3.4's exact
# failure mode: prefill fell from 3,658 to ~800 tok/s, decode from ~34 to 4 t/s
# per slot, power from 195 W to 57 W, while utilisation still read 61-97%.
#
# Every non-destructive route was tried and failed:
#   * POST /api/generate keep_alive:0 -- accepted, `done_reason: unload`, and the
#     runner keeps its allocation. Repeated with our server stopped; no change.
#   * reload at num_ctx 512 -- HTTP 200 with a 4.7 s reload, still reports
#     context_length 131072 and 20.5 GiB. This build ignores the option.
#   * taskkill on the runner -- access denied; it belongs to another account and
#     needs elevation.
#
# So the card is freed by an administrator, or not at all:
#
#     taskkill /PID 42436 /T /F      # in an ELEVATED shell: tray -> daemon -> runner
#
# This script does the rest. It waits for real headroom, restarts our server with
# the geometry the -ans configs need, preflights, and resumes the run. Every
# stage resumes from disk, so the ~11 episodes already generated are kept.
#
#   scripts/ops/resume-when-free.sh &
set -uo pipefail

NEED_MIB=${NEED_MIB:-16000}     # weights 4.6 + KV 9.0 + buffers, with margin
LOG=runs/resume-when-free.log
LOCK=runs/resume-when-free.lock
mkdir -p runs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# One instance only. Two watchers would each start a server and a pipeline the
# moment the card frees, and the second would race the first into the same run
# directory -- which is how 3.13's corpus got written twice over. mkdir is the
# atomic primitive available in every shell here; a pid file read-then-write is
# not, and this script exists precisely because processes here are hard to see.
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another watcher holds $LOCK; not starting a second" >&2
  exit 3
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM

say "waiting for ${NEED_MIB} MiB free; the foreign Ollama process must be stopped first"

while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  case "$free" in
    ''|*[!0-9]*) say "cannot read free VRAM; retrying"; sleep 60; continue ;;
  esac
  if [ "$free" -ge "$NEED_MIB" ]; then
    say "$free MiB free -- starting"
    break
  fi
  sleep 60
done

# Start our server only once the memory is genuinely there. Starting it under
# pressure is worse than not starting: llama.cpp falls back to a partial offload,
# which is the one configuration that looks like it works and runs at CPU speed.
if ! curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
  say "starting llama-server"
  scripts/ops/serve.sh --detach 2>&1 | tee -a "$LOG"
  for _ in $(seq 1 60); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break
    sleep 5
  done
fi

if ! curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
  say "server did not come up -- stopping"
  exit 1
fi

say "server healthy; resuming the answer-budget run"
exec bash scripts/experiments/answer-run.sh
