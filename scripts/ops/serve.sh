#!/usr/bin/env bash
# Start llama-server with the geometry docs/LLAMACPP-SETUP.md derives.
#
# The flags are not preferences. `-c` is divided across `--parallel` slots, so
# the two must move together or every long prompt is silently rejected; `-ub`
# is what bounds the prefill activation peak and is the actual reason `xhard`
# runs here at all. Changing either without redoing the arithmetic in the setup
# doc is how this project loses its fourth corpus.
#
#   scripts/ops/serve.sh              # foreground, logs to runs/llama-server.log
#   scripts/ops/serve.sh --detach     # background, prints the pid
#
# Then always: python scripts/ops/preflight.py --config configs/llamacpp/<tier>.yaml
set -uo pipefail

BIN=${BIN:-vendor/llamacpp/llama-server.exe}
MODEL=${MODEL:-models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf}
PORT=${PORT:-8000}
CTX=${CTX:-73728}
PARALLEL=${PARALLEL:-4}
LOG=runs/llama-server.log

mkdir -p runs

if [ ! -x "$BIN" ]; then
  echo "no llama-server at $BIN -- see docs/LLAMACPP-SETUP.md" >&2
  exit 2
fi
if [ ! -f "$MODEL" ]; then
  echo "no model at $MODEL -- see docs/LLAMACPP-SETUP.md" >&2
  exit 2
fi

# Refuse to start a second server on the port rather than racing one that is
# already serving a different model. verify_model in the configs would catch
# the mismatch, but only after a tier had been queued behind it.
if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
  echo "something is already serving on :${PORT}" >&2
  curl -s "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | head -c 300 >&2
  echo >&2
  exit 3
fi

# Per-slot context, printed because it is the number that actually bounds a
# request and the one that is easiest to get wrong.
echo "llama-server: $((CTX / PARALLEL)) tokens/slot (-c $CTX / --parallel $PARALLEL)"

run() {
  "$BIN" -m "$MODEL" \
    --host 127.0.0.1 --port "$PORT" \
    -ngl 99 \
    -c "$CTX" --parallel "$PARALLEL" \
    -b 2048 -ub 512 \
    --flash-attn on \
    "$@"
}

if [ "${1:-}" = "--detach" ]; then
  run >>"$LOG" 2>&1 &
  echo "pid $! -> $LOG"
else
  run 2>&1 | tee -a "$LOG"
fi
