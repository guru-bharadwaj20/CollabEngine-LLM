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
#   scripts/ops/serve.sh --model q8 --detach       # a named preset
#   scripts/ops/serve.sh --list                    # every preset and its arithmetic
#
# Then always: python scripts/ops/preflight.py --config configs/llamacpp/<tier>.yaml
#
# **The presets exist because slot geometry is per-model arithmetic, not a
# preference.** Every arm added in Final Sweep 1.2-1.4 changes the weights, and
# the weights change how much of the card is left for KV cache, and `-c` is
# divided across `--parallel` slots. Getting that wrong does not fail loudly: it
# shrinks the per-request window until the final-round turns -- the ones that
# submit the answer -- stop fitting, in the arm with the longest contexts.
#
# Every preset holds **18,432 tokens per slot**, which is the number the configs
# require and the only thing that must not vary across arms. What varies is how
# many slots fit, which costs throughput and nothing else. KV per token is
# `layers x kv_heads x head_dim x 2 (K,V) x 2 bytes`:
#
#   model                     weights    KV/token   slots   total
#   llama Q4_K_M (default)     4.6 GiB   128 KiB      4     ~14.2 GiB
#   llama Q8_0                 8.0 GiB   128 KiB      4     ~17.6 GiB
#   llama f16                 15.0 GiB   128 KiB      2     ~20.1 GiB
#   qwen 7B Q4_K_M             4.4 GiB    56 KiB      4     ~ 8.4 GiB
#   mistral 7B Q4_K_M          4.1 GiB   128 KiB      4     ~13.7 GiB
#   qwen 14B Q4_K_M            8.9 GiB   192 KiB      3     ~19.6 GiB
#
# The card is 24 GiB and it is shared (RESEARCH-LOG 3.9), so the f16 and 14B
# rows are deliberately one slot short of what would fit alone. Windows does not
# raise OOM as allocations approach the card -- it pages to host memory over
# PCIe while every dashboard still reads 100% utilisation (3.4).
set -uo pipefail

BIN=${BIN:-vendor/llamacpp/llama-server.exe}
PORT=${PORT:-8000}
LOG=runs/llama-server.log

# preset -> "file|parallel". CTX is always parallel * 18432, computed below, so
# the two cannot drift apart the way they did in the `--parallel` trap.
SLOT=18432
presets() {
  case "$1" in
    q4|llama-q4|default) echo "models/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf|4" ;;
    q8|llama-q8)         echo "models/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf|4" ;;
    f16|llama-f16)       echo "models/Meta-Llama-3.1-8B-Instruct.f16.gguf|2" ;;
    qwen|qwen-7b)        echo "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf|4" ;;
    mistral|mistral-7b)  echo "models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf|4" ;;
    qwen-14b|14b)        echo "models/qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf|3" ;;
    *) return 1 ;;
  esac
}

PRESET=${PRESET:-q4}
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --model) PRESET="$2"; shift 2 ;;
    --list)
      for p in q4 q8 f16 qwen mistral qwen-14b; do
        spec=$(presets "$p"); f=${spec%|*}; n=${spec#*|}
        printf '  %-10s %-52s %d slots x %d tokens\n' "$p" "$f" "$n" "$SLOT"
      done
      exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

if ! spec=$(presets "$PRESET"); then
  echo "unknown --model $PRESET; try --list" >&2
  exit 2
fi
MODEL=${MODEL:-${spec%|*}}
PARALLEL=${PARALLEL:-${spec#*|}}
CTX=${CTX:-$((PARALLEL * SLOT))}

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
