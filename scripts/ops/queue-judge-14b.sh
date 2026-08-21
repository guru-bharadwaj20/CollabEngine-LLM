#!/usr/bin/env bash
# Is kappa = 0.29 a model-capacity limit or a wording one? MINUTES OF GPU.
#
# **This is not a card-night. It is 160 requests of 8 tokens each -- roughly
# five minutes once the weights are loaded**, and the loading is most of it.
# Final Sweep 7.3 calls it "acceptable and free" and it is the cheapest item on
# the whole page: the 14B is already on disk for 2.3, the 40 hand-coded messages
# are already in docs/data/handcode-sample.json, and the four codebooks are
# already written. Nothing here generates an episode.
#
# What it settles. RESEARCH-LOG 4.17 ran four codebooks against the same 40
# human labels on the local 8B and moved kappa from 0.07 to 0.288, then stopped:
# v2 0.219, v4 0.230, v5 0.288, v3 (coarse) 0.056, against a run-to-run wobble of
# roughly +/-0.02 measured on the same prompt and the same seeds. Its conclusion
# was "not blocked on wording any more" and its named next step was "a larger
# judge -- testable on these same 40 messages the moment a 14B is on disk". A 14B
# is on disk. This is that test.
#
# The two outcomes are both useful and only one of them is expensive:
#   * kappa rises materially (say past ~0.5) -> the limit was capacity, Phase 2
#     is re-codable with the 14B, and 4.8's convergent-validity claim has a road
#     back. That costs a coding pass over 468 messages, which is still not a
#     night.
#   * kappa stays near 0.29 -> the limit is neither wording nor capacity at this
#     scale, and Final Sweep 7.3's third option applies: Phase 2 is demoted to
#     the appendix and stops anchoring the paper's credibility. That costs
#     nothing but a section move, and the ablation-side nulls do not depend on it.
#
# **The GPU gate below is copied verbatim from scripts/ops/queue-judge.sh and
# must stay that way.** It carries a fix that was paid for: nvidia-smi on Windows
# ends its lines with CRLF, a trailing CR survives `tr -d ' '` invisibly, and
# `$(( 24570 - 17348<CR> ))` is an arithmetic syntax error that leaves $free
# unset and lets the gate fall open. The first version of that script did exactly
# that and launched a judge into a card holding 17.3 GiB of another account's job
# -- the precise outcome the gate exists to prevent. A guard that fails open is
# worse than no guard, because it is trusted. Do not rewrite it, do not
# "simplify" it, and do not drop the `tr -d ' \r'`.
#
# It waits; it does not kill. The card is shared with student accounts and a
# second project, several of them administrators, so no lock is possible and the
# only correct response to a busy card is to wait (RESEARCH-LOG 3.9).
#
# The one process this script will stop is the llama-server **it started
# itself**, and only if it started it. That pid is an MSYS job pid, not a Windows
# pid: `kill` from this shell reaches it, `taskkill /PID` would need the Windows
# pid instead and would be aimed at the wrong process. Getting those two confused
# has broken three separate things in this project; the script never leaves the
# MSYS side.
#
#   bash scripts/ops/queue-judge-14b.sh
set -uo pipefail

CONFIG=${CONFIG:-configs/llamacpp/medium-14b.yaml}
LOG=runs/queue-judge-14b.log
DATA=docs/data
NEED_MIB=${NEED_MIB:-20500}   # qwen-14b preset: 8.9 GiB weights + 3 x 18,432 KV
STABLE=${STABLE:-3}           # consecutive clear checks (x60s) before believing it
# PY comes from _python.sh, which refuses the Microsoft Store stub. See its header.
. scripts/ops/_python.sh

mkdir -p runs
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [ ! -f "$DATA/handcode-sample.json" ]; then
  say "no $DATA/handcode-sample.json -- the human column is the whole experiment"
  exit 2
fi

# ---------------------------------------------------------------------------
# GPU gate -- verbatim from scripts/ops/queue-judge.sh. See the header.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Serve the 14B, but never on top of somebody else's server.
# ---------------------------------------------------------------------------
STARTED=""
if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
  served=$(curl -s http://127.0.0.1:8000/v1/models 2>/dev/null | tr -d ' \r')
  case "$served" in
    *qwen2.5-14b*|*Qwen2.5-14B*)
      say "a 14B server is already up on :8000; reusing it" ;;
    *)
      # Not ours to replace. verify_model in the config would catch the mismatch
      # anyway, but only after the run had been queued behind it.
      say "something else is serving on :8000:"
      echo "$served" | head -c 300 | tee -a "$LOG"; echo | tee -a "$LOG"
      say "not restarting a server this script did not start. Stop it by hand"
      say "if it is yours, then re-run this. Exiting."
      exit 3 ;;
  esac
else
  say "starting llama-server on the qwen-14b preset (3 slots x 18,432)"
  pidline=$(scripts/ops/serve.sh --model qwen-14b --detach 2>&1 | tee -a "$LOG")
  STARTED=$(echo "$pidline" | sed -n 's/^pid \([0-9]*\).*/\1/p')
  say "server pid (MSYS job pid, not a Windows pid): ${STARTED:-unknown}"
  for _ in $(seq 1 90); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break
    sleep 5
  done
fi

# Stop only what we started, and only on the way out. A server we merely found
# stays up: it may be the cap sweep's, and queue-tier2.sh chains the two.
cleanup() {
  if [ -n "$STARTED" ]; then
    say "stopping the llama-server this script started (msys pid $STARTED)"
    kill "$STARTED" 2>/dev/null
  fi
}
trap cleanup EXIT INT TERM

if ! curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
  say "server did not come up in 7.5 minutes -- stopping"
  exit 1
fi

# ---------------------------------------------------------------------------
# The measurement.
# ---------------------------------------------------------------------------
#
# judge_sweep.py rather than handcode_kappa.py, for two reasons and neither is
# preference. handcode_kappa.py hardcodes configs/llamacpp/hard.yaml, so it
# cannot be aimed at the 14B without editing it, and it writes its labels back
# into the `local8b` column of handcode-sample.json, which would overwrite the
# 8B result this is supposed to be compared against. judge_sweep.py takes
# --config and keeps the human column untouched.
#
# Running all four codebooks rather than only v5 costs three extra minutes and
# buys the thing that makes the answer interpretable: 4.17's finding is a
# *pattern* across books (v3's catch-all collapse, v5's worked examples), and a
# single number from a single book cannot say whether the pattern moved with the
# model or only its level did.
#
# judge_sweep.py's output path is hardcoded, so the 8B sweep is moved aside and
# put back rather than clobbered. 4.17's table is a published result.
if [ -f "$DATA/judge-sweep.json" ] && [ ! -f "$DATA/judge-sweep.8b.json" ]; then
  say "preserving the 8B sweep as $DATA/judge-sweep.8b.json"
  cp "$DATA/judge-sweep.json" "$DATA/judge-sweep.8b.json"
fi

say "=== four codebooks x 40 messages on Qwen2.5-14B (this is the minutes part) ==="
if ! $PY -u scripts/analysis/judge_sweep.py --config "$CONFIG" 2>&1 | tee -a "$LOG"; then
  say "the sweep failed; the 8B result on disk is unchanged"
  [ -f "$DATA/judge-sweep.8b.json" ] && cp "$DATA/judge-sweep.8b.json" "$DATA/judge-sweep.json"
  exit 1
fi

if [ -f "$DATA/judge-sweep.json" ]; then
  cp "$DATA/judge-sweep.json" "$DATA/judge-sweep.14b.json"
  # Restore the 8B file to the path 4.17's table was produced from, so the
  # published number stays reproducible by the documented command.
  [ -f "$DATA/judge-sweep.8b.json" ] && cp "$DATA/judge-sweep.8b.json" "$DATA/judge-sweep.json"
  say "14B labels -> $DATA/judge-sweep.14b.json (8B restored to judge-sweep.json)"
fi

# Confirm it was not paging the whole time. Five minutes that took fifteen is
# still a valid kappa, but the throughput line in the log would be a lie about
# the hardware (RESEARCH-LOG 3.4).
say "post-run PCIe sample (sustained >2000 MB/s means it was paging):"
nvidia-smi dmon -s put -c 3 2>&1 | tee -a "$LOG"

say "READ THE TABLE ABOVE against RESEARCH-LOG 4.17:"
say "  v2 0.219 | v4 0.230 | v5 0.288 | v3 (coarse, not comparable) 0.056"
say "  run-to-run wobble on this measurement is ~+/-0.02, so only a move"
say "  larger than that is a move. Phase 2 needs ~0.6; 4.8's convergent"
say "  validity claim needs ~0.78. Anything short of 0.6 means Final Sweep"
say "  7.3's third option -- demote Phase 2 to the appendix -- and that is a"
say "  result, not a failure."
say "ALL DONE"
