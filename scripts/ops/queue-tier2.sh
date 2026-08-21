#!/usr/bin/env bash
# The Tier-2 GPU lane, queued behind the corpus rebuild that is running now.
#
# One entry point, two stages, in cost order:
#
#   1. the 14B judge check       ~5 minutes of card   (Final Sweep 7.3 / 2.3)
#   2. the dose-response sweep   ~3-4 card-nights     (Final Sweep 7.4 / 2.4)
#
# Stage 1 goes first because it is minutes and because its answer can retire a
# whole section of the paper: if kappa is still ~0.29 on a 14B, Phase 2 is
# demoted to the appendix and stops being a thing to schedule around. Spending
# three card-nights on the sweep before asking a five-minute question is the
# ordering mistake this file exists to prevent.
#
# **Why a queue script at all.** Final Sweep 9 prices the GPU lane at 5-7 weeks
# of nights on a shared card and says plainly that the response to a busy card
# is to wait, never to kill. A human deciding when the next stage starts adds a
# night of latency per stage, and this project has already lost card time to
# exactly that -- the gap between a run finishing at 03:00 and somebody noticing
# at 09:00 is six hours of an idle 24 GiB card that a student account will
# otherwise take. So the stages chain themselves, and every one of them resumes
# from disk, so a death costs only the unfinished episodes of the stage it
# happened in.
#
# **The gate is copied verbatim from scripts/ops/queue-judge.sh.** It carries a
# CRLF fix that was paid for: nvidia-smi on Windows ends its lines with CRLF, a
# trailing CR survives `tr -d ' '` invisibly, and `$(( 24570 - 17348<CR> ))` is
# an arithmetic syntax error that leaves $free unset and lets the gate fall
# open. That is how a judge run was once launched into a card holding 17.3 GiB
# of another account's job. Do not rewrite it and do not drop the `tr -d ' \r'`.
#
# **Two kinds of pid appear in this file and they are not interchangeable.**
# The MSYS job pid that `serve.sh --detach` prints is reachable with `kill` from
# this shell and means nothing to `taskkill`. The Windows pid that
# Get-CimInstance returns is the reverse. Confusing the two has broken three
# separate things in this project, so each site below says which one it holds.
#
# **What this script will and will not stop.** It will stop an idle
# `llama-server` of ours on :8000 when a stage needs a different slot geometry,
# because that is our own inference server sitting on the card with nothing
# running against it. It will never stop a compute process it did not start,
# never touch another account's job, and never kill anything to make room. Set
# STOP_SERVER=0 to disable even that, in which case the script says what it
# would have done and waits for the card the ordinary way.
#
#   nohup bash scripts/ops/queue-tier2.sh >/dev/null 2>&1 &
#
# Launch it detached, not from an agent session: a session exit kills its
# children, which has already cost this project one full stage-1 run.
set -uo pipefail

LOG=runs/queue-tier2.log
LOCK=runs/queue-tier2.lock
WATCH_LOG=${WATCH_LOG:-runs/rebuild-corpus.log}
STOP_SERVER=${STOP_SERVER:-1}
GONE_LIMIT=${GONE_LIMIT:-5}    # consecutive misses (x60s) before believing it
STABLE=${STABLE:-3}            # consecutive clear checks (x60s) before believing it
# PY comes from _python.sh, which refuses the Microsoft Store stub. See its header.
. scripts/ops/_python.sh

# The sweep's geometry, from docs/PREREG-cap-sweep.md section 2. Every rung is
# served at the LARGEST rung's slot or the 6144 rung silently rejects its answer
# turns -- in the one arm the sweep exists to measure.
SWEEP_SLOT=$((15360 + 6144))
SWEEP_PARALLEL=3
SWEEP_CTX=$((SWEEP_SLOT * SWEEP_PARALLEL))

mkdir -p runs
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# One instance only. Two queues would each start a server and a pipeline the
# moment the card frees, and the second would race the first into the same run
# directory -- which is how RESEARCH-LOG 3.13's corpus got written twice over.
# mkdir is the atomic primitive available in every shell here; a pid file
# read-then-write is not.
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another queue holds $LOCK; not starting a second" >&2
  exit 3
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT INT TERM

# ---------------------------------------------------------------------------
# Is one of our pipelines still generating episodes?
# ---------------------------------------------------------------------------
# Match the command line, not the image name. Counting `python.exe` processes
# treats a test run or a one-off probe as the pipeline still being alive and --
# worse -- treats the moment between two of them as the pipeline having died,
# which starts the next stage while the card is still busy. Both `pipeline` and
# `ablate` count: rebuild-corpus.sh runs the two in sequence and the gap between
# them is seconds.
pipeline_alive() {
  powershell -NoProfile -Command "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*collabengine.cli pipeline*' -or \$_.CommandLine -like '*collabengine.cli ablate*' } | Measure-Object).Count" \
    | tr -d '\r '
}

# ---------------------------------------------------------------------------
# GPU gate -- verbatim from scripts/ops/queue-judge.sh. See the header.
# ---------------------------------------------------------------------------
wait_for_card() {
  local need=$1 clear_count=0 total used free holder
  say "waiting for ${need} MiB free, stable for ${STABLE} checks"
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
    if [ "$free" -ge "$need" ]; then
      clear_count=$((clear_count + 1))
      say "free ${free} MiB (${clear_count}/${STABLE})"
      [ "$clear_count" -ge "$STABLE" ] && break
    else
      # Name the holder. A job we could stop and a job we must not are
      # indistinguishable from the free figure alone.
      if [ "$clear_count" -ne 0 ]; then say "card reclaimed; restarting the count"; fi
      clear_count=0
      holder=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' \r' | paste -sd, -)
      say "free ${free} MiB, need ${need}; compute pids: ${holder:-none}"
    fi
    sleep 60
  done
}

# ---------------------------------------------------------------------------
# Stop OUR idle llama-server, and only that.
# ---------------------------------------------------------------------------
# Needed because the stages want different slot geometries: the rebuild's server
# is 4 x 18,432, stage 1 needs the 14B preset, stage 2 needs 3 x 21,504. A
# server left up at the wrong geometry does not fail loudly -- it holds ~13 GiB
# so the gate never opens, or it accepts the run and rejects the long answer
# turns. Neither is visible in a headline metric (3.4).
#
# The pid here is a WINDOWS pid, because this server was started by some other
# shell and its MSYS job pid does not exist in this one. That is the opposite of
# the pid queue-judge-14b.sh handles, and the reason both sites say so.
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
  powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" | ForEach-Object { Write-Output \$_.ProcessId; Stop-Process -Id \$_.ProcessId -Force }" 2>&1 | tee -a "$LOG"
  for _ in $(seq 1 30); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 || break
    sleep 5
  done
}

# ---------------------------------------------------------------------------
# 0. Wait for the rebuild that is running now.
# ---------------------------------------------------------------------------
say "queue-tier2: waiting for the corpus rebuild before anything else starts"
gone=0
while true; do
  if tr '\r' '\n' < "$WATCH_LOG" 2>/dev/null | grep -qa "ALL DONE"; then
    say "$WATCH_LOG reports ALL DONE"
    break
  fi
  alive=$(pipeline_alive)
  if [ "${alive:-0}" = "0" ]; then
    # Absent once is not finished. rebuild-corpus.sh runs a pipeline and then an
    # ablate, and the gap between them is a few seconds of no matching process.
    # Firing into that gap starts a second model on a card the rebuild is about
    # to reclaim, and two 13 GiB models in 24 GiB do not fail -- they page
    # (3.4).
    gone=$((gone + 1))
    say "no collabengine pipeline found (${gone}/${GONE_LIMIT} consecutive)"
    if [ "$gone" -ge "$GONE_LIMIT" ]; then
      say "pipeline gone for ${GONE_LIMIT} checks; treating the rebuild as over"
      break
    fi
  else
    gone=0
  fi
  sleep 60
done

# ---------------------------------------------------------------------------
# 1. The 14B judge check. Minutes.
# ---------------------------------------------------------------------------
if [ -f docs/data/judge-sweep.14b.json ]; then
  say "=== 1/2 skipped: docs/data/judge-sweep.14b.json already exists ==="
  say "    delete it to re-measure; re-running it silently would overwrite a result"
else
  say "=== 1/2 the 14B judge check (minutes of card, Final Sweep 7.3) ==="
  stop_our_server
  # Its own gate runs too; this one only avoids paying for a server start that
  # the gate would then refuse.
  wait_for_card "${JUDGE_MIB:-20500}"
  if ! bash scripts/ops/queue-judge-14b.sh 2>&1 | tee -a "$LOG"; then
    # Not fatal to the queue. The sweep does not depend on the judge answer, and
    # a card-night is worth more than a tidy exit code.
    say "the 14B judge check failed; continuing to the sweep, which is independent"
  fi
fi

# ---------------------------------------------------------------------------
# 2. The dose-response cap sweep. 3-4 card-nights.
# ---------------------------------------------------------------------------
say "=== 2/2 dose-response cap sweep (Final Sweep 7.4, PREREG-cap-sweep) ==="
stop_our_server

# 3 x 21,504 tokens of KV at 128 KiB/token is ~7.9 GiB against 4.6 GiB of
# weights: ~12.5 GiB, less than the default q4 preset's ~13.6. The gate asks for
# 14,000 so there is headroom for the buffers rather than exactly enough.
wait_for_card "${SWEEP_MIB:-14000}"

if ! curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
  say "starting llama-server at the sweep geometry: ${SWEEP_PARALLEL} x ${SWEEP_SLOT}"
  CTX=$SWEEP_CTX PARALLEL=$SWEEP_PARALLEL scripts/ops/serve.sh --detach 2>&1 | tee -a "$LOG"
  for _ in $(seq 1 90); do
    curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1 && break
    sleep 5
  done
fi

if ! curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
  say "server did not come up -- stopping before the sweep spends anything"
  exit 1
fi

# cap-sweep.sh preflights every rung against this server before it generates an
# episode, so a wrong geometry costs one probe request rather than a night.
if ! bash scripts/experiments/cap-sweep.sh 2>&1 | tee -a "$LOG"; then
  say "the sweep stopped early. Completed rungs are on disk and readable:"
  say "    python scripts/analysis/dose_response.py"
  say "Re-run this script or cap-sweep.sh to resume; nothing regenerates."
  exit 1
fi

say "ALL DONE -- both Tier-2 GPU items are measured."
say "  the curve:  python scripts/analysis/dose_response.py"
say "  read it against docs/PREREG-cap-sweep.md section 4 before writing anything"
