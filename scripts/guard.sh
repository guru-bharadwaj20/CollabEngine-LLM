#!/usr/bin/env bash
# Keep the run alive across anything that kills the server.
#
# WHAT THIS IS NOT: it is not a lock on the GPU. There is no such thing here.
# Student, Student2, Student5 and CSC10 are all in BUILTIN\Administrators on
# this box, and an elevated administrator can terminate any process regardless
# of its DACL -- take ownership, rewrite the ACL, enable SeDebugPrivilege, or
# just reboot. Hardening the process ACL would stop only the accounts that
# already cannot touch it (Student3/4/6, standard users, blocked by the default
# per-user process security that gave *us* "access denied" on Student6's Ollama
# on 2026-08-12). So this script does not try to make the run unkillable. It
# makes being killed cheap: ~2 minutes instead of the ~8 hours that the watcher
# teardown cost on 2026-08-13.
#
# It also works against the things that are far likelier than a hostile admin:
# a crash, an OOM, a Windows update, a session teardown.
#
# RULES IT WILL NOT BREAK
#
#   1. It kills nothing except our own client, and only by a pid it resolved
#      from a command line rooted in this repo. Another user's job -- and the
#      user's other project in Codec-Answer-Utility-Selection-Engine -- is never
#      touched, never signalled, never raced.
#   2. If the card is busy when the server needs restarting, it WAITS. It does
#      not compete for VRAM and it does not evict anyone. Waiting is correct
#      even when the waiting is ours: two jobs paging against each other on a
#      24 GB card cost 4x throughput on 2026-08-12 (RESEARCH-LOG 3.4).
#   3. Client first, then server -- always. Killing the server under a live
#      client is what wrote 137 errored turns across 22 episodes in 3.13.
#
#   scripts/guard.sh            # foreground
#   scripts/guard.sh --detach   # background, prints the pid
set -uo pipefail

PORT=${PORT:-8000}
PARALLEL=${PARALLEL:-7}
CTX=${CTX:-129024}
# What the 7-slot server needs on the card. Measured, not derived: 21,399 MiB
# resident with -c 129024 --parallel 7.
NEED_MIB=${NEED_MIB:-21000}
# Consecutive failed health checks before we believe it. One missed probe is a
# busy server, not a dead one; acting on a single miss would restart a server
# that was merely slow and hand the corpus two clients writing one file.
STRIKES=${STRIKES:-3}
POLL=${POLL:-15}
LOG=runs/guard.log
LOCK=runs/.guard.lock
REPO=$(cd "$(dirname "$0")/.." && pwd)

mkdir -p runs
say() { echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a "$LOG"; }

# Re-exec detached, before the lock is taken, so the child is the holder. The
# parent must not touch the lock at all or its exit trap would delete the lock
# the child is relying on.
if [ "${1:-}" = "--detach" ]; then
  shift
  nohup "$0" "$@" >/dev/null 2>&1 &
  echo "guard pid $! -> $LOG"
  exit 0
fi

# --- single instance -------------------------------------------------------
# Atomic: mkdir either creates or fails, with no window between the test and
# the create. Three watchers ran simultaneously on 2026-08-12 because the check
# was `[ -f ... ]` followed by a touch. The pid file inside lets a later start
# tell a live holder from a lock left behind by SIGKILL, which is the other half
# of that bug -- the trap does not fire when the process is killed outright.
# --list-clients only reads, so it neither takes the lock nor is blocked by one.
# If it did take the lock it would be useless for its actual purpose: inspecting
# the selector while a guard is live.
if [ "${1:-}" != "--list-clients" ]; then
# The pid recorded is the WINDOWS pid, not the MSYS one, and liveness is asked
# of Windows. `kill -0 $msys_pid` is not a reliable liveness test across two
# separately-launched Git Bash processes: a second guard asked it about a live
# holder, got "no", declared the lock stale, deleted it, and took it -- leaving
# two guards and, when one exited, no lock at all. Observed, not theorised.
# Same root cause as the 3.13 cleanup that signalled MSYS pids and killed
# nothing: in this environment a pid is meaningless without saying whose.
WINPID=$(cat "/proc/$$/winpid" 2>/dev/null || echo $$)

holder_alive() {
  # Pid reuse is real, so the image and the command line are checked too. A
  # recycled pid that happens to belong to something else must read as dead.
  local p=$1
  [ -n "$p" ] || return 1
  local out
  out=$(powershell.exe -NoProfile -NonInteractive -Command "
    \$p = Get-CimInstance Win32_Process -Filter 'ProcessId=$p' -ErrorAction SilentlyContinue
    if (\$p -and \$p.Name -eq 'bash.exe' -and \$p.CommandLine -like '*guard.sh*' \
        -and \$p.CommandLine -notlike '*.claude*') { 'alive' } else { 'dead' }" \
    2>/dev/null | tr -d ' \r\n')
  [ "$out" = "alive" ]
}

if ! mkdir "$LOCK" 2>/dev/null; then
  holder=$(cat "$LOCK/pid" 2>/dev/null || echo "")
  if holder_alive "$holder"; then
    echo "guard already running as win pid $holder -- nothing to do" >&2
    exit 0
  fi
  say "clearing a stale lock (holder '${holder:-unknown}' is not a live guard)"
  rm -rf "$LOCK"
  mkdir "$LOCK" 2>/dev/null || { echo "could not take $LOCK" >&2; exit 1; }
fi
echo "$WINPID" >"$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT INT TERM
fi

# Bounded, so a hung socket is a miss rather than a stalled guard. Verified
# against the live server at full saturation: /health answers 200 {"status":
# "ok"} with all 7 slots processing, so a busy card cannot be mistaken for a
# dead one. Older llama.cpp builds returned 503 when no slot was idle, which on
# this workload would have been a false positive on essentially every probe.
healthy() {
  curl -sf --max-time 10 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1
}

# The run is over when the runner says so. This matters more than it looks: a
# guard that keeps restarting a finished run would hold ~21 GB of a 24 GB card
# indefinitely and starve the user's other project forever -- turning "protect
# my run" into "nothing else ever runs on this box". Finishing means letting go.
run_finished() {
  grep -q "ALL DONE" runs/answer-run.log 2>/dev/null
}

free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | head -1 | tr -d ' \r'
}

# Windows pids for our own client, resolved through CIM rather than `ps -W`.
# `ps -W` prints the MSYS pid first and the Windows pid fourth, and signalling
# the first one does nothing at all -- that is precisely how the 3.13 cleanup
# believed it had stopped a client that was still writing.
#
# Four independent conditions have to hold before a pid is eligible, because
# the obvious filter is wrong in both directions and was caught being wrong:
# matching '*CollabEngine-LLM*' in the command line MISSED the real runner --
# it is invoked as `bash scripts/answer-run.sh`, a relative path that never
# names the repo -- while MATCHING three of Claude Code's own shells and the
# probe process doing the matching. A selector that misses its target and hits
# bystanders is the 3.13 cleanup bug with a different mask on.
#
#   1. the image is python.exe or bash.exe -- never a shell, never a probe
#   2. the command line names our entry point
#   3. it is not a Claude Code shell, this guard, or a -NoProfile probe
#   4. the owner is the account running this guard
#
# (4) is what makes rule 1 of the header a property of the code rather than a
# promise in a comment: a process belonging to Student2 or to the user's other
# project cannot be selected even if it somehow satisfied the other three.
client_pids() {
  powershell.exe -NoProfile -NonInteractive -Command "
    Get-CimInstance Win32_Process |
      Where-Object { \$_.Name -in @('python.exe','bash.exe') -and
                     \$_.CommandLine -and
                     (\$_.CommandLine -like '*collabengine.cli*' -or
                      \$_.CommandLine -like '*scripts/answer-run.sh*' -or
                      \$_.CommandLine -like '*scripts\\answer-run.sh*') -and
                     \$_.CommandLine -notlike '*.claude*' -and
                     \$_.CommandLine -notlike '*guard.sh*' -and
                     \$_.CommandLine -notlike '*NoProfile*' } |
      Where-Object { (Invoke-CimMethod -InputObject \$_ -MethodName GetOwner).User \
                       -eq \$env:USERNAME } |
      Select-Object -ExpandProperty ProcessId" 2>/dev/null | tr -d ' \r'
}

stop_client() {
  local pids
  pids=$(client_pids)
  if [ -z "$pids" ]; then
    say "no client to stop"
    return 0
  fi
  for p in $pids; do
    say "stopping our client pid $p"
    powershell.exe -NoProfile -NonInteractive \
      -Command "Stop-Process -Id $p -Force -ErrorAction SilentlyContinue" \
      >/dev/null 2>&1
  done
  # An episode only reaches the corpus when it is complete, so a client stopped
  # mid-episode loses that episode and writes nothing -- the resume skips what
  # finished and re-runs the rest. That is why stopping the client is cheap and
  # stopping the server under it is not.
  sleep 3
}

# Only ever our own server: the image has to be the llama-server.exe that lives
# in this repo's vendor/ directory, and the owner has to be us. Another
# account's llama-server -- or one someone else vendored elsewhere -- does not
# match, so "release the card" can never mean "release someone else's job".
server_pids() {
  powershell.exe -NoProfile -NonInteractive -Command "
    Get-CimInstance Win32_Process |
      Where-Object { \$_.Name -eq 'llama-server.exe' -and
                     \$_.ExecutablePath -like '*CollabEngine-LLM*' } |
      Where-Object { (Invoke-CimMethod -InputObject \$_ -MethodName GetOwner).User \
                       -eq \$env:USERNAME } |
      Select-Object -ExpandProperty ProcessId" 2>/dev/null | tr -d ' \r'
}

stop_server() {
  local pids
  pids=$(server_pids)
  [ -z "$pids" ] && { say "no server of ours to stop"; return 0; }
  for p in $pids; do
    say "stopping our server pid $p"
    powershell.exe -NoProfile -NonInteractive \
      -Command "Stop-Process -Id $p -Force -ErrorAction SilentlyContinue" \
      >/dev/null 2>&1
  done
  sleep 5
}

wait_for_card() {
  local waited=0 f
  while :; do
    f=$(free_mib)
    if [ -n "$f" ] && [ "$f" -ge "$NEED_MIB" ] 2>/dev/null; then
      [ "$waited" -gt 0 ] && say "card free again (${f} MiB) after ${waited}s"
      return 0
    fi
    if [ "$waited" = 0 ]; then
      say "card busy (${f:-?} MiB free, need ${NEED_MIB}) -- waiting, not competing"
    elif [ $((waited % 300)) = 0 ]; then
      say "still waiting for the card (${f:-?} MiB free) after ${waited}s"
    fi
    sleep "$POLL"
    waited=$((waited + POLL))
  done
}

start_server() {
  say "starting server (-c $CTX --parallel $PARALLEL)"
  ( cd "$REPO" && PORT="$PORT" PARALLEL="$PARALLEL" CTX="$CTX" \
      bash scripts/serve.sh --detach ) >>"$LOG" 2>&1
  local waited=0
  while [ "$waited" -lt 300 ]; do
    if healthy; then
      say "server up after ${waited}s"
      # The slot count is the number that silently rejects every long prompt
      # when it is wrong, so it is checked rather than assumed.
      curl -s "http://127.0.0.1:${PORT}/props" 2>/dev/null \
        | head -c 200 >>"$LOG" 2>&1
      return 0
    fi
    sleep 5
    waited=$((waited + 5))
  done
  say "server did NOT come up within 300s -- leaving it alone for a human"
  return 1
}

start_client() {
  say "resuming the run (completed episodes are skipped)"
  ( cd "$REPO" && nohup bash scripts/answer-run.sh >>runs/answer-nohup.log 2>&1 & )
  sleep 5
}

# Print what stop_client would signal, and signal nothing. The selector is the
# one part of this script that can do damage, so it has to be inspectable
# without arming it.
if [ "${1:-}" = "--list-clients" ]; then
  echo "would stop these pids (and nothing else):"
  for p in $(client_pids); do echo "  $p"; done
  exit 0
fi

say "guard up: watching :$PORT every ${POLL}s, ${STRIKES} strikes before acting"
say "it will never signal a process outside $REPO, and never evict another job"

misses=0
while :; do
  if run_finished; then
    say "run reported ALL DONE -- standing down"
    # Releasing is the default because the whole point of finishing is that
    # someone else can start. Every downstream step -- gate_report, figures,
    # the bootstrap and permutation tests -- is numpy on a JSONL file and wants
    # no GPU at all. Holding 21 GB so that a stats script *could* have it would
    # block the other project for nothing. RELEASE_ON_DONE=0 keeps it up for a
    # judge pass, which is the one downstream step that does need the model.
    if [ "${RELEASE_ON_DONE:-1}" = "1" ]; then
      stop_server
      say "card released -- $(free_mib) MiB free"
    else
      say "RELEASE_ON_DONE=0 -- leaving the server up, card stays occupied"
    fi
    exit 0
  fi
  if healthy; then
    misses=0
  else
    misses=$((misses + 1))
    say "health check failed ($misses/$STRIKES)"
    if [ "$misses" -ge "$STRIKES" ]; then
      say "server is gone -- recovering"
      stop_client          # 1. client first, always
      wait_for_card        # 2. wait for room; never take it
      if start_server; then
        start_client
      fi
      misses=0
    fi
  fi
  sleep "$POLL"
done
