# Resolve a real Python, and refuse the one that only looks like one.
#
# Source this, do not execute it:
#
#     . scripts/ops/_python.sh    # sets PY, from the repository root
#
# **The incident.** `PY=${PY:-python}` is correct in an interactive shell here
# and wrong under `nohup`. Windows ships an App Execution Alias at
# `%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe` that is not an interpreter:
# it prints "Python was not found; run without arguments to install from the
# Microsoft Store" and exits. An interactive shell has the conda environment
# ahead of it on PATH, a detached one started from a different parent does not,
# and the queue scripts are all launched detached by design.
#
# So the whole GPU queue came up, gated on the card correctly, stopped the
# server correctly, started it at the right geometry, and then failed every
# preflight it attempted with a Microsoft Store advertisement. Nothing was
# generated and no card time was spent on episodes, but the failure looked like
# a preflight problem and it was not.
#
# This is the same shape as the CRLF bug in `queue-judge.sh`: a guard or a tool
# that fails in a way that reads as something else. The stub does not fail like
# a missing interpreter -- it exits cleanly enough to look like the script's own
# fault.
#
# **What this does.** Takes the first candidate that is a real interpreter, and
# tests that by asking it to print its version rather than by testing that the
# file exists. The stub exists. Order matters: an explicit `PY` from the
# environment wins so a caller can override, then the project's own conda
# environment, then whatever `python`/`python3` resolve to, which is correct on
# a machine where PATH is sane.
#
# It is deliberately loud when it finds nothing. A queue that runs for a week
# must not start at all rather than discover on stage six that it cannot import
# the package it was measuring.

_is_python() {
  [ -n "${1:-}" ] || return 1
  "$1" -c 'import sys; sys.exit(0)' >/dev/null 2>&1
}

_resolve_python() {
  local c
  for c in \
    "${PY:-}" \
    "${COLLABENGINE_PYTHON:-}" \
    "C:/Users/Temp/miniconda3/envs/collabengine/python.exe" \
    "$HOME/miniconda3/envs/collabengine/python.exe" \
    "$(command -v python 2>/dev/null || true)" \
    "$(command -v python3 2>/dev/null || true)"
  do
    if _is_python "$c"; then
      printf '%s\n' "$c"
      return 0
    fi
  done
  return 1
}

if ! PY=$(_resolve_python); then
  echo "no working Python found." >&2
  echo "  The Microsoft Store stub at WindowsApps/python.exe is not one: it" >&2
  echo "  prints an advertisement and exits, which reads downstream as the" >&2
  echo "  calling script's own failure." >&2
  echo "  Set COLLABENGINE_PYTHON to the interpreter that has collabengine" >&2
  echo "  installed, or activate the conda environment before launching." >&2
  exit 127
fi
export PY
