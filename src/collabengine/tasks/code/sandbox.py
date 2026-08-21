"""Running untrusted, model-written code.

Nothing here ever executes a submission in the harness process. The grader
writes the submission and its test suite into a scratch directory, runs a
standalone harness script there in a fresh interpreter, and reads results back
out of a file. The subprocess is disposable; that, and not any of the guards
below, is the actual boundary.

What the guards are for, in the order the failure modes actually occur:

  infinite loops   the parent kills the child on a wall-clock deadline, and the
                   child flushes each result as it goes, so the tests that
                   finished before the hang still score
  stray imports    the submission runs with a builtins mapping whose `__import__`
                   accepts an allowlist -- `os`, `socket`, `subprocess` and
                   everything else outside it raise ImportError
  filesystem       `open` is removed from that mapping, which also stops a
                   submission overwriting the results file it is being scored by
  memory           RLIMIT_AS where the platform has it; Windows has no cheap
                   equivalent, so there the deadline is the only guard, which is
                   why the parent kills rather than waits

**This is a research harness, not a security sandbox.** A submission determined
to escape can still walk `object.__subclasses__()` out of the restricted
builtins. The threat model is a small model writing `import os` or `while True`,
not an adversary, and saying so plainly is better than implying a boundary that
is not there.

Determinism is a requirement rather than a nicety, because the corpus has to be
rescorable offline months later: the child gets a scrubbed environment, a seeded
`random`, and no network, and the per-test deadline is set two orders of
magnitude above what a correct submission needs so that it fires only on
pathology rather than on a busy machine.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Modules a submission may import. Allowlist rather than denylist: a denylist
#: is a list of the escapes someone already thought of.
ALLOWED_IMPORTS: tuple[str, ...] = (
    "array",
    "bisect",
    "cmath",
    "collections",
    "copy",
    "dataclasses",
    "decimal",
    "enum",
    "fractions",
    "functools",
    "heapq",
    "itertools",
    "json",
    "math",
    "numbers",
    "operator",
    "random",
    "re",
    "statistics",
    "string",
    "textwrap",
    "types",
    "typing",
)

#: Per test, in seconds. A correct submission on the largest preset runs the
#: whole suite in single-digit milliseconds, so this is pure headroom.
DEFAULT_PER_TEST_S = 2.0

#: Interpreter startup plus module exec, before any test runs.
DEFAULT_STARTUP_S = 10.0

#: Address-space ceiling in the child, where the platform supports one.
DEFAULT_MEMORY_MB = 512


@dataclass(frozen=True, slots=True)
class SandboxResult:
    parsed: bool
    """The submission compiled. False means a `syntax` score of zero."""
    loaded: bool
    """Module-level execution finished without raising."""
    defined: dict[str, bool]
    """Required function name -> defined and callable in the module namespace."""
    passed: dict[str, bool]
    """Test id -> passed. A test the child never reached is absent, not False;
    the grader treats absence as failure and records why."""
    timed_out: bool
    error: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


def run_submission(
    code: str,
    required: tuple[str, ...],
    tests: list[dict[str, Any]],
    *,
    per_test_s: float = DEFAULT_PER_TEST_S,
    startup_s: float = DEFAULT_STARTUP_S,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> SandboxResult:
    """Execute one submission against one test suite in a disposable child."""
    workdir = Path(tempfile.mkdtemp(prefix="collabengine-code-"))
    try:
        harness = workdir / "_harness.py"
        payload_path = workdir / "_payload.json"
        results_path = workdir / "_results.jsonl"

        harness.write_text(_HARNESS_SOURCE, encoding="utf-8")
        payload_path.write_text(
            json.dumps(
                {
                    "code": code,
                    "required": list(required),
                    "tests": tests,
                    "per_test_s": per_test_s,
                    "memory_mb": memory_mb,
                    "allowed_imports": list(ALLOWED_IMPORTS),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        deadline = startup_s + per_test_s * max(1, len(tests))
        timed_out = False
        try:
            subprocess.run(
                [
                    sys.executable,
                    # -I isolates: no PYTHONPATH, no user site, no cwd on the
                    # path beyond the scratch directory the harness sits in.
                    "-I",
                    # No __pycache__ writes into a directory we are about to
                    # delete, and no cache reuse between two different
                    # submissions that happen to land on the same temp name.
                    "-B",
                    str(harness),
                    str(payload_path),
                    str(results_path),
                ],
                cwd=workdir,
                env=_scrubbed_env(),
                stdin=subprocess.DEVNULL,
                # The submission's own prints are noise and can be unbounded.
                # The protocol goes through the results file instead.
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=deadline,
                check=False,
            )
        except subprocess.TimeoutExpired:
            timed_out = True

        return _read_results(results_path, required, tests, timed_out, deadline)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _scrubbed_env() -> dict[str, str]:
    """The minimum a Python interpreter needs to start, and nothing else.

    Passing the parent's environment through would hand a submission the API
    keys the serving layer runs on. Windows additionally cannot start a process
    without SystemRoot, which is why that one is not optional.
    """
    keep = ("SYSTEMROOT", "SystemRoot", "WINDIR", "PATH", "TEMP", "TMP", "TMPDIR")
    return {k: os.environ[k] for k in keep if k in os.environ}


def _read_results(
    path: Path,
    required: tuple[str, ...],
    tests: list[dict[str, Any]],
    timed_out: bool,
    deadline: float,
) -> SandboxResult:
    parsed = False
    loaded = False
    defined = {name: False for name in required}
    passed: dict[str, bool] = {}
    first_error = ""

    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # The last line of a killed child can be half written.
                continue
            event = row.get("event")
            if event == "parse":
                parsed = bool(row.get("ok"))
                first_error = first_error or str(row.get("error") or "")
            elif event == "load":
                loaded = bool(row.get("ok"))
                first_error = first_error or str(row.get("error") or "")
            elif event == "defined":
                defined[str(row.get("name"))] = bool(row.get("ok"))
            elif event == "test":
                passed[str(row.get("tid"))] = bool(row.get("ok"))
                if not row.get("ok"):
                    first_error = first_error or str(row.get("error") or "")

    unreached = [t["tid"] for t in tests if t["tid"] not in passed]
    return SandboxResult(
        parsed=parsed,
        loaded=loaded,
        defined=defined,
        passed=passed,
        timed_out=timed_out,
        error=first_error[:200],
        detail={
            "tests_reached": len(passed),
            "tests_unreached": len(unreached),
            "deadline_s": deadline,
        },
    )


#: Written to the scratch directory and run there. Kept as a string rather than
#: a module of its own so that nothing in `collabengine` is importable from the
#: child: the harness has to work from a bare interpreter and an isolated path,
#: and a child that could import the package could import the corpus writer too.
_HARNESS_SOURCE = r'''"""Score one submission. Written to a scratch directory; never imported."""
import builtins
import json
import random
import sys
import time

with open(sys.argv[1], encoding="utf-8") as fh:
    payload = json.load(fh)

# Line buffered and flushed per row: when the parent kills this process for
# running long, whatever finished before the hang is still on disk and still
# scores.
results = open(sys.argv[2], "w", encoding="utf-8", buffering=1)


def emit(**row):
    results.write(json.dumps(row) + "\n")
    results.flush()


try:
    import resource

    cap = int(payload["memory_mb"]) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
except Exception:
    # No RLIMIT_AS on Windows. The parent's wall-clock kill is the guard there.
    pass

ALLOWED = set(payload["allowed_imports"])
_real_import = builtins.__import__


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in ALLOWED:
        raise ImportError("import of %r is not permitted in this sandbox" % root)
    return _real_import(name, globals, locals, fromlist, level)


safe = {k: getattr(builtins, k) for k in dir(builtins) if not k.startswith("_")}
for banned in ("open", "eval", "exec", "compile", "input", "breakpoint", "help"):
    safe.pop(banned, None)
safe["__import__"] = guarded_import
# Without this a `class` statement in the submission raises NameError, which
# would score a legitimate submission as broken rather than as sandboxed.
safe["__build_class__"] = builtins.__build_class__

ns = {"__name__": "submission", "__builtins__": safe}

# A submission that samples must score the same twice, or the corpus stops being
# rescorable.
random.seed(0)

try:
    compiled = compile(payload["code"], "<submission>", "exec")
except BaseException as exc:
    emit(event="parse", ok=False, error="%s: %s" % (type(exc).__name__, exc))
    results.close()
    sys.exit(0)
emit(event="parse", ok=True)

load_error = ""
try:
    exec(compiled, ns)
except BaseException as exc:
    load_error = "%s: %s" % (type(exc).__name__, exc)
emit(event="load", ok=not load_error, error=load_error)

for name in payload["required"]:
    emit(event="defined", name=name, ok=callable(ns.get(name)))

limit = float(payload["per_test_s"])
for case in payload["tests"]:
    fn = ns.get(case["function"])
    if not callable(fn):
        emit(event="test", tid=case["tid"], ok=False, error="function is not defined")
        continue
    started = time.perf_counter()
    try:
        got = fn(list(case["args"]))
        # `True == 1` in Python, and a submission returning a bool where an int
        # was specified has not implemented the specification.
        ok = (not isinstance(got, bool)) and got == case["expected"]
        err = "" if ok else "returned %r" % (got,)
    except BaseException as exc:
        ok, err = False, "%s: %s" % (type(exc).__name__, exc)
    elapsed = time.perf_counter() - started
    if elapsed > limit:
        ok, err = False, "exceeded the %gs per-test limit" % limit
    emit(event="test", tid=case["tid"], ok=ok, error=err)

results.close()
'''
