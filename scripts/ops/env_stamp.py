"""Record the exact stack a run was produced by, beside the run.

The reproducibility claim this project wants to make is not "here is the code".
It is "here is the code, the weights, the server build and the interpreter that
produced these numbers". Three of those four have gone unrecorded, and the one
time it mattered the project could not tell whether a server was holding the
weights the config named -- which is why `verify_model` exists (LOG 3.12).

A stamp is written into the run directory as `env.json` and is cheap enough to
write on every run. What it captures:

  python + package versions   from the lockfile-installed interpreter
  llama-server build          `--version`, e.g. b10369 (6e62ba538)
  model file + sha256         the actual bytes, not the config's label for them
  GPU + driver                the card is shared, and which card matters
  git commit + dirty flag     the code, including whether it was uncommitted

**The sha256 is the point.** A GGUF filename says the quantisation and nothing
about the conversion: the same `Q4_K_M` name covers uploads that differ in
imatrix, in tokenizer metadata and in tensor layout. The precision ladder in
this study is only a ladder because all three rungs come from one conversion
(`scripts/ops/fetch_models.py`), and a hash is the only thing that proves it
afterwards.

    python scripts/ops/env_stamp.py --config configs/llamacpp/medium-h3b.yaml
    python scripts/ops/env_stamp.py --out - # print instead of writing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

LLAMA_BIN = Path("vendor/llamacpp/llama-server.exe")


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    text = (out.stdout + out.stderr).strip()
    return text or None


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    """Full-file hash. Slow on a 16 GiB f16 GGUF and worth it once per run.

    Deliberately not a partial hash of the first N bytes: GGUF puts metadata at
    the front, so two different conversions of the same model can agree there
    and disagree in every tensor after it.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def packages() -> dict[str, str]:
    from importlib import metadata

    wanted = (
        "numpy", "pyyaml", "httpx", "pandas", "scipy", "statsmodels",
        "matplotlib", "python-docx", "transformers", "huggingface-hub",
        "pytest",
    )
    out = {}
    for name in wanted:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def stamp(config_path: str | None, hash_models: bool = True) -> dict:
    git_commit = _run(["git", "rev-parse", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])

    out: dict = {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages(),
        "git": {
            "commit": git_commit,
            # A stamp that hides uncommitted changes is worse than no stamp,
            # because it looks authoritative.
            "dirty": bool(dirty),
        },
        "llama_server": _run([str(LLAMA_BIN), "--version"]) if LLAMA_BIN.exists() else None,
        "gpu": _run([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]),
    }

    if config_path:
        from collabengine.config import ExperimentConfig

        cfg = ExperimentConfig.load(config_path)
        model = getattr(cfg.backend, "model", None)
        out["config"] = {"path": config_path, "model": model, "name": cfg.name}
        if hash_models and model:
            path = Path(model)
            if path.exists():
                out["model_sha256"] = {path.name: sha256(path)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--out", default=None,
                    help="path to write, '-' for stdout; default is the run dir")
    ap.add_argument("--no-hash", action="store_true",
                    help="skip the model hash (it reads the whole file)")
    args = ap.parse_args()

    data = stamp(args.config, hash_models=not args.no_hash)
    text = json.dumps(data, indent=2, sort_keys=True)

    dest = args.out
    if dest is None and args.config:
        from collabengine.config import ExperimentConfig

        run_dir = ExperimentConfig.load(args.config).run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        dest = str(run_dir / "env.json")

    if dest in (None, "-"):
        print(text)
    else:
        Path(dest).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
