"""Pack a corpus for release, and fetch a released one back.

The paper says the transcripts are released. Until 2026-08-21 they were not:
`runs/` is gitignored, the archive did not exist, and the repository was
private. `scripts/analysis/build_paper.py` reads its numbers from the corpus, so
without it the paper does not build at all -- which is the correct failure and
the reason this file exists rather than a note in the README.

    python scripts/ops/fetch_corpus.py --pack runs/llama31-8b-q4-medium-h3b
    python scripts/ops/fetch_corpus.py                 # fetch and unpack
    python scripts/ops/fetch_corpus.py --list          # what a release holds

**The manifest is the point, not the tarball.** A corpus without one is a pile
of JSONL that cannot be traced back to the configuration that produced it, and
this project has already lost three corpora to instrument settings nobody could
read back afterwards. `--pack` records, per run directory: the resolved config,
the environment stamp, per-condition episode counts, the seed range, and a
sha256 of every file. A fetched corpus is verified against it and refuses to
unpack over a mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

RUNS = Path("runs")
MANIFEST = "MANIFEST.json"

#: Where a packed corpus is published. A release asset rather than git-lfs:
#: transcripts are large, immutable once measured, and nobody needs their
#: history -- the seeds are the history.
REPO = "guru-bharadwaj20/CollabEngine-LLM"
TAG = os.environ.get("CORPUS_TAG", "corpus-v1")
ASSET = "collabengine-corpus.tar.gz"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def describe(run_dir: Path) -> dict:
    """Per-condition counts and the seed range, read from the episodes."""
    from collabengine.transcripts.store import TranscriptReader

    conditions: Counter[str] = Counter()
    seeds: list[int] = []
    for path in sorted(run_dir.glob("*.jsonl")):
        for rec in TranscriptReader(str(path)):
            conditions[rec.condition] += 1
            seeds.append(rec.instance_seed)
    return {
        "episodes": sum(conditions.values()),
        "conditions": dict(sorted(conditions.items())),
        "seed_min": min(seeds) if seeds else None,
        "seed_max": max(seeds) if seeds else None,
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(run_dir.glob("*"))
            if p.is_file()
        },
    }


def pack(run_dirs: list[Path], out: Path) -> None:
    manifest = {"runs": {}}
    for d in run_dirs:
        if not d.is_dir():
            print(f"  skip {d} (not a directory)", file=sys.stderr)
            continue
        print(f"  describing {d}", flush=True)
        manifest["runs"][d.name] = describe(d)

    RUNS.mkdir(exist_ok=True)
    (RUNS / MANIFEST).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with tarfile.open(out, "w:gz") as tar:
        tar.add(RUNS / MANIFEST, arcname=f"runs/{MANIFEST}")
        for d in run_dirs:
            if d.is_dir():
                tar.add(d, arcname=f"runs/{d.name}")
    total = sum(r["episodes"] for r in manifest["runs"].values())
    print(f"\npacked {total} episodes from {len(manifest['runs'])} run(s) "
          f"-> {out} ({out.stat().st_size / 2**20:.1f} MiB)")
    print(f"manifest at {RUNS / MANIFEST}")


def verify() -> int:
    """Check an unpacked corpus against its manifest. Returns problems found."""
    path = RUNS / MANIFEST
    if not path.exists():
        print(f"no manifest at {path}", file=sys.stderr)
        return 1
    manifest = json.loads(path.read_text(encoding="utf-8"))
    problems = 0
    for name, info in manifest["runs"].items():
        for fname, meta in info["files"].items():
            f = RUNS / name / fname
            if not f.exists():
                print(f"  MISSING {f}")
                problems += 1
            elif sha256(f) != meta["sha256"]:
                print(f"  CHANGED {f}")
                problems += 1
        print(f"  {name}: {info['episodes']} episodes, "
              f"seeds {info['seed_min']}-{info['seed_max']}, "
              f"{len(info['conditions'])} conditions")
    print("corpus verified" if not problems else f"{problems} problem(s)")
    return problems


def fetch(url: str, dest: Path) -> None:
    print(f"fetching {url}", flush=True)
    with urllib.request.urlopen(url) as r, dest.open("wb") as fh:
        while block := r.read(1 << 20):
            fh.write(block)
    print(f"  {dest.stat().st_size / 2**20:.1f} MiB")
    with tarfile.open(dest) as tar:
        tar.extractall(".")
    print("unpacked into runs/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", nargs="*", metavar="RUN_DIR",
                    help="pack these run directories (default: all of runs/)")
    ap.add_argument("--out", default="collabengine-corpus.tar.gz")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--url", default=None, help="override the release URL")
    args = ap.parse_args()

    if args.verify:
        return 1 if verify() else 0

    if args.pack is not None:
        dirs = [Path(p) for p in args.pack] or [
            d for d in sorted(RUNS.iterdir()) if d.is_dir()
        ]
        pack(dirs, Path(args.out))
        return 0

    url = args.url or f"https://github.com/{REPO}/releases/download/{TAG}/{ASSET}"
    dest = Path(args.out)
    try:
        fetch(url, dest)
    except Exception as exc:
        print(f"could not fetch the corpus: {exc}", file=sys.stderr)
        print("\nThe corpus is a release asset, not a git object. If this is a "
              "fresh clone and\nthe release does not exist yet, regenerate it "
              "instead:\n"
              "    scripts/ops/serve.sh --detach\n"
              "    bash scripts/experiments/rebuild-corpus.sh\n"
              "Note that regenerating re-measures rather than reproduces -- see "
              "that script's header.", file=sys.stderr)
        return 2
    return 1 if verify() else 0


if __name__ == "__main__":
    sys.exit(main())
