"""Fetch every GGUF this study serves, into models/.

Supersedes `fetch_14b.py`, which fetched one of them.

**One conversion provenance per comparison.** The precision ladder (Q4_K_M,
Q8_0, f16) is taken entirely from `mradermacher/Meta-Llama-3.1-8B-Instruct-GGUF`
so that the only thing varying across those three arms is the quantisation. A
ladder assembled from three uploaders would confound precision with whoever ran
`convert_hf_to_gguf.py` and with which imatrix, if any — the same class of
mistake as measuring one tier on a different instrument (LOG 4.11).

For the same reason the Q4_K_M point of the ladder *is* the study's primary
instrument, rather than a second Q4 from elsewhere that happens to agree.

    python scripts/ops/fetch_models.py               # everything
    python scripts/ops/fetch_models.py --only llama-q4 qwen-7b
    python scripts/ops/fetch_models.py --list

Sizes are the download, not the resident footprint; see docs/LLAMACPP-SETUP.md
for what each one costs on the card once loaded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# key -> (repo_id, [filenames], approx GB, what it is for)
MODELS: dict[str, tuple[str, list[str], float, str]] = {
    "llama-q4": (
        "mradermacher/Meta-Llama-3.1-8B-Instruct-GGUF",
        ["Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf"],
        4.9,
        "primary instrument, and the low rung of the precision ladder",
    ),
    "llama-q8": (
        "mradermacher/Meta-Llama-3.1-8B-Instruct-GGUF",
        ["Meta-Llama-3.1-8B-Instruct.Q8_0.gguf"],
        8.5,
        "precision ladder, middle rung",
    ),
    "llama-f16": (
        "mradermacher/Meta-Llama-3.1-8B-Instruct-GGUF",
        ["Meta-Llama-3.1-8B-Instruct.f16.gguf"],
        16.1,
        "precision ladder, top rung -- unquantised reference",
    ),
    "qwen-7b": (
        "bartowski/Qwen2.5-7B-Instruct-GGUF",
        ["Qwen2.5-7B-Instruct-Q4_K_M.gguf"],
        4.7,
        "second model family, matched scale and quantisation",
    ),
    "mistral-7b": (
        "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        ["Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"],
        4.4,
        "third model family, matched scale and quantisation",
    ),
    "qwen-14b": (
        "Qwen/Qwen2.5-14B-Instruct-GGUF",
        [f"qwen2.5-14b-instruct-q4_k_m-0000{i}-of-00003.gguf" for i in (1, 2, 3)],
        8.9,
        "the scale step-up PLAN section 6 reserves for '7-8B too weak'",
    ),
}

OUT = Path("models")


def fetch(key: str) -> list[Path]:
    from huggingface_hub import hf_hub_download

    repo, files, gb, purpose = MODELS[key]
    print(f"\n=== {key}  ({gb:.1f} GB)  {purpose}", flush=True)
    print(f"    {repo}", flush=True)
    got = []
    for f in files:
        dest = OUT / f
        if dest.exists():
            print(f"    have {f} ({dest.stat().st_size / 2**30:.2f} GiB)", flush=True)
            got.append(dest)
            continue
        print(f"    get  {f}", flush=True)
        p = hf_hub_download(repo_id=repo, filename=f, local_dir=str(OUT))
        got.append(Path(p))
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(MODELS), default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        total = 0.0
        for k, (repo, files, gb, purpose) in MODELS.items():
            print(f"{k:<12} {gb:>5.1f} GB  {repo}\n{'':<12} {purpose}")
            total += gb
        print(f"\n{'total':<12} {total:>5.1f} GB")
        return 0

    OUT.mkdir(exist_ok=True)
    keys = args.only or list(MODELS)
    for k in keys:
        fetch(k)
    print("\nALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
