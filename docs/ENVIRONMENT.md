# The pinned stack

Every number in this project is produced by four things, and until 2026-08-20
only one of them was recorded anywhere.

| layer | pinned by | why it matters here |
|---|---|---|
| Python + packages | `requirements.lock.txt`, 52 packages | `scipy` and `statsmodels` versions decide the *p*-values |
| The serving build | `b10369 (6e62ba538)`, CUDA 12.4, Windows x64 | the slot geometry and the batching arrangement live here |
| The weights | sha256, below | a `Q4_K_M` filename names a quantisation, not a conversion |
| The code | git commit, recorded per run | including whether the tree was dirty |

`scripts/ops/env_stamp.py` writes all four into `env.json` in the run directory.
Run it beside every corpus; it costs one full-file hash.

---

## Python

```bash
conda create -n collabengine python=3.10.20
pip install -r requirements.lock.txt
pip install -e ".[dev,analysis]" --no-deps
pytest -q          # 439 tests, seconds, no GPU
```

`requirements.lock.txt` is a full `pip freeze` of the environment that produced
the current corpus, not a curated subset. `pyproject.toml` keeps its `>=`
floors, which is the right constraint for a library and the wrong one for a
result — the lockfile is what a reproduction should install.

A container is in `Dockerfile`. It reproduces the *analysis* exactly and the
*generation* approximately, and the file says why in its own header rather than
implying otherwise.

## The serving build

```
vendor/llamacpp/llama-server.exe --version
version: 10369 (6e62ba538)
built with Clang 20.1.8 for Windows x86_64
```

Fetched from `ggml-org/llama.cpp` release `b10369`, asset
`llama-b10369-bin-win-cuda-12.4-x64.zip`, plus
`cudart-llama-bin-win-cuda-12.4-x64.zip` for the CUDA runtime. This is the same
build the corpora in RESEARCH-LOG §4.9 onward were generated on; it reports
`n_slots = 4, n_ctx_slot = 18432` on startup, which is the arithmetic in
[LLAMACPP-SETUP.md](LLAMACPP-SETUP.md) confirmed by the server rather than by us.

## The weights, and one deliberate change

**Every Llama GGUF now comes from `mradermacher/Meta-Llama-3.1-8B-Instruct-GGUF`.**
Previously the Q4_K_M instrument came from one uploader and no other precision
existed. The precision ladder (Final Sweep 1.4) is only a ladder if its three
rungs share a conversion, so all three are taken from that one repository and
the Q4_K_M rung *is* the study's primary instrument rather than a second Q4 that
happens to agree.

The cost is that the primary instrument's weights are a different upload from
the one behind the pre-2026-08-20 corpora. Those corpora no longer exist, so
nothing is being invalidated that still stood — but any number quoted from the
old log is from a conversion that cannot now be checked, and that is stated
where the numbers are.

| sha256 (first 16) | file | role |
|---|---|---|
| `56e1a31ac6e50371` | `Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf` | primary instrument; ladder low rung |
| `258c49555c709d90` | `Meta-Llama-3.1-8B-Instruct.Q8_0.gguf` | ladder middle rung |
| `572cf644f099d0d4` | `Meta-Llama-3.1-8B-Instruct.f16.gguf` | ladder top rung, unquantised |
| `65b8fcd92af6b4fe` | `Qwen2.5-7B-Instruct-Q4_K_M.gguf` | second model family |
| `1270d22c0fbb3d09` | `Mistral-7B-Instruct-v0.3-Q4_K_M.gguf` | third model family |
| `a09ea5e7b1eafb1b` | `qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf` | scale step-up, shard 1 |
| `21b9457d079680d2` | `qwen2.5-14b-instruct-q4_k_m-00002-of-00003.gguf` | scale step-up, shard 2 |
| `c8d37006760a387a` | `qwen2.5-14b-instruct-q4_k_m-00003-of-00003.gguf` | scale step-up, shard 3 |

Fetch them all with `python scripts/ops/fetch_models.py`, which pins repository
and filename for each. 44.3 GiB total.

## What is still not pinned, and is named rather than hidden

- **The GPU driver.** 595.95 here. A driver change can move llama.cpp's kernel
  selection and therefore the arithmetic; it is recorded in `env.json` and not
  controlled.
- **The host OS.** Windows 11. The Docker image is Linux, so a container run is
  a different generation stack even at identical weights and build.
- **Batching arrangement.** Continuous batching means what else is resident in
  a batch can change a sampled token. This is why reproducibility here rests on
  instances being deterministic in `(seed, difficulty)` and on transcripts being
  re-scorable offline, rather than on regenerating identical text.
