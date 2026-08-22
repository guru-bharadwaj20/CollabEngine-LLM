# Technical details

**Hardware, software, data and compute for CollabEngine-LLM.** Every figure was
read off the machine or out of a run log on 2026-08-22; nothing here is recalled.
Where a number cannot be recovered, it says so rather than estimating.

---

## 1. Hardware

| | |
|---|---|
| GPU | **NVIDIA RTX 4500 Ada Generation**, 24,570 MiB (24 GB) |
| Driver | 595.95 |
| CUDA (driver-reported) | 13.2 |
| CPU | Intel Core i9-14900 — 24 cores, 32 threads |
| System RAM | 127.7 GB |
| OS | Windows 11 Pro, build 10.0.26200 |

**The card is shared and that is a methodological fact, not a footnote.** Other
accounts — including administrator accounts — run jobs on the same device, so no
lock is possible. Every queue script in `scripts/ops/` therefore *waits* for free
VRAM rather than reclaiming it, and gates on a stable reading across three
consecutive checks before starting.

**One Windows-specific hazard governs the slot arithmetic.** Windows WDDM pages
GPU memory to host RAM over PCIe instead of raising OOM. A job that does not
quite fit therefore does not fail — it runs at bus speed while `nvidia-smi` still
reports 90–100% utilisation. The tell is power draw: ~120–150 W when healthy
against ~57 W when paging. This is why the f16 and 14B presets are deliberately
configured one slot short of what would nominally fit (RESEARCH-LOG §3.4).

---

## 2. Software

### Serving

| | |
|---|---|
| Engine | **llama.cpp `b10369` (`6e62ba538`)** |
| Build | `llama-b10369-bin-win-cuda-12.4-x64.zip`, official release asset |
| Compiler | Clang 20.1.8, Windows x86_64 |
| CUDA build target | 12.4 |
| Serving mode | OpenAI-compatible HTTP on `127.0.0.1:8000`, continuous batching |
| Slot geometry | **18,432 tokens per slot in every arm**, slot count varies by model |

Per-slot context is the one quantity held fixed across every arm in the study,
because it is the cap whose asymmetry the whole project is about. Slot *count*
varies with weight size and KV cost; that changes throughput and nothing else.

### Analysis

| | |
|---|---|
| Python | **3.10.20** (conda environment `collabengine`) |
| numpy | 2.2.6 |
| scipy | 1.15.3 |
| matplotlib | 3.10.9 |
| PyYAML | 6.0.3 |
| python-docx | 1.2.0 |
| pytest | 9.1.1 |

Full pin: `requirements.lock.txt` (52 packages). `Dockerfile` reproduces the
**analysis** path exactly and the **generation** path only approximately — the
difference is documented rather than papered over, because continuous batching
is not bit-reproducible. `scripts/ops/env_stamp.py` writes four layers into
`env.json` per run: Python packages, llama.cpp build, model file + sha256, and
git commit with a dirty flag.

**Determinism, stated honestly.** Scoring and analysis are deterministic and
reproduce exactly. *Generation* does not: sampling runs at temperature 0.8 and
llama.cpp's continuous batching does not guarantee bit-identical output across
runs. Regenerating a corpus from the same seeds is therefore a **re-measurement**,
not a reproduction, and the project labels it that way.

---

## 3. Models

All served locally as GGUF. No hosted API model was used for any result in the
paper.

| Model | Quantisation | Size | Slots | Status |
|---|---|---|---|---|
| Meta-Llama-3.1-8B-Instruct | Q4_K_M | 4.92 GB | 4 | **primary instrument** |
| Meta-Llama-3.1-8B-Instruct | Q8_0 | 8.54 GB | 4 | measured (450 episodes) |
| Meta-Llama-3.1-8B-Instruct | f16 | 16.07 GB | 2 | downloaded, **cut before running** |
| Qwen2.5-7B-Instruct | Q4_K_M | 4.68 GB | 4 | measured (450 episodes) |
| Mistral-7B-Instruct-v0.3 | Q4_K_M | 4.37 GB | 4 | downloaded, **unusable** — see below |
| Qwen2.5-14B-Instruct | Q4_K_M (3 shards) | 8.99 GB | 3 | downloaded, **cut before running** |

Total weights on disk: **47.5 GB**. Every file carries a recorded sha256 in
`docs/ENVIRONMENT.md`; all Llama rungs come from a single uploader so the
precision ladder is one conversion rather than three.

**Mistral could not be run at all.** Its chat template raises
`Conversation roles must alternate user/assistant/...` and supports no `system`
role, while the orchestrator sends a system message and, in the team arm,
consecutive same-role turns. Running it would have required different prompt
assembly for one family, confounding model family with prompt construction. The
arm was withdrawn rather than repaired (`docs/PREREG-families.md`).

**Judge models.** A Qwen2.5-14B pass over the 40-message hand-coded validation
set was used once, to test whether the κ = 0.29 behavioural judge was
capacity-limited. It is not (RESEARCH-LOG §4.25).

---

## 4. Data

**There is no external dataset. Every task instance is generated from a seed.**

| Family | Module | What it is |
|---|---|---|
| Scheduling | `src/collabengine/tasks/` | Multi-constraint allocation instances, graded by constraint satisfaction |
| Code generation | `src/collabengine/tasks/code/` | Python implementation tasks with a hidden test suite, graded in a sandbox |

Both are procedural generators with four independently scored components. Nothing
is downloaded, nothing is scraped, and there is **no contamination risk from
pretraining**, because no instance existed before it was generated. That is a
deliberate design choice and one of the study's stronger properties: a public
benchmark at 7–8B scale cannot rule out memorisation, and this can.

**Seed ranges.** `0–47` is the hypothesis-generating pilot. `1000–1149` is the
confirmatory fresh-seed range, disjoint by construction. The fresh-seed rule
overturned the project's own headline result and is not optional.

**Human-labelled data.** One 40-message behavioural-coding validation set,
hand-coded by the author, used only for inter-rater agreement (κ) against LLM
judges. No personal data, no human subjects, no external annotators.

**Corpora produced** (`runs/`, gitignored, ~11 directories):

| Corpus | Episodes |
|---|---|
| `llama31-8b-q4-medium-h3b` (fresh-seed headline) | 1,050 |
| `llama31-8b-q4-code-medium` (second task family) | 1,050 |
| `llama31-8b-q4-cap{0512,1024,2048,3072,6144}` (dose-response) | 1,000 |
| `qwen25-7b-q4-medium` | 450 |
| `llama31-8b-q8-medium` | 450 |
| `llama31-8b-q4-medium-ans` (pilot, restored) | 336 |
| `llama31-8b-q4-{medium,hard,xhard}{,-ans}` (figure corpora) | 360 |
| **Total** | **~4,700** |

---

## 5. GPU hours

**Measured from run-log timestamps for 2026-08-21/22.** These are durations of
this project's own jobs, not wall-clock on a shared card.

| Stage | Duration |
|---|---|
| Fresh-seed corpus rebuild (450 + 600 episodes) | 6.09 h |
| Pilot corpus restoration (336 episodes) | 1.93 h |
| Figure corpora, 5 tiers (360 episodes) | 3.27 h |
| **Dose-response cap sweep (1,000 episodes)** | **5.42 h** |
| **Code task family (1,050 episodes)** | **3.15 h** |
| Qwen2.5-7B gate (450 episodes) | 3.53 h |
| Llama Q8_0 gate (450 episodes) | 4.70 h |
| 14B judge validation | ~0.03 h |
| **Measured subtotal** | **≈ 28.1 h** |

| Also spent, not in the table above | |
|---|---|
| Failed corpus rebuild, 450/450 errored against a mid-refactor tree | ~3.2 h |
| Smoke tests, preflights, calibration probes | not separately logged |

**Total recoverable for this period: ≈ 31.3 GPU-hours.**

### What is *not* recoverable, and why

**The project's lifetime GPU consumption cannot be reconstructed.** `runs/` is
gitignored and was cleaned before 2026-08-21, taking every prior run log with it.
The research log records the *findings* of those runs across §4.1–§4.24 —
multiple corpora at three difficulty tiers, two serving instruments, several
withdrawn analyses, and a behavioural-coding phase — but not their durations.

An order-of-magnitude statement that is defensible: the surviving work represents
**at least 30 GPU-hours**, and the full project including the lost corpora and
the abandoned bf16/`transformers` instrument is **plausibly 3–5× that**. That
figure is an estimate and is labelled as one. Do not quote it as measured.

### Deliberately not spent

| Arm | Estimated cost | Why not |
|---|---|---|
| Llama f16 gate | ~8 h | No reviewer has raised full-precision; the paper scopes itself to 4-bit |
| Qwen2.5-14B grid | ~15 h | Answers a real objection *confoundedly* — it moves scale and model family in one step |

Both were cut on 2026-08-22 when five corpora sat unscored and analysis, not
generation, had become the binding constraint. Configs, weights and
preregistrations are intact; both resume from disk.

---

## 6. Test and reproduction surface

| | |
|---|---|
| Test suite | **494 tests**, 1 skipped, ~41 s, **no GPU required** |
| Fresh-clone verification | unauthenticated clone into a scratch directory, `PYTHONPATH` forced to the clone's own `src` |
| CI-sized fixture | 40-episode slice + golden file, so the analysis path is testable without a 4 GB download |
| Build-time lints | citations, register split, p-value/effect-size pairing, anonymity |

**The reproduction gap that remains.** The paper is assembled from **eleven**
corpora. None are released — `runs/` is gitignored and no archive exists. The
analysis path reproduces from a committed fixture; the *numbers in the paper* do
not reproduce without the corpora. This is tracked as Final Sweep §1.0.a and is
the one outstanding item that is also a correctness issue, because the draft
currently claims the transcripts are released.
