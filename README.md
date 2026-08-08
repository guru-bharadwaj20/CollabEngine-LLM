# CollabEngine-LLM

A local framework for spinning up LLM agent teams with **no assigned roles** and causally verifying whether the division of labor they develop is functionally real.

Several instances of the *same* open model are given a shared task and natural-language communication. Nobody is told to plan, criticize, or verify. If roles appear, the question is whether they are functionally real or a pattern being read into transcripts — and that question is settled by ablation, not by reading.

See [PLAN.md](PLAN.md) for the research design, prior-art assessment, and phase breakdown.

---

## The claim being tested

> Emergent role differentiation in same-model LLM agent teams is behaviorally observable and statistically stable — but its transcript-derived labels predict causal contribution only weakly.

Either direction of that result is publishable, which is the mark of a well-posed experiment.

## Two things that make this different from prior work

**Specialization is a claim about *differential* contribution, not total contribution.** A scalar performance drop when you remove an agent proves *participation* — remove any competent contributor from any team and output falls. The result is an **agent × task-component interaction**: ablating the emergent "critic" must damage criticism-loaded components more than planning-loaded ones. The analysis double-centers the ablation matrix to strip both main effects and reports what survives.

**One model, many contexts, is the control — not a convenience.** Identical weights across every agent means observed differentiation cannot come from model heterogeneity. Prior observational work on unassigned roles used several different LLMs, leaving that confound open.

---

## Install

```bash
pip install -e ".[dev,analysis]"
pytest -q          # 257 tests, seconds, no GPU
```

Python 3.10+. `dev` brings the test suite; `analysis` brings pandas and
statsmodels for the mixed-effects test — everything else works without them, and
the module that needs them imports lazily and says so if they are missing.

**No GPU is required for any of the above.** The mock backend runs the entire
pipeline end to end, which is what makes the instrument validation below free.
Real runs additionally want `torch` with CUDA and `transformers`; those are
deliberately not declared as dependencies, so a laptop checkout stays small.

## Validate the instrument before trusting it

```bash
collabengine selftest
```

This builds three worlds whose answer is already known — genuinely specialized agents, wholly undifferentiated agents, and agents whose behavior tracks turn slot rather than identity — and checks the analysis recovers the right answer in each:

| world | diagonal dominance | interaction strength |
|---|---:|---:|
| specialized | 1.00 | 0.171 |
| positional | 0.25 *(chance)* | 0.015 |
| null | 0.25 *(chance)* | 0.008 |

Dominance separates the specialized world from the other two; it cannot separate
positional from null, because both put ownership at chance by construction and
chance is 1/n_agents, not zero. Interaction strength is what distinguishes those
two, and both sit an order of magnitude below the specialized world. The exact
dominance figure for a chance-level world moves between runs — with four
components a coincidental hit is a one-in-four event per column — so read it
against 0.25, never against 0.00.

A pipeline that reports specialization in the null world is manufacturing its result. Catching that here costs seconds; catching it after a rented GPU run costs the study.

## Run against a real model

The whole study fits on one 24 GB card — an RTX 4500 Ada here, serving Qwen3-8B
at bf16 to every agent. vLLM has no supported Windows build, so the default path
generates in-process through `transformers` on CUDA and supplies the batching
itself.

**Batching is the entire difference between an overnight run and a two-week
one.** Sweeping batch size at a realistic 1600-token context on this card:

| batch | aggregate | per sequence | peak |
|---|---|---|---|
| 8 | 72.6 tok/s | 9.1 | 16.8 GiB |
| 16 | 94.1 tok/s | 5.9 | 18.3 GiB |
| 32 | **108.1 tok/s** | 3.4 | 21.4 GiB |
| 48 | 71.1 tok/s | 1.5 | 24.4 GiB |

Decoding at this size is memory-bandwidth-bound, so extra sequences are nearly
free until they are not — and the batch-48 row is the one to read twice.
Throughput falls by a third there. **Windows does not raise OOM as allocations
approach the card; it pages to host memory over PCIe and keeps reporting 100%
GPU utilization**, so the collapse is invisible in `nvidia-smi` and looks
exactly like a healthy run. Sustained PCIe traffic under a pure decode workload
is the tell.

**The remedy is a hard cap, not a better budget.** `memory_fraction` (default
0.85) calls `set_per_process_memory_fraction`, so an over-ambitious batch fails
with an ordinary OOM that the batcher recovers from by halving, instead of
being absorbed into host memory. Tuning `max_batch_tokens` only makes paging
less likely; three successive budgets here all looked plausible and all paged.

That cap also exposed a bug it had been hiding, worth repeating because the
shape is general: **retrying after `except torch.cuda.OutOfMemoryError` cannot
free the memory that failed.** While the block runs, the exception's traceback
holds every frame inside `generate`, including the KV cache, so
`empty_cache()` reclaims nothing and each halved retry OOMs against the same
full allocator — turning one oversized batch into a stage of empty turns.
Recovery has to happen after the block exits.

Two earlier figures in this file were measured inside that regime and were
wrong: a "134 tok/s at batch 8" that came from 50-token prompts and did not
survive real contexts, and a "~490 KiB/token" KV cost inferred from an inflated
peak, which set the token budget three times too small and held the card near
40% of its throughput. Marginal cost across the healthy rows above is
~123 KiB/token, close to the 144 KiB the GQA geometry predicts (36 layers ×
8 KV heads × 128 dim × 2 × bf16).

Batches are bounded by **tokens, not request count**, and the budget must
include `max_tokens` as well as the prompt, since generation extends every
sequence in the chunk. This is what holds peak memory roughly constant as
contexts grow: at 44,000 tokens the batcher runs 32 sequences in round one and
12 by round three, at about the same GiB either way.

The remaining inefficiency is structural rather than a misconfiguration. A
chunk runs until its *longest* member stops, so with a 1024-token cap and ~600
tokens of useful output, roughly 40% of decode steps generate padding for
sequences that already finished. Continuous batching is the fix and it is the
main reason to prefer vLLM (~3–5× here) where a supported build exists.

```bash
collabengine pipeline --config configs/local-gpu.yaml --auto-difficulty
collabengine analyze  --config configs/local-gpu.yaml
```

`pipeline` is the recommended entry point: it calibrates, picks the operating
point, then runs baseline, the symmetry sweep, the fixed-order control and the
ablation grid — all in one process. Weights load once instead of per subcommand,
the independent phases share a single work queue so the batcher never drains
between them, and no phase waits on a human to read a table and start the next.
The individual steps still exist (`calibrate`, `baseline`, `ablate`) when you
want one of them on its own.

Then the observational half, and the question the project exists to answer:

```bash
collabengine code     --config configs/local-gpu.yaml --judge self --judge-name local8b
collabengine code     --config configs/local-gpu.yaml --judge gemini --judge-name gemini3 \
                      --judge-model gemini-3-flash-preview --condition baseline --limit 1
collabengine kappa    runs/<name>/codes.local8b.jsonl runs/<name>/codes.gemini3.jsonl
collabengine converge --config configs/local-gpu.yaml --codes runs/<name>/codes.local8b.jsonl
```

**On judges, and doing this for free.** PLAN.md specifies a frontier API judge,
because a local 7–8B is not a reliable coder. Frontier judges are supported
(`--judge anthropic`, `--judge gemini`), and a judge only ever reads finished
transcripts — using one to *produce* agent turns would destroy the one-model
control the design rests on.

Without a paid key, the free tiers do not stretch to a corpus: Gemini's free
quota is metered **per day**, at 20 requests per model, which is fewer than one
episode. The free path is therefore to code everything with the local model —
coding replies are ~8 tokens, so batched on the same card it is minutes and
costs nothing — and spend the daily frontier quota on a *subsample*, reporting
Cohen's κ between the two. That does not make the local judge as good as a paid
one. It measures how good it is, which is what κ was always for: a value far
below the 0.78 of the prior-art paper means Phase 4's correlation is not
trustworthy on those labels, and that is a result rather than a hidden weakness.

Coding checkpoints per episode and resumes, and it **aborts** after a run of
consecutive judge failures rather than continuing — a dead judge otherwise
labels every remaining message `other`, and a file of `other` is
indistinguishable from real data once written.

To serve with vLLM instead — WSL2, or a rented Linux box — point at
`configs/vllm-8b.yaml`. Only `backend.kind` differs.

Runs resume, and resume is covered by a test that asserts planned and recorded
episode ids are identical for every mode. This is not defensive: they once
disagreed (`live:A1:0` planned against `live:A1:tiny:0` recorded), so every
restart silently re-ran the entire ablation grid with no error to notice.

## What the analysis reports

`analyze` prints four things, in the order they should be read:

| Quantity | What it answers |
|---|---|
| **Interaction strength** | RMS of the double-centered drops. Zero when ablation is purely additive — i.e. when there is no specialization to find, only participation. |
| **Diagonal dominance** | Of all the agents you could remove, does the one the transcript calls a component's owner hurt it most? Scored against chance, `1/n_agents`. |
| **Per-mode drops and fungibility** | `Δ(frozen_replay) − Δ(live)`. Near zero means the contribution was irreplaceable; large means the role was real but any of them could have filled it. Excision is printed against its volume-matched control, never alone. |
| **Mixed-effects interaction** | Whether the interaction survives a significance test with episode as a random effect. The effect size comes from double-centering; this says whether it is distinguishable from noise. |

The random effect is not decoration. The same instance is played by the
un-ablated team and by every ablated variant, so an unusually hard instance
drags all of its rows down together; treating them as independent shrinks the
standard errors and manufactures significance. The test is joint across all
interaction coefficients rather than per-term — with four agents and four
components there are nine of them, so scanning for the smallest p-value would
find one under 0.05 essentially every time under a true null.

---

## Three confounds, handled in the design

**Compensation.** Remove an agent and the survivors reorganize to absorb its function, masking the very specialization you are measuring. Handled by running ablation three ways — `live` (compensation allowed), `frozen_replay` (regenerate surviving turns in the recorded schedule), and `frozen_excise` (delete the messages, re-read the answer, zero model calls). `Δ(frozen_replay) − Δ(live)` is the fungibility measure.

> **Measured caveat.** Plain excision was *expected* to give the largest drop, since it blocks compensation entirely. It gives nearly the smallest — ~0.002 against live drops of 0.03–0.23. Agents restate the whole working answer every turn, so a contribution is copied into everyone else's messages as soon as it is made; deleting the originator removes the words but not the content. `ablation.propagation_index()` detects this and decides which frozen mode to trust. Run it before reporting any excision-based number.

**Position vs. identity.** Roles may attach to turn order rather than agent identity — protocol, not specialization. Speaking order is reshuffled every round, which drives a purely positional world's dominance down to chance while leaving genuine identity-bound roles untouched. Under a *fixed* order the positional world still scores 0.50 against a 0.25 baseline: enough to be mistaken for real specialization.

**Symmetry breaking is the independent variable.** Identical model, prompt, and context produce identical output and nothing can specialize. What breaks the tie — name, seed, private scratchpad — is swept (`SymmetryBreaking`) rather than fixed and forgotten. The question is whether *minimal* asymmetry amplifies into stable roles.

> **Caveat under in-process batching.** `generate` samples from one global RNG
> for a whole batch, so per-agent seeds do not produce independent streams the
> way vLLM's per-request seeds do. Rows still sample independently, but
> `name_only` and `name_seed` stop being distinguishable conditions on this
> backend — they differ *only* by seed. Reproducing one specific episode
> bit-for-bit needs `max_batch_size: 1`; the analysis is unaffected either way,
> since transcripts are re-read rather than regenerated.

---

## Layout

| Path | What |
|---|---|
| `tasks/` | Instance generation (satisfiable by construction), per-component grading, prompt rendering |
| `orchestrator/` | Episode loop, team composition, turn scheduling |
| `backends/` | Model serving: `mock` (no GPU), `hf_local` (in-process CUDA, token-budget batching), `openai_compat` (vLLM), and the `anthropic` / `gemini` judges |
| `ablation/` | Live, frozen-replay, frozen-excise, plus capacity and random-message controls |
| `analysis/` | `interaction` (double-centering, diagonal dominance), `mixed` (episode as random effect), `coding` (action labels vs a permutation null), `convergent` (do labels predict contribution) |
| `runner/` | Bounded-concurrency execution with resume |
| `transcripts/` | JSONL episode records, sharded for parallel writers |

**Orchestration is hand-written on purpose.** Every mainstream agent framework ships role scaffolding in its prompt templates, which would silently plant the structure this project claims to observe emerging.
