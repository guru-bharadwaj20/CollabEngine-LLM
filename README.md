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
pip install -e ".[dev]"
pytest -q
```

No GPU required for anything above. The mock backend runs the entire pipeline.

## Validate the instrument before trusting it

```bash
collabengine selftest
```

This builds three worlds whose answer is already known — genuinely specialized agents, wholly undifferentiated agents, and agents whose behavior tracks turn slot rather than identity — and checks the analysis recovers the right answer in each:

| world | diagonal dominance | interaction strength |
|---|---:|---:|
| specialized | 1.00 | 0.171 |
| positional | 0.25 *(chance)* | 0.012 |
| null | 0.00 | 0.007 |

A pipeline that reports specialization in the null world is manufacturing its result. Catching that here costs seconds; catching it after a rented GPU run costs the study.

## Run against a real model

```bash
vllm serve Qwen/Qwen3-8B --max-model-len 8192 --max-num-seqs 64

collabengine calibrate --config configs/vllm-8b.yaml   # pick a difficulty
collabengine baseline  --config configs/vllm-8b.yaml   # Phase 2
collabengine ablate    --config configs/vllm-8b.yaml   # Phase 3
collabengine analyze   --config configs/vllm-8b.yaml
```

Runs resume: a dropped connection re-runs only what is missing.

---

## Three confounds, handled in the design

**Compensation.** Remove an agent and the survivors reorganize to absorb its function, masking the very specialization you are measuring. Handled by running ablation three ways — `live` (compensation allowed), `frozen_replay` (regenerate surviving turns in the recorded schedule), and `frozen_excise` (delete the messages, re-read the answer, zero model calls). `Δ(frozen_replay) − Δ(live)` is the fungibility measure.

> **Measured caveat.** Plain excision was *expected* to give the largest drop, since it blocks compensation entirely. It gives nearly the smallest — ~0.002 against live drops of 0.03–0.23. Agents restate the whole working answer every turn, so a contribution is copied into everyone else's messages as soon as it is made; deleting the originator removes the words but not the content. `ablation.propagation_index()` detects this and decides which frozen mode to trust. Run it before reporting any excision-based number.

**Position vs. identity.** Roles may attach to turn order rather than agent identity — protocol, not specialization. Speaking order is reshuffled every round, which drives a purely positional world's dominance down to chance while leaving genuine identity-bound roles untouched. Under a *fixed* order the positional world still scores 0.50 against a 0.25 baseline: enough to be mistaken for real specialization.

**Symmetry breaking is the independent variable.** Identical model, prompt, and context produce identical output and nothing can specialize. What breaks the tie — name, seed, private scratchpad — is swept (`SymmetryBreaking`) rather than fixed and forgotten. The question is whether *minimal* asymmetry amplifies into stable roles.

---

## Layout

| Path | What |
|---|---|
| `tasks/` | Instance generation (satisfiable by construction), per-component grading, prompt rendering |
| `orchestrator/` | Episode loop, team composition, turn scheduling |
| `backends/` | Model serving: mock (no GPU) and OpenAI-compatible (vLLM / llama.cpp) |
| `ablation/` | Live, frozen-replay, frozen-excise, plus capacity and random-message controls |
| `analysis/` | Double-centered interaction, diagonal dominance |
| `runner/` | Bounded-concurrency execution with resume |
| `transcripts/` | JSONL episode records, sharded for parallel writers |

**Orchestration is hand-written on purpose.** Every mainstream agent framework ships role scaffolding in its prompt templates, which would silently plant the structure this project claims to observe emerging.
