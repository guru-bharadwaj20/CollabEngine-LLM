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

The study runs on one 24 GB card. vLLM has no supported Windows build, so the
default path generates in-process through `transformers` on CUDA, with dynamic
micro-batching to keep the card saturated — measured at 9.6 output tok/s for a
single turn against 134 tok/s at batch 8, which is the difference between a
two-week grid and an overnight one.

```bash
collabengine calibrate --config configs/local-gpu.yaml   # Phase 1: pick a difficulty
collabengine pipeline  --config configs/local-gpu.yaml   # Phases 2-3, one process
collabengine analyze   --config configs/local-gpu.yaml
```

`pipeline` runs baseline, the symmetry sweep, the fixed-order control and the
ablation grid back to back. Weights load once rather than per subcommand, and
the three independent phases share a single work queue so the batcher never
drains between them.

Then the observational half, and the question the project exists to answer:

```bash
collabengine code     --config configs/local-gpu.yaml --judge-name haiku
collabengine code     --config configs/local-gpu.yaml --judge-name sonnet \
                      --judge-model claude-sonnet-4-5      # second family
collabengine kappa    runs/<name>/codes.haiku.jsonl runs/<name>/codes.sonnet.jsonl
collabengine converge --config configs/local-gpu.yaml --codes runs/<name>/codes.haiku.jsonl
```

Coding uses a frontier API model, because a local 7–8B is not a reliable judge.
It reads finished transcripts and never runs an episode — using one to *produce*
agent turns would destroy the one-model control the whole design rests on. Set
`ANTHROPIC_API_KEY`, or pass `--judge self` for a pipeline smoke test whose
labels nobody should report.

To serve with vLLM instead — WSL2, or a rented Linux box — point at
`configs/vllm-8b.yaml`. Only `backend.kind` differs.

Runs resume, and resume is checked by a test: plan ids and recorded episode ids
are asserted identical for every mode, because a mismatch re-runs the whole grid
silently rather than failing.

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
| `backends/` | Model serving: mock (no GPU), in-process CUDA, OpenAI-compatible (vLLM), and the frontier judge |
| `ablation/` | Live, frozen-replay, frozen-excise, plus capacity and random-message controls |
| `analysis/` | Double-centered interaction and diagonal dominance; behavioral coding against a permutation null; convergent validity |
| `runner/` | Bounded-concurrency execution with resume |
| `transcripts/` | JSONL episode records, sharded for parallel writers |

**Orchestration is hand-written on purpose.** Every mainstream agent framework ships role scaffolding in its prompt templates, which would silently plant the structure this project claims to observe emerging.
