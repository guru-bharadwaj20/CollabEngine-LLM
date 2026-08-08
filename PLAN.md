# CollabEngine-LLM — Project Plan

A local framework for spinning up LLM agent teams and causally verifying their division of labor.

---

## 0. Verdict on the idea

**The intersection you identified is real and, as far as I can find, unoccupied.** But it is narrower than "observe emergence + ablate," and the framing needs one correction to be defensible.

### What already exists

| Work | Roles | Method | Gap it leaves |
|---|---|---|---|
| [ROMA (2020)](https://arxiv.org/pdf/2003.08039) and the MARL lineage | Emergent | Learned role embeddings | Not LLM agents; roles are architectural, not linguistic |
| [Behavioral Differentiation Without Role Assignment (2604.00026)](https://arxiv.org/pdf/2604.00026) | **Emergent** | **Observational only** — 208 runs, 13,786 coded messages, LLM-judge behavioral coding, cosine similarity, Cohen's κ | **No causal ablation.** Explicitly the "reading it into transcripts" problem |
| [Agents that Matter (2605.27621)](https://arxiv.org/abs/2605.27621) | **Pre-assigned** | **Causal LOO ablation**, attribution as a cooperative game | Ablation used to *optimize* a system (find bottleneck agents), not to *validate* that specialization is real |
| [IntrospecLOO (2505.22192)](https://arxiv.org/abs/2505.22192) | Pre-assigned | LOO approximation via introspection | — |
| [AgentDropout (2503.18891)](https://arxiv.org/abs/2503.18891) | Pre-assigned | Dynamic agent elimination for efficiency | — |

So: the observation half is done (April 2026, and thoroughly). The ablation half is done (May 2026) but only on **assigned** roles and only as an engineering tool. **Nobody has pointed the causal instrument at the emergent phenomenon.** That is your slot.

Two pieces of external support for your premise, both worth citing:
- *Agents that Matter* found that **introspective LLM judges do not faithfully approximate ablation behavior**. That is direct third-party evidence that transcript-reading ≠ causal reality — the exact motivation for your project.
- *Behavioral Differentiation* found that groups "spontaneously exhibit compensatory response patterns when an agent crashes." That is simultaneously encouraging and the single largest threat to your design (see §2).

### The correction your framing needs

> "Knock out one agent, measure the team's performance drop, prove specialization is functionally real."

A scalar performance drop does **not** demonstrate specialization. It demonstrates **participation**. If you remove any competent contributor from any team, output falls — that is true of four identical agents with no division of labor at all.

**Specialization is a claim about *differential* contribution, not total contribution.** The correct causal signature is an **agent × task-component interaction**: if agent A emerged as "the critic" and B as "the planner," then ablating A must damage the criticism-loaded components of the task *more than* the planning-loaded ones, and ablating B must do the reverse. A main effect proves nothing; the crossover interaction is the result.

This single change is what separates a publishable finding from a restatement of "more agents do more work." Design the task around it (§3).

---

## 1. Hardware reality check — read before anything else

Your premise: *"One quantized 7–8B model serving all agent turns fits 24GB easily."* Correct — and this machine is that machine.

| | Detected (2026-08-08) |
|---|---|
| CPU | Intel Core i9-14900 — **24 cores / 32 threads**, Raptor Lake |
| RAM | **128 GB** |
| GPU | **NVIDIA RTX 4500 Ada Generation, 24 GB GDDR6** — driver 595.95, CUDA 13.2, `torch.cuda.is_available() == True` |

> **This section previously described an i3-5005U with 7.9 GB of RAM and no CUDA device, and on that basis routed every real number to a rented GPU.** That hardware is not what the project is running on. The correction matters beyond bookkeeping: it moves Phases 1–4 from "rent a box first" to "run tonight," and it is why the phase list below no longer gates on a rental. What the old constraint produced — a backend abstraction, a mock that exercises the full pipeline for free, and resumable runs — stays, because each is worth having for its own reasons.

**Sizing the experiment:**

- One episode ≈ 4 agents × ~10 turns × ~300 output tokens ≈ **12k output tokens**
- Statistical power for an interaction effect ≈ **100 episodes per condition**
- Conditions ≈ 1 baseline + (4 agents × 2 ablation modes) + controls ≈ **~12**
- Total ≈ **~14M output tokens**

An 8B model at bf16 is ~16.4 GB of weights, leaving ~7 GB for KV cache on a 24 GB card — enough for a batch of 16–32 at 4–8k context. Sustained batched throughput in that regime is a few hundred to ~1k aggregate output tok/s, putting the full grid at **roughly 4–12 hours of local wall clock**. Overnight, at zero marginal cost.

**Decision: develop and execute locally.** Rental is no longer on the critical path. It returns only for the Phase 4 robustness sweep (2–3 model families × 3–5 team sizes), which is the one part of the plan that genuinely wants more than one card, and even that is a convenience rather than a blocker.

**Serving caveat — Windows.** vLLM ships no supported Windows build, so the batching server the plan assumed is not directly available. Two paths exist: vLLM inside the WSL2 Ubuntu instance already present on this box, or in-process generation via `transformers` on CUDA. The phases below run against `backends/hf_local.py` — a CUDA `transformers` backend that micro-batches concurrent turns — which keeps the OpenAI-compatible backend intact and unused-but-ready for a WSL vLLM server or a rented box. **The backend abstraction earns its keep here:** switching between them is a config line, exactly as Phase 0 required.

The "one model, many conversation contexts" insight stays exactly right — and it is more than a convenience. It is your **critical control**: identical weights across all agents means observed differentiation *cannot* come from model heterogeneity. Note that 2604.00026 used 7 *different* LLMs, so their differentiation is partly confounded by model identity. Yours is not. **That is a second, independent novelty axis — name it in the paper.**

---

## 2. The three confounds that will kill this if unhandled

Address these in the design, not the discussion section.

### C1 — Compensation (the big one)
When you remove agent i and re-run, the survivors reorganize and absorb its function. A small drop is then ambiguous: *"i did nothing"* or *"i's role was real but fungible."* Both look identical in a scalar metric.

**Fix: run three ablation modes and treat their differences as results.**
- **Live ablation** — delete agent i, re-run the episode from scratch with N−1 agents. Measures *necessity of the agent*. Compensation is allowed.
- **Frozen-replay** — keep the recorded schedule, drop agent i's turns, and regenerate the surviving agents' turns against the modified context. Compensation is allowed within a turn slot but not across the schedule. **This is the primary frozen measure.**
- **Frozen-excise** — delete agent i's messages from the recorded transcript and re-read the answer from what remains. Costs *zero model calls*, so it can run over the entire corpus for free — but see the caveat below.

`Δ(frozen_replay) − Δ(live)` **is your fungibility/redundancy metric.**

> ⚠️ **Propagation caveat — measured, not predicted.** I originally expected plain excision to give the *largest* drop, since it blocks compensation completely. Implemented and measured against the mock, it gave nearly the *smallest*: ~0.002 against live drops of 0.03–0.23, a hundredfold underestimate.
>
> The cause is content propagation. Agents restate the whole working answer every turn, so a contribution is copied into everyone else's messages almost as soon as it is made. Deleting the originating messages removes the words but not the content. Read naively this reports "agent contributed nothing" when the truth is "this measurement does not work on this transcript."
>
> `ablation.propagation_index()` measures how much of an agent's content is echoed by others later in the episode, and decides which frozen mode to trust. **Run it before reporting any excision-based number.** Whether real 7–8B agents restate as aggressively as the mock is an open empirical question for Phase 2 — but the diagnostic is now in place to answer it rather than assume.

### C2 — Position, not identity
The most likely deflating explanation of the entire phenomenon: agent 1 "plans" only because it speaks first. Roles attach to *turn position*, not to *agent identity* — and that is not specialization, it is protocol.

**Fix:** randomize speaking order per episode and per turn. Then test whether role labels track identity or slot. Cheap to run, and if roles turn out to be positional you have learned that in Phase 3 instead of after the writeup. **Do this early — it is the highest-information-per-GPU-hour test in the whole project.**

### C3 — Symmetry breaking is the independent variable
Identical model + identical prompt + identical context ⇒ near-identical outputs, and there is nothing to specialize. Something must break symmetry: distinct agent names, per-agent sampling seeds, turn order, a shared scratchpad. The real question is **whether minimal symmetry-breaking amplifies into stable roles** — so make it a swept parameter, not an implementation detail. Sweep: name-only → name+seed → name+seed+private scratchpad.

---

## 3. Task design

Requirements: programmatically gradable, decomposable into components that load on *different* capabilities, long enough that division of labor pays, unlimited episode supply, no benchmark contamination.

**Recommendation: a synthetic multi-constraint task generator**, not an off-the-shelf benchmark.

Concretely — a constrained scheduling/allocation problem where each instance carries tagged constraint classes:
- **Arithmetic/consistency** constraints (checkable exactly)
- **Search/enumeration** constraints (require exploring alternatives)
- **Verification** constraints (a planted error that must be *caught*, not produced)
- **Synthesis** constraints (require combining two agents' partial results)

Why this shape: each class is independently scored, giving you the **task-component axis for the interaction test in §0**. Difficulty knobs are continuous. Instances are unlimited and uncontaminated. And a 7–8B model can actually do it — unlike agentic coding, where small models fail for reasons unrelated to your question.

Avoid SWE-bench-style tasks at this model scale; the failure floor swamps the effect you are looking for.

---

## 4. Phases

### Phase 0 — Foundations (local, ~1 week)
- Repo scaffold, `uv`/`venv`, config-driven experiment runner, deterministic seeding, full transcript logging to JSONL (every message, every agent, every seed — you will re-analyze this many times).
- Model server abstraction: one interface, two backends (llama.cpp local / vLLM OpenAI-compatible remote). Swapping backends must be a config line.
- **Deliverable:** 4 agents on a 1B model complete one toy episode locally, transcript logged and replayable.

### Phase 1 — Task generator + grader (local, ~1 week)
- Instance generator with tagged constraint classes and difficulty knobs.
- Deterministic grader returning **per-component scores**, not a scalar.
- Calibration: single-agent baseline and 4-agent baseline across difficulty, to find the band where the task is hard enough to need collaboration but not so hard the model floors out. **If there is no such band, stop and redesign the task — everything downstream depends on it.**
- **Deliverable:** difficulty curve plot; a chosen operating point.

### Phase 2 — Emergence measurement (local GPU, ~1 week)
- Local 24 GB + 7–8B (Qwen3-8B or Llama-3.1-8B-Instruct). Batch aggressively — this is the difference between hours and days, whether the batching happens in vLLM or in `hf_local`.
- Run the symmetry-breaking sweep (C3) × baseline episodes.
- Behavioral coding of transcripts into action-type labels. **Use a frontier API model as the judge** — a local 7–8B is too weak to code reliably. Two judges from different families, blind to condition, human-validated on a ~500-message subsample, report Cohen's κ (2604.00026 got 0.78; match or beat it). Bulk coding with Claude Haiku 4.5 on ~14k messages runs roughly $5.
- Metrics: per-agent action distributions, pairwise JS divergence, role stability within episode, role consistency across seeds — each against a **permutation null**.
- **Run the C2 position-vs-identity test here.**
- **Deliverable:** does differentiation occur, is it stable, and does it attach to identity or to slot? Go/no-go gate for Phase 3.

### Phase 3 — Causal ablation (GPU, ~2 weeks) — *the core contribution*
- Implement both ablation modes from C1 (live re-run; frozen-transcript excision).
- Run the full condition grid: baseline + per-agent ablation × both modes.
- **Controls that make the result interpretable:**
  - **Capacity control** — an N−1 team that *never had* agent i, run from scratch. Separates "specialization" from "one fewer worker."
  - **Random-message ablation** — excise a volume-matched random subset of messages. Separates "this agent's contribution" from "less text in context."
- **Primary analysis: the agent × task-component interaction**, mixed-effects model with episode as a random effect. The specialization claim lives or dies on this term, not on the main effect.
- **Deliverable:** per-agent causal contribution profiles; the fungibility metric `Δ(frozen) − Δ(live)`.

### Phase 4 — Convergent validity (GPU, ~1 week)
The question that makes the paper matter: **does the transcript-derived role label predict the ablation-derived contribution profile?**
- Correlate Phase 2 behavioral labels against Phase 3 causal profiles.
- A weak correlation is *not* a failed project — it is the strongest possible version of your original thesis, and it independently replicates *Agents that Matter*' finding that introspective judgment diverges from ablation. Say so plainly.
- Robustness: 2–3 model families, 3–5 team sizes, difficulty sweep.

### Phase 5 — Writeup and release (~2 weeks)
- Preregister Phases 3–4 hypotheses **before** running them (OSF). Cheap, and it is what makes the interaction test credible rather than post-hoc.
- Release: harness, task generator, all transcripts, analysis notebooks. The transcript corpus alone has standalone value.

**Total: ~8 weeks part-time, $0 of GPU (local card), ~$10 of judge API.**

---

## 5. Stack

| Layer | Choice | Note |
|---|---|---|
| Serving (primary) | `hf_local` — `transformers` on CUDA with dynamic micro-batching | Runs on Windows, where vLLM does not. Batching is what makes this tractable — do not use one-request-at-a-time |
| Serving (alternate) | vLLM, OpenAI-compatible endpoint | For a WSL2 server or a rented box. Reached by changing `backend.kind`, nothing else |
| Serving (free) | `mock` | Full-pipeline debugging and instrument validation with no GPU at all |
| Models | Qwen3-8B, Llama-3.1-8B-Instruct, Mistral-7B | ≥2 families for Phase 4 robustness |
| Judge | Frontier API model, 2 families, blind to condition | Local models are not adequate judges |
| Orchestration | Plain Python. No LangGraph/AutoGen/CrewAI | Frameworks impose role structure — the exact thing you are trying to observe emerging. A hidden prompt template would invalidate the whole result |
| Analysis | pandas + statsmodels/pymer4 mixed models | — |

**The orchestration row is not a style preference.** Every mainstream agent framework ships opinionated role scaffolding in its prompts. Using one would silently plant the roles you claim to have observed emerging.

---

## 6. Risk register

| Risk | Likelihood | Response |
|---|---|---|
| Roles are positional, not identity-bound | **High** | Test in Phase 2 (C2). If positional — that is a real, publishable negative result about the emergence literature |
| 7–8B too weak to collaborate meaningfully | Medium | Calibrate in Phase 1; step up to 14B (still fits 24 GB at Q4) if the floor is too low |
| Compensation masks all ablation effects | Medium | Frozen-transcript mode (C1) is the mitigation, and its gap is a result in itself |
| Differentiation is real but tiny / high-variance | Medium | Power analysis before Phase 3; increase episodes — compute is cheap here |
| Judge coding unreliable at 7–8B transcript quality | Low–Med | Two frontier judges + human validation + report κ |
| Scooped mid-project | Low–Med | The field is moving fast (both key papers are 2026). Preregister early; Phase 3 is the defensible core |

---

## 7. The claim to aim for

> Emergent role differentiation in same-model LLM agent teams is behaviorally observable and statistically stable — but its transcript-derived labels predict causal contribution only weakly. Specialization is functionally real to the extent that ablating an agent damages *its* task components differentially; where that interaction is absent, apparent specialization is an artifact of reading structure into text.

Either direction of that result is worth publishing. That is the mark of a well-posed experiment.
