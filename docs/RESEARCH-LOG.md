# CollabEngine-LLM — Research and Engineering Log

A complete record of the project to date: what was built, what broke, what each
failure cost, and which findings survive. Written to be usable as raw material
for a paper, so it separates **results** (things measured about agent teams)
from **instrument history** (things learned about measuring them) — and is
explicit about the several occasions where the second was mistaken for the
first.

**Status at time of writing: 2026-08-08 23:00.** 45 commits, 277 passing tests,
one completed Phase 1 measurement, no Phase 3 results yet. A `hard`-difficulty
run is in progress; its solo arm is in and reported in §4.2.

---

## 1. The claim under test

> Emergent role differentiation in same-model LLM agent teams is behaviorally
> observable and statistically stable — but its transcript-derived labels
> predict causal contribution only weakly. Specialization is functionally real
> to the extent that ablating an agent damages *its* task components
> differentially; where that interaction is absent, apparent specialization is
> an artifact of reading structure into text.

### 1.1 The correction that defines the design

The project began from "knock out one agent, measure the performance drop,
prove specialization is real." That is not a test of specialization. A scalar
drop demonstrates **participation** — remove any competent contributor from any
team and output falls, which is equally true of four identical agents with no
division of labor.

Specialization is a claim about **differential** contribution. The causal
signature is an **agent × task-component interaction**: if agent A emerged as
the critic and B as the planner, ablating A must damage criticism-loaded
components *more than* planning-loaded ones, and ablating B the reverse. The
main effect proves nothing; the crossover is the result.

Every downstream design decision follows from this: per-component grading
rather than a scalar score, double-centering to strip main effects, and a
mixed-effects model with episode as a random intercept to test the interaction
term specifically.

### 1.2 Position against prior work

| Work | Roles | Method | Gap |
|---|---|---|---|
| ROMA (2020) and MARL lineage | Emergent | Learned role embeddings | Not LLM agents; roles architectural, not linguistic |
| Behavioral Differentiation Without Role Assignment (2604.00026) | Emergent | Observational only — 208 runs, 13,786 coded messages, κ = 0.78 | **No causal ablation** |
| Agents that Matter (2605.27621) | Pre-assigned | Causal LOO ablation | Ablation used to *optimize*, not to *validate* emergence |

The observational half and the causal half both exist. Nobody has pointed the
causal instrument at the emergent phenomenon. Two external results support the
premise: *Agents that Matter* found introspective LLM judges do **not**
faithfully approximate ablation behaviour (third-party evidence that
transcript-reading ≠ causal reality), and *Behavioral Differentiation* found
teams "spontaneously exhibit compensatory response patterns when an agent
crashes" (simultaneously encouraging and the largest threat to the design).

**Second novelty axis:** 2604.00026 used seven *different* LLMs, so their
differentiation is partly confounded with model identity. Using one model for
every agent makes identical weights a control — observed differentiation cannot
come from model heterogeneity.

### 1.3 The three confounds, addressed in design

- **C1 Compensation.** Survivors reorganize and absorb a removed agent's
  function, so a small drop is ambiguous between "did nothing" and "was real but
  fungible." Handled by running three ablation modes and treating their
  *differences* as the result. `Δ(frozen_replay) − Δ(live)` is the fungibility
  metric.
- **C2 Position vs identity.** Agent 1 may "plan" only because it speaks first —
  that is protocol, not specialization. Handled by randomizing speaking order
  per episode and per turn, plus a `fixed_order` control condition.
- **C3 Symmetry breaking as the independent variable.** Identical model, prompt
  and context produce identical outputs; something must break symmetry. Made a
  swept parameter (name-only → name+seed → name+seed+scratchpad) rather than an
  implementation detail.

---

## 2. Chronology

### 2.1 Day 1 (2026-08-07, 17:48 – 20:05) — scaffold on the assumption of no GPU

Eight commits built the whole pipeline against a mock backend: task generator
with tagged constraint classes, per-component grader, prompt rendering and
answer parsing, orchestrator, transcript store, ablation modes with controls,
the propagation diagnostic, an OpenAI-compatible backend, a parallel runner, and
the CLI/config system.

This was done under the belief that the machine was an i3-5005U with 7.9 GB RAM
and no CUDA device, which is what PLAN.md §1 stated. That belief shaped the
architecture productively — it forced a backend abstraction, a mock capable of
exercising the full pipeline for free, and resumable runs. All three were worth
keeping once the belief turned out to be false.

### 2.2 Day 2 (2026-08-08) — the GPU day

| Time | Event |
|---|---|
| 07:32 | Hardware corrected: RTX 4500 Ada 24 GB, CUDA available. Python floor lowered to 3.10 |
| 07:38 | `hf_local` CUDA backend added — the enabler for every GPU phase |
| 07:58 | Phase 2–4 analysis added; **resume bug found and fixed** |
| 08:01 | Batches bounded by tokens rather than request count |
| 08:23 | Gemini judge added after the judge-availability pivot |
| 09:36 | Phase 1 gate folded into the run as a paired solo condition |
| 09:41 | **Ablation reference pollution fixed** |
| 12:26 | Batch budget reset from measurement (first, wrong, measurement) |
| 17:11 | **Integrity filter added** — token cap was writing zeros into the grid |
| 17:45 | Mid-stage throughput heartbeat added |
| 18:17 | Production-shape benchmark; allocator exonerated |
| 19:17 | **OOM recovery fixed** — retry inside `except` can never free memory |
| 21:42 | **Operating point moved to `hard`** — Phase 1 gate failed at `medium` |

---

## 3. Failures, in detail

Ordered by how much they mattered. Every one of the first four produced
**plausible wrong data rather than an error**, which is the thread connecting
them and the main methodological lesson of the project so far.

### 3.1 The token cap was writing zeros into the ablation grid

**Symptom.** In the first real corpus, 3 of 12 solo episodes scored exactly
0.0 with `detail: {malformed: true}`.

**Diagnosis.** `finish_reason: "length"` on every turn of those episodes.
Agents spent all 512 tokens walking 16 jobs through 29 constraints and were cut
off before emitting the `<answer>` JSON block. Across the corpus, **72% of all
agent turns (26/36) hit the cap**; mean completion was 424 of 512 tokens.

**Why it was dangerous rather than merely wasteful.** A malformed episode grades
0.0 on every component, which is indistinguishable in the means from a team that
answered badly. Left in, it biases in both directions at once: it deflates the
baseline reference (understating every drop) and inflates whichever ablation
cell it lands in (overstating that one). Worse, **truncation is not random with
respect to the phenomenon under study** — a turn is cut off when its author
writes a lot, and how much each agent writes is precisely the emergent behaviour
being measured. That manufactures an agent × component interaction out of the
token budget: a false positive in the exact direction the study cannot afford.

**Fix.** `max_tokens` 512 → 1024, plus `analysis/integrity.py`, which classifies
an episode as an instrument failure rather than a result. The rule is
deliberately narrow — malformed **and** every agent turn truncated, meaning no
turn ever had the opportunity to finish. A malformed answer after a turn that
ended on its own is a genuine refusal to commit and stays in the means.
`analyze` prints the per-condition table before any number that depends on it,
so the filter is never silent.

**Effect of the fix.** Truncation 72% → 28%, mean completion 424 → 627 tokens,
malformed 3/12 → 0/12.

**Cost.** Roughly two hours of GPU time spent generating a corpus that had to be
archived.

### 3.2 The calibration measured the token cap and reported it as a difficulty

**Symptom.** With the cap fixed, solo score at `medium` jumped from 0.55 to
0.879.

**Diagnosis.** `calibrate` had chosen `medium` as the operating point while
`max_tokens` was 512. Truncation was suppressing solo performance, which made
medium look correctly pitched. Fixing the cap raised the scores and invalidated
the difficulty choice in the same stroke — **one bug, two casualties, the second
visible only after fixing the first**.

**Consequence.** See §4.1 — the Phase 1 gate fails at `medium`. Had this not
been caught, roughly six hours of ablation grid would have run at an operating
point where the team contributes nothing, producing small drops against a
baseline the extra agents were not lifting, and those drops would have looked
like results.

**Generalisable lesson, and the one most worth putting in the paper.** *A
calibration is only as trustworthy as the instrument settings in force when it
ran, and nothing in its output records what those were.* This is why `analyze`
now prints `finish_reason` accounting ahead of every dependent number.

### 3.3 CUDA OOM recovery could never work

**Symptom.** After adding a hard memory cap, a stage wrote 12 episodes in four
minutes — every turn `finish_reason: "error"`, zero tokens, all graded 0.0.

**Diagnosis.** The halving retry lived *inside* `except
torch.cuda.OutOfMemoryError`. While an except block runs, the active exception's
traceback holds every frame inside `generate()` — including the KV cache that
just failed to fit. `torch.cuda.empty_cache()` therefore reclaims nothing, each
halved retry starts from a still-full allocator and OOMs in turn, all the way
down to a single sequence. One oversized batch became an entire stage of empty
turns.

**The bug was always present**; paging (§3.4) had been masking it by absorbing
oversized batches into host memory instead of failing.

**Fix.** Set a flag in the handler, recover after the block exits, where `del`
and `empty_cache()` do what they say. The integrity filter was extended: an
errored turn condemns its episode on its own — unlike truncation there is no
reading under which it says anything about the agent whose turn it was — and it
counts even if the surviving turns still produced a parseable answer, because
the episode then describes a team missing a member it should have had.

### 3.4 Windows pages to host RAM instead of raising OOM

**The single most expensive failure of the project.**

**Symptom.** Runs at 100% GPU utilization, no errors, producing almost nothing.
One stage ran 4 hours and completed zero episodes.

**Why it resisted diagnosis for most of a day.** Every surface that normally
signals trouble looks healthy:

| Signal | Paging | Healthy |
|---|---|---|
| GPU utilization | 100% | 98–99% |
| Errors | none | none |
| Peak allocated bytes | plausible | plausible |
| **Power draw** | **62–64 W** of 210 | **155–176 W** |
| **PCIe traffic** | **18,000 / 12,600 MB/s** | **50–80 / 15–32 MB/s** |

The WDDM driver does not refuse an allocation that no longer fits; it migrates
the working set to host memory over PCIe and the run continues several times
slower. `nvidia-smi dmon -s t` is the only readily available surface that
distinguishes the two states. **Sustained multi-GB/s under a pure decode
workload means the budget is too high, whatever the peak allocation says.**

**Fix.** `torch.cuda.set_per_process_memory_fraction(0.90)`. This makes paging
*impossible* rather than merely unlikely: the allocation fails first, and a
failure is an ordinary OOM the batcher recovers from. Tuning the token budget
only ever changes the probability — three successive budgets (44000, 40000,
30000) all looked defensible and all paged.

**Cost.** Approximately six hours of GPU time across several runs, plus one run
killed roughly 30 minutes before it would have written its episodes.

### 3.5 Resume was broken for the entire ablation grid

**Symptom.** Restarts silently re-ran work already on disk.

**Diagnosis.** Planned episode ids (`live:A1:0`) did not match recorded ids
(`live:A1:tiny:0`), so no plan was ever skipped. No error — just hours of
repeated card time.

**Fix.** `_run_id` / `_derived_id` helpers, plus a test asserting that the set of
planned ids equals the set of recorded ids across every mode. That test is the
thing that keeps it fixed.

### 3.6 The ablation reference was polluted by non-baseline episodes

**Symptom.** Found by inspection, not by failure.

**Diagnosis.** Stage 1 writes baseline, solo, symmetry and fixed-order episodes
to one transcript. `_component_means` averaged across all of them, folding a
one-agent team's score into the reference that four-agent ablations are measured
against — shrinking every drop toward zero and, where solo scored low enough,
past it.

**Fix.** Condition filter defaulting to `baseline`, plus a test asserting that
filtered and unfiltered means differ.

### 3.7 Smaller failures

| Failure | Consequence | Fix |
|---|---|---|
| `model.generate(generator=...)` unsupported | — | `transformers.set_seed` before each batch |
| `set_seed` rejects ≥ 2³² | Crash | Hash batch seeds mod 2³²−1 |
| asyncio queue bound to first event loop | Crash on reuse | Track loop, rebuild queue on change |
| Heartbeat divided by previous pass's tokens | Rate read 17→26→29 for a run steady at 31 | Count tokens before reporting |
| Followup matched `python.exe` by name | Judge could start while card busy | Match command line |
| Six stale followup processes accumulated | Six 16 GiB judge loads on completion | Kill and re-arm exactly one |
| `grep` pipe block-buffering | Hid background output repeatedly | Write to files, `python -u` |
| Qwen3 `<think>` block | Would consume the whole turn budget | `enable_thinking=False` |
| Test didn't exercise length-sorting (FakeHF overrode the method) | False confidence | Extract `_sorted_chunks`, test directly |

---

## 4. Results

### 4.1 Phase 1 gate at `medium` — **fails**

Qwen3-8B, 12 episodes per condition, identical instances, `max_tokens: 1024`:

| condition | n | mean | sd |
|---|---|---|---|
| baseline (4 agents × 3 rounds) | 12 | 0.905 | 0.095 |
| solo (1 agent × 3 rounds) | 12 | 0.879 | 0.100 |
| symmetry: name_seed_scratch | 10 | 0.851 | 0.098 |

**Team − solo gap: +0.026, against a standard error of ~0.029.**
Indistinguishable from zero.

PLAN.md Phase 1 makes this a stop condition: *"find the band where the task is
hard enough to need collaboration but not so hard the model floors out. If there
is no such band, stop and redesign."* It stops Phase 3 at this operating point
rather than stopping the project — an ablation grid run where the team
contributes nothing measures its own noise floor.

**This is a real finding and belongs in the paper**, both as a negative result
about 8B agent teams on constraint-satisfaction tasks, and as the difficulty
curve Phase 1 asked for. The corpus is archived, not deleted.

**Read this together with §4.2.** Under `fraction` the medium gap is +0.026;
under whole-instance feasibility the same transcripts give 0.250 vs 0.417 --
teams produce a fully feasible schedule 1.7x as often. The gate verdict does
not change (that gap is still inside its error at n=12), but the headline
metric was hiding the largest signal in the corpus.

Action taken: operating point moved to `hard` (24 jobs vs 16, 6 workers vs 5, 6
exclusions vs 4, 5 synthesis constraints vs 3, capacity slack 1.1 vs 1.2, value
floor 0.72 vs 0.65).

**Open:** whether `hard` clears the gate. If it does not, the honest reading is
that an 8B model has no band in this task where collaboration pays — a Phase 1
result, and not a licence to run Phase 3 anyway.

### 4.2 The scoring metric hides difficulty but is not what limits discrimination

Solo scored 0.879 at `medium` and 0.842 at `hard` — a 50% increase in instance
size moved the score by 0.037. The hypothesis was that
fraction-of-constraints-satisfied cannot discriminate, since extra constraints
add to numerator and denominator together.

Instances are deterministic in `(seed, difficulty)`, so this was testable
against solutions already on disk at **zero GPU cost** (`scripts/rescore.py`).
Three metrics, same transcripts:

| medium, n=12 each | solo | team | gap | Cohen's *d* |
|---|---|---|---|---|
| `fraction` (current) | 0.879 | 0.905 | +0.026 | +0.27 |
| `strict` (all-or-nothing per component) | 0.616 | 0.689 | +0.074 | +0.25 |
| `feasible` (whole instance) | 0.250 | 0.417 | +0.167 | +0.36 |

**The hypothesis was half right, and the wrong half was the important one.**
Strict scoring exposes large headroom — solo falls from 0.879 to 0.616, and only
a quarter of solo answers are fully feasible — but **Cohen's *d* barely moves**
(0.27 → 0.25 → 0.36). Changing the metric changes the numbers without changing
the separation. On the medium corpus the team genuinely is not much better than
one agent; that is not a measurement artifact and re-grading does not rescue it.

What the experiment does establish is more useful: **`hard` is genuinely hard,
and the fraction metric conceals it.**

| hard, solo, n=12 | value |
|---|---|
| `fraction` | 0.842 |
| `strict` | 0.490 |
| `feasible` | **0.083** (1 of 12) |
| arithmetic, fraction → strict | 0.74 → **0.08** |

A single agent produces a fully feasible schedule in one episode out of twelve
and gets every capacity constraint right in one out of twelve. Under `fraction`
that reads as 0.842 and looks near ceiling. **The Phase 1 gate should therefore
be evaluated on all three metrics**, not on `fraction` alone — a conclusion that
applies retroactively to the medium result in §4.1, whose `feasible` gap
(0.250 → 0.417, teams 1.7× as often feasible) is the largest signal in that
corpus and was invisible in the headline number.

Two caveats for the paper. An effect of *d* = 0.36 needs roughly 120 episodes
per group for 80% power, so a real effect of that size cannot reach significance
at N = 12. And a `hard` team result will be read against a solo baseline of
1/12, where single episodes move the proportion substantially.

### 4.3 Throughput characterisation

Batch sweep, 1600-token context, 192 new tokens:

| batch | aggregate | per sequence | peak allocated |
|---|---|---|---|
| 8 | 72.6 tok/s | 9.1 | 16.8 GiB |
| 16 | 94.1 tok/s | 5.9 | 18.3 GiB |
| 24 | 102.8 tok/s | 4.3 | 19.8 GiB |
| **32** | **108.1 tok/s** | 3.4 | 21.4 GiB |
| 48 | 71.1 tok/s | 1.5 | 24.4 GiB ← paging |

Production shape (13 sequences, 1300-token context, **1024** new tokens):
**54.9 tok/s, 4.22 decode steps/s, 19.3 GiB peak, 242 s per pass.**
`cache_implementation="static"` failed — routes through inductor, no MSVC
toolchain on this machine.

**KV cost:** ~123 KiB/token measured, against 144 KiB/token predicted by the GQA
geometry (36 layers × 8 KV heads × 128 dim × 2 tensors × 2 bytes). An earlier
figure of ~490 KiB/token was taken inside the paging regime and set the token
budget three times too small.

**Two structural limits, neither a misconfiguration:**

1. **Prompt size caps the batch.** Every turn carries the ~2500-token task brief
   (`hard`: larger), growing to ~5000 by round three. `budget / (context +
   max_tokens)` is how many turns share a forward pass: 8 early, under 4 late.
2. **A pass runs until its slowest member stops.** With a 1024-token cap and
   ~627 tokens of useful output, roughly 40% of decode steps generate padding
   for sequences that already finished.

Both are exactly what vLLM's prefix caching and continuous batching solve;
neither is reachable through `transformers`. vLLM has no Windows build, and the
WSL2 instance on this machine is broken (`Wsl/Service/E_UNEXPECTED`, persisting
after `--shutdown`), needing a reboot or reinstall not undertaken unilaterally.
Realistic expected gain: **3–5×**, not the 5–10× claimed earlier.

### 4.4 Judge availability — pivot, and what it costs

PLAN.md assumed a paid frontier judge (~$5 for 14k messages). No paid API was
available. Measurement of the free Gemini tier found the binding limit is
**per-day, not per-minute**:
`GenerateRequestsPerDayPerProjectPerModel-FreeTier = 20 requests/day/model` —
roughly three coded messages short of a *single* episode. Bulk coding through it
is not slow, it is impossible.

**Substitute:** code the full corpus with the local 8B (coding replies are ~8
tokens, so batched on the same card this is minutes and costs nothing), then
spend the daily free quota of a frontier model on a subsample and report
Cohen's κ between them, with a bootstrap interval.

This does not make the local judge as good as a frontier one — **it measures how
good it is**, which is what the κ was always for. If κ falls far below the 0.78
reported by 2604.00026, the honest conclusion is that Phase 4's correlation is
untrustworthy on these labels, and that is itself reportable.

**Caveat to state wherever the number appears:** both judges are Google models
(2.5-flash and 3-flash-preview), not two families. Correlated errors are likelier
between them, so their agreement is an **upper bound** on true reliability.

### 4.5 Propagation diagnostic — a prediction that was wrong

Plain excision was expected to give the *largest* ablation drop, since it blocks
compensation entirely. Measured against the mock it gave nearly the *smallest*:
~0.002 against live drops of 0.03–0.23, a hundredfold underestimate.

**Cause:** content propagation. Agents restate the whole working answer every
turn, so a contribution is copied into everyone else's messages almost as soon as
it is made. Deleting the originating messages removes the words but not the
content. Read naively this reports "agent contributed nothing" when the truth is
"this measurement does not work on this transcript."

`ablation.propagation_index()` measures how much of an agent's content is echoed
by others later, and decides which frozen mode to trust. Mock propagation index:
0.52–0.54, which correctly triggers the warning to prefer `frozen_replay`.
Whether real 8B agents restate as aggressively is an open question for the real
corpus — but the diagnostic now answers it rather than assuming.

---

## 5. Instrument validity work

Disproportionate effort went here, and in retrospect that was correct: four of
the day's failures produced plausible wrong data rather than errors.

**Null-world validation.** The pipeline is run against a mock with no
specialization; if it reports specialization there, it is manufacturing results.
Note the correct null for diagonal dominance with 4 agents is **0.25**, not 0.00
— with four agents and four components a coincidental hit is a one-in-four event
per column. The README stated 0.00 and was corrected.

**Double-centering both matrices in convergent validity.** Without it a merely
verbose agent produces apparent convergence between transcript labels and causal
profile.

**Identity stripping in behavioral coding.** The judge sees one message at a
time with the agent's self-label removed, so it cannot invent a consistent
character for an agent across an episode — the precise artifact the project
exists to distinguish from a real effect.

**Judge-failure abort.** `record_error` raises `JudgeUnavailable` after 8
consecutive failures, so a judge silently returning `other` for every message
cannot quietly produce a corpus of default labels.

**Seed-blind arm skipping.** `SymmetryBreaking.NAME_ONLY` and `NAME_SEED` differ
*only* by per-agent seed. `transformers.generate` samples a whole batch from one
global RNG, so on this backend they are the same condition. Backends now declare
`honors_request_seed`; the arm is skipped where it cannot be distinguished,
which avoids both wasted card time and reading a difference into sampling noise.

**Ownership as share, not volume.** Otherwise the most talkative agent owns
every component.

---

## 6. Reproducibility notes for the paper's methods section

| | |
|---|---|
| CPU / RAM | Intel i9-14900, 24 cores / 32 threads; 128 GB |
| GPU | NVIDIA RTX 4500 Ada, 24 GB GDDR6, driver 595.95 |
| OS | Windows 11 Pro 26200 |
| Python / torch / transformers | 3.10.11 / 2.5.1+cu121 / 5.1.0 |
| Model | Qwen3-8B, bf16, 15.3 GiB resident, `enable_thinking=False`, SDPA attention |
| Team | 4 agents × 3 rounds, temperature 0.8, top_p 0.95, `max_tokens` 1024 |
| Batching | token-budgeted, length-sorted, 30000 padded tokens, `memory_fraction` 0.90 |

**Sampling reproducibility is per batch, not per request.** vLLM seeds each
sequence independently; `model.generate` draws the whole batch from one global
RNG, so an episode's text depends on which other episodes happened to batch with
it. Member seeds are folded into a single per-batch seed so an identical batch
replays identically, but arbitrary batch composition does not. This does not
affect the analysis — transcripts store every message verbatim and are re-read,
never regenerated — but reproducing a *specific* episode bit-for-bit requires
`max_batch_size=1`. The setting is recorded in every run's resolved config.

**No agent framework.** Plain Python orchestration. Every mainstream agent
framework ships opinionated role scaffolding in its prompts; using one would
silently plant the roles the study claims to observe emerging. This is a
validity requirement, not a style preference.

**The task brief contains no role language**, no suggested decomposition, and no
hint that division of labor is expected. `tests/test_render.py` asserts on every
preset that `ground_truth` and `planted_errors` never reach agent-visible text.

---

## 7. What remains

1. **Does `hard` clear the Phase 1 gate?** Early read from solo episodes;
   full gap ~2 h into the run. Decision point before the ablation grid commits.
2. Ablation grid: live, frozen_replay, frozen_excise, random_message, capacity.
3. Interaction strength, diagonal dominance vs chance 0.25, fungibility
   `Δ(frozen_replay) − Δ(live)`, propagation index on real transcripts,
   mixed-effects joint Wald test.
4. Behavioral coding with the local 8B; κ against the frontier subsample.
5. Convergent validity: do transcript labels predict causal profiles?
6. Extend N via resume; Phase 4 robustness across model families and team sizes.
7. Preregister Phases 3–4 before running them.

**A weak correlation at step 5 is not a failed project.** It is the strongest
version of the original thesis and an independent replication of *Agents that
Matter*' finding that introspective judgment diverges from ablation.

---

## 8. Lessons worth generalising

1. **Prefer failures that are loud over failures that are cheap.** The memory
   cap made the system fail *more often* and was strictly an improvement,
   because silent paging cost more than every OOM combined.
2. **A metric that is healthy in the failure mode is not a metric.** GPU
   utilization read 100% whether the card was computing or thrashing. Power draw
   and PCIe traffic separated them; utilization never could.
3. **Instrument settings are part of a measurement's identity.** The calibration
   that chose `medium` was not wrong arithmetic — it correctly measured a
   configuration nobody recorded alongside its output.
4. **Zeros need provenance.** "The team failed" and "the harness failed" are
   the same number and opposite claims. Any pipeline that averages scores needs
   to know which it has.
5. **Bugs that bias with the phenomenon are worse than bugs that add noise.**
   Truncation correlating with verbosity, where verbosity *is* the emergent
   behaviour under study, manufactures the effect being tested for.
6. **Test the path that runs.** A test for length-sorted batching passed while
   never executing it, because the fake backend overrode the method containing
   it.
7. **Measure the shape you will run.** A benchmark at 192 new tokens said the
   configuration was healthy; the same configuration at 1024 was not.

---

## Appendix A — Commit ledger

<!-- 44 commits; `git log --reverse --format='%h %ad %s' --date=short` -->

| Day 1 | |
|---|---|
| `728230e` | Initial commit |
| `f74128d` | Project Plan |
| `cb09886` | Task generator and per-component grader |
| `d548b9e` | Prompt rendering, answer parsing, message protocol |
| `86b6568` | Orchestrator, transcripts, mock backend, interaction analysis |
| `1597f52` | Ablation modes, controls, propagation diagnostic |
| `7a06d1e` | OpenAI-compatible backend and parallel runner |
| `d8bfd6c` | CLI, config system, example configs, README |

| Day 2 — enabling the GPU | |
|---|---|
| `aaca83b` | Correct the hardware section; lower Python floor to 3.10 |
| `90b83b2` | In-process CUDA backend |
| `47296a2` | Calibrate against one agent as well as N; survive loop reuse |
| `21cff55` | Phase 2–4 analysis, chained pipeline, **resume fix** |
| `4d96ba7` | Bound batches by tokens, not request count |
| `cadad85` | Per-mode drops and the fungibility metric |
| `d9c2a42` | Pipeline calibrates and picks its own operating point |
| `9294b89` | Skip expandable_segments on Windows |
| `96fb540` | Mixed-effects interaction test |
| `be9815b` | Run calibration cells concurrently |
| `6509900` | Correct the null world's dominance figure (0.00 → 0.25) |

| Day 2 — judge pivot | |
|---|---|
| `e10417c` | Gemini judge backend |
| `0d859e9` | Coding targets specific conditions |
| `c68ca73` | Stop coding when the judge stops answering |
| `999c763` | Code episodes concurrently for local judges |
| `5266202` | Record the judge plan against measured free-tier limits |
| `51f96a4` | README brought up to date |

| Day 2 — validity | |
|---|---|
| `c90d01b` | Phase 1 gate moved inside the run as a solo condition |
| `3b430e1` | Sort batches by context length before chunking |
| `b97078b` | **Measure ablation drops against baseline episodes only** |
| `cc519a5` | Bootstrap interval alongside Cohen's κ |
| `e2b669d` | Report mean batch size at end of run |
| `b9cc553` | Budget batches on measured memory |
| `2345d1a` | Skip the symmetry arm a seed-blind backend cannot distinguish |

| Day 2 — the long debugging arc | |
|---|---|
| `2ceb764` | **Do not let the token cap write zeros into the ablation grid** |
| `d8616dc` | Batch budget from a sweep, not head-count arithmetic |
| `9633ec0` | Chain the analysis behind the pipeline |
| `8d81786` | Replace throughput figures measured while paging |
| `7817e42` | Report throughput mid-stage |
| `ca7c9d6` | Measure the production shape; stop blaming the allocator |
| `ab070dc` | **Recover from CUDA OOM outside the except block** |
| `72e8536` | Detect the pipeline by command line |
| `cb28879` | Document the memory cap and OOM-recovery trap |
| `55cdc4a` | Count a pass's tokens before reporting its rate |
| `5ea7875` | **Move the operating point to `hard`** |
| `2c9aa07` | **Record the Phase 1 gate result** |

## Appendix B — Module inventory

| Module | Lines | Role |
|---|---|---|
| `cli.py` | 1292 | Commands: calibrate, baseline, ablate, pipeline, analyze, code, kappa, converge |
| `backends/hf_local.py` | 607 | CUDA generation, token-budgeted micro-batching, memory cap, heartbeat |
| `analysis/coding.py` | 520 | Behavioral coding, identity stripping, κ, differentiation vs null |
| `tasks/generator.py` | 311 | Instance generator with tagged constraint classes |
| `backends/mock.py` | 309 | Full-pipeline debugging and null-world validation |
| `ablation/modes.py` | 329 | live, frozen_replay, frozen_excise, capacity, random_message |
| `backends/gemini_judge.py` | 243 | Free-tier judge with per-day quota discrimination |
| `analysis/mixed.py` | 218 | Mixed-effects interaction, joint Wald test |
| `tasks/schema.py` / `grader.py` / `render.py` | 214 / 191 / 209 | Components, per-component grading, prompt/answer |
| `analysis/convergent.py` | 192 | Convergent validity with double-centering |
| `analysis/integrity.py` | 191 | Instrument-failure classification |
| `analysis/interaction.py` | 160 | Ablation matrix, double-centering, dominance |

**Tests: 277 across 12 files** (~2,450 lines), notably
`test_instrument_validity.py` (null-world checks) and `test_integrity.py`
(harness-zero vs team-zero).
