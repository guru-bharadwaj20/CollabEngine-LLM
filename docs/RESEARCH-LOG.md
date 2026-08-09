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

### 3.8 One chunk's failure blanked the whole batch — the corpus-costing bug

**Symptom.** A gate check reported 24 of 36 episodes as instrument failures.
Every four-agent episode in the corpus was affected.

**Diagnosis.** `RuntimeError: CUDA OOM generating a single sequence` on **96 of
144** agent turns. A batch is split into chunks by token budget, each run as its
own forward pass. When one chunk raised, the exception escaped
`_generate_batch`, and the queue worker's handler failed the *entire group* — so
one round-three context too long to fit blanked every turn batched alongside it,
including turns that had already generated successfully.

**Why it was expensive rather than merely annoying.** An errored turn produces
an empty message and the episode still completes, still parses an answer from
whatever turns survived, and still receives a plausible score. Nothing crashed.
The corpus looked finished. Its headline number (§4.1) was published before
anyone asked how many turns were in it.

**Fix.** Chunks are isolated — a failure produces error responses for that chunk
only. A single sequence that cannot fit returns an error rather than raising,
for the same reason. `_release()` now reclaims between chunks: Python frees the
tensors when a frame returns, but the caching allocator keeps the blocks and
`set_per_process_memory_fraction` counts what is *reserved*, so a stage drifts
into OOM over time with nothing actually leaking. That drift is the likely
reason single sequences were failing at all, since one 6000-token sequence needs
under a gigabyte of KV against several free.

**A near-miss worth recording.** The `_release` helper was initially added by a
string replacement that silently matched nothing. All 283 tests still passed,
because the fake backend used in tests overrides the method that calls it — the
real run would have hit `NameError` on its first chunked batch. It was caught
only because a test was written that imported the helper directly. This is the
same shape as the length-sorting test in §3.7: **a test that exercises a fake
cannot tell you the real path exists.**

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

### 3.9 The card is shared, and the guard that knew it failed open

2026-08-09, 10:03. Phase 2 behavioural coding was launched onto what looked
like an idle card. It was not idle: a *different account on the same
workstation* (`Student2`) had started an unrelated Qwen3-VL job twenty seconds
earlier — `profile_e8_efficiency.py`, a parent plus eight worker processes,
holding 17.3 GiB of the 24 GB and writing to a `FINAL/` results directory under
`--resume`.

Qwen3-8B in bf16 needs 15.3 GiB for weights alone. The two do not fit. Nothing
raised an error, because this is §3.4 again: WDDM paged the working set to host
RAM over PCIe. `nvidia-smi` read 100% utilisation and 24,118 MiB in use, which
is what a card doing useful work looks like. The tells were the ones §3.4
already identified — 60 W of a 210 W budget, and `dmon -s put` showing a
sustained 8 GB/s in and 11 GB/s out with the memory controller at 0%.

**Why it is worth a section of its own.** §3.4 was diagnosed as *our own*
second model colliding with the pipeline, and the fix was written to match that
diagnosis: `followup.sh` waits for our pipeline process to disappear. That
guard is correct and would have passed here, because our pipeline had exited at
09:36. The mental model — "the danger is two of our jobs overlapping" — was one
special case of "the danger is 15.3 GiB not fitting in what's left", and the
narrower model had been encoded into the tooling as though it were the general
one.

`scripts/queue-judge.sh` replaces the process check with a free-VRAM check:
≥18 GiB, stable across three 60-second samples, naming the compute PIDs that
hold the card while it waits.

The three-sample requirement was carried over from `followup.sh`'s `GONE_LIMIT`
as a precaution and was vindicated within the hour. At 12:43:27 the card read
24,288 MiB free — the foreign job had finished a stage — and sixty seconds
later a new process of theirs held it again:

```
[12:43:27] free 24288 MiB (1/3)
[12:44:27] card reclaimed; restarting the count
[12:44:27] free 7237 MiB, need 18000; compute pids: ...,17004
```

A single-sample gate, or a two-sample one, would have started a 15.3 GiB model
into that gap. **"Free right now" and "free" are different measurements when
the resource is contended**, and the difference is one poll interval wide. It **waits rather than kills**. The foreign job is
mid-grid with `--resume` and is not ours to stop, and this is worth stating
explicitly in a log that otherwise treats the GPU as a private resource.

**The guard's first version failed open.** `nvidia-smi` emits CRLF on Windows;
a trailing CR survives `tr -d ' '` invisibly; `$(( 24570 - 17348<CR> ))` is an
arithmetic syntax error; `$free` was left unset; the loop fell through and
started the judge into the busy card at 10:09:39 — the precise outcome the
guard existed to prevent, thirty seconds after the guard was written. Fixed by
stripping CR, validating both figures are digits, and treating anything
unparseable as *busy*.

The general form is worth keeping: **a guard that fails open is worse than no
guard, because it gets trusted.** Without it the card would have been checked
by hand. With it, the check was delegated to nine lines of shell that returned
"clear" on a parse error. This is the same shape as §3.8 (an errored turn
grading 0.0) and the retraction in §4.1 (a filter that was bypassed): in each
case the failure produced a *plausible permissive value* rather than a stop.

### 3.10 A third corpus lost to the same cause, and the fix that should have come first

The first `xhard` run reported `24 done, 0 failed` and was worthless. That
count is episodes that *completed*, not turns that generated:

```
finish_reason: {'stop': 44, 'length': 46, 'error': 90}

condition   eps  unusable   trunc
baseline     12        12     19%     <- the entire team arm
solo         12         7     53%
```

**19 of 24 episodes unusable, 90 of 180 agent turns errored.** The same shape as
the retracted corpus in §4.1, five months of lessons later.

The OOMs began at **5223-token prefills**, on a card that had run 10,661-token
prefills at `hard` under an identical `memory_fraction`. Throughput fell from 24
to 9 tok/s across the run.

**My first diagnosis was wrong, and I acted on it before testing it.** The
shared card (§3.9) had corrupted two corpora already, the symptom matched, and
I concluded the other account had reclaimed the GPU mid-run. I then built a
retry-with-backoff on that assumption, committed it, wrote it up here as fact,
and relaunched. The relaunch OOMed at the same 5223 tokens **with the card
empty** — 2% utilisation, 16 GB held by our own process, no other job. The
evidence that refuted it took one screenshot of Task Manager.

**The real cause, measured directly.** Loading the model exactly as the run
does and walking prefill sizes up:

| prefill | allocation attempted |
|---|---|
| 1 × 5,223 tok | 834 MiB *on top of 22.37 GiB already held* |
| 1 × 8,000 tok | **7.63 GiB** |
| 1 × 12,000 tok | **17.17 GiB** |
| 3 × 5,223 tok | 9.76 GiB |

The failing allocation scales with **prompt tokens**, at roughly 1–1.5 MB each,
and is independent of batch size — 3 × 5,223 and 1 × 12,000 fail alike. That is
the logits tensor: Qwen3's vocabulary is 151,936, and the prefill forward
materialises a distribution for *every* position, upcast, plus a copy. The KV
cache — the thing every memory knob in this project tunes, and the thing three
sections of this log reason about — was never the constraint.

`logits_to_keep=1` does not help: passed to `generate` **or** directly to
`forward`, the attempted allocations are byte-identical. There is no fix from
our side of the API.

**Consequence: `xhard` is not runnable on this card at bf16.** Its prompts
*start* near 5,200 tokens, which is already past the ceiling with 15.3 GiB of
weights resident. Not a tuning problem — `max_batch_tokens`, `max_batch_size`
and `max_model_len` all leave the per-token cost untouched. The tier needs
quantised weights (which would confound the difficulty curve it exists to
extend) or a larger card.

**Why `hard` and `medium` were fine.** Their briefs render at 1704 tokens
against `xhard`'s 2520. The whole tier sat on the far side of a cliff that the
two measured operating points sit comfortably below — which is also why nothing
before it exposed the bug.

**The retry stays, with its justification corrected.** It is a reasonable safety
net for genuine contention, which does happen here, and it costs only time. But
it did not address this failure and was never going to: against a hard ceiling
it burns two minutes per turn and then records the error anyway, which is
exactly what it did while the card sat at 2% for several minutes. `oom_retries`
is lowered accordingly.

**The lesson is not about memory.** It is that a diagnosis matching a familiar
pattern is not a tested diagnosis. Contention was real, recent, and had caused
this exact symptom twice — which made it the most available explanation and the
one least likely to be checked. The check cost one command.

**The test drives the real `_generate_batch`.** Every other test in
`test_hf_local.py` goes through `FakeHF`, which overrides that method — the
subclass exists to avoid touching CUDA. An earlier OOM fix passed all 283 tests
while doing nothing for exactly that reason (§3.3), so this one stubs the
tokenizer and model and drives the real code path, asserting the backoff
schedule and that the recorded error says retries were spent.

---

## 4. Results

### 4.1 Phase 1 gate at `medium` — **RETRACTED, the team arm was not a team**

> **This result was reported and is withdrawn.** The numbers below were computed
> from 12 four-agent episodes in which **96 of 144 agent turns failed to
> generate** — `RuntimeError: CUDA OOM generating a single sequence`. Two thirds
> of the team never spoke. What was measured was not a four-agent team scoring
> 0.905; it was one or two agents with most turns empty, and the flat gap has an
> obvious alternative explanation that has nothing to do with difficulty.
>
> | condition | n | mean | sd | |
> |---|---|---|---|---|
> | baseline (nominally 4 agents) | 12 | 0.905 | 0.095 | **96/144 turns errored** |
> | solo (1 agent) | 12 | 0.879 | 0.100 | 0/36 errored |
>
> The failure is §3.8: one chunk of a batch raising took down every turn batched
> alongside it. An empty turn grades 0.0, which is indistinguishable in the means
> from a team that answered badly.
>
> **`analysis/integrity.py` caught this correctly** — it flagged all 24
> four-agent episodes as instrument failures. The number was published anyway
> because it was computed with an ad-hoc script that filtered only on
> `malformed`, before the filter was extended to cover errored turns. The lesson
> is not that the filter was missing; it is that *a result computed outside the
> pipeline's own validity checks is not a result*.
>
> **Consequences.** The decision to abandon `medium` for `hard` rests on this
> retracted comparison and is therefore unsupported. The solo arm (0.879, clean)
> stands. Whether `medium` actually fails the gate is **unknown** and needs 12
> uncorrupted four-agent episodes to answer — cheap to obtain, since the solo
> arm does not need regenerating.

### 4.1b Phase 1 gate at `hard` — **fails**

The first valid team-vs-solo measurement the project has produced. Qwen3-8B,
`hard`, `max_tokens` 1024, `max_model_len` 12288, `memory_fraction` 0.95 —
the first configuration in which a four-agent episode survives round three
intact.

![Phase 1 gate at hard](figures/gate.png)

*Left: every usable episode, four arms, `fraction` metric. Right: the team−solo
gap under each metric with its 95% bootstrap interval. Regenerate with
`python scripts/figures.py`.*

The measurement was taken twice. The first pass ran on 9 team episodes because
three had died on long-context OOM; since that failure mode selects for long
transcripts and only team transcripts get long, the surviving nine were a
suspect subset. The tainted episodes were regenerated at `memory_fraction`
0.95 before the numbers below were recomputed. **Both readings are kept here
— the difference between them is the finding.**

| metric | solo (n=12) | team, n=9 (first pass) | team, n=11 (regenerated) |
|---|---|---|---|
| `fraction` | 0.842 | 0.862 (*d* +0.31) | 0.871 (*d* +0.40) |
| `strict` | 0.490 | 0.508 (*d* +0.09) | 0.529 (*d* +0.17) |
| `feasible` | 0.083 | 0.000 (*d* −0.43) | 0.091 (*d* +0.03) |

The gate verdict is unchanged, but **one claim from the first pass is
withdrawn.** It read "on the strictest reading the team is *worse* — no team
episode produced a fully feasible schedule, while one solo episode did." That
was subset bias, exactly of the kind predicted: with the two long episodes
restored, one of them *is* feasible, and `feasible` moves from −0.083 to
+0.008. A one-episode gap in a 12-episode arm was never an effect, and the
episode was missing because it was long.

Significance, computed on the regenerated arms (10,000-permutation null on the
arm labels, BCa bootstrap for the interval):

| metric | gap | perm *p* | 95% CI of the gap |
|---|---|---|---|
| `fraction` | +0.029 | 0.373 | [−0.029, +0.088] |
| `strict` | +0.039 | 0.706 | [−0.151, +0.226] |
| `feasible` | +0.008 | 1.000 | [−0.250, +0.273] |

Every interval spans zero. Four agents over three rounds do not measurably beat
one agent on any metric.

Other arms, same corpus: `symmetry:name_seed_scratch` 0.851 (n=12),
`fixed_order` 0.854 (n=10). Both sit within a hundredth of solo. An earlier
note here read `fixed_order` at 0.784 and wondered aloud whether randomised
turn order helps; that was n=4 and it evaporated at n=10.

**A variance result, tested and withdrawn.** The regenerated baseline arm has
roughly a third of solo's variance (sd 0.050 vs 0.095, ratio 3.6, permutation
*p* = 0.019 one-sided) — teams scoring no higher but more consistently is a
plausible mechanism and would have been the one positive result of Phase 1. It
does not survive the other two team arms. `fixed_order` (sd 0.100) and
`symmetry` (sd 0.093) are four-agent teams too, and both are as variable as
solo; pooling all 33 team episodes against solo gives a variance ratio of 1.36
at *p* = 0.21. One arm in three is a fluke, and the direction was chosen after
seeing the data. Recorded because the near-miss is instructive: the arm with
the tight spread is also the arm that lost an episode to OOM, and OOM removes
the tail.

**Two caveats remain.** `feasible` is one episode in each arm, so its CI is an
artefact of a binary outcome at n≈11, not a measurement. And at this n only a
very large effect could reach significance; the design cannot distinguish "no
benefit" from "a benefit too small to see here".

**What it does establish.** Across both difficulties, a 50% increase in
instance size moved a single agent from 0.879 to 0.842 and moved the team
nowhere. The task does not reward collaboration at this model scale, which is
the Phase 1 stop condition, and the ablation grid was not run. An
agent × component interaction measured where the team contributes nothing would
be measuring the noise floor.

Phase 2 remains open and worth running on this corpus: **behavioural
differentiation does not require a performance benefit.** Agents may divide
labour visibly while the division buys nothing, and that combination —
observable roles, no causal contribution — is the strongest form of the
project's original thesis rather than a null result.

#### 4.1c The withdrawn `medium` write-up, kept verbatim

What follows is the text §4.1 carried before the retraction, left in place
because how a wrong result reads while you believe it is part of the record.
Every number in it is contaminated by §3.8. **Do not cite it.**

> Qwen3-8B, 12 episodes per condition, identical instances, `max_tokens: 1024`:
>
> | condition | n | mean | sd |
> |---|---|---|---|
> | baseline (4 agents × 3 rounds) | 12 | 0.905 | 0.095 |
> | solo (1 agent × 3 rounds) | 12 | 0.879 | 0.100 |
> | symmetry: name_seed_scratch | 10 | 0.851 | 0.098 |
>
> **Team − solo gap: +0.026, against a standard error of ~0.029.**
> Indistinguishable from zero.
>
> PLAN.md Phase 1 makes this a stop condition: *"find the band where the task is
> hard enough to need collaboration but not so hard the model floors out. If
> there is no such band, stop and redesign."* It stops Phase 3 at this operating
> point rather than stopping the project — an ablation grid run where the team
> contributes nothing measures its own noise floor.
>
> **This is a real finding and belongs in the paper**, both as a negative result
> about 8B agent teams on constraint-satisfaction tasks, and as the difficulty
> curve Phase 1 asked for. The corpus is archived, not deleted.
>
> **Read this together with §4.2.** Under `fraction` the medium gap is +0.026;
> under whole-instance feasibility the same transcripts give 0.250 vs 0.417 --
> teams produce a fully feasible schedule 1.7x as often. The gate verdict does
> not change (that gap is still inside its error at n=12), but the headline
> metric was hiding the largest signal in the corpus.

Note what the confident paragraph asserts: a negative result that "belongs in
the paper", read alongside a §4.2 comparison in which teams look 1.7× better on
feasibility. Both readings came from a corpus where two thirds of the team's
turns were empty. The write-up was not careless about statistics — it quoted a
standard error and refused to over-claim the gap. It was careless about
*whether the episodes were episodes*, which no amount of downstream rigour
recovers.

Action taken at the time: operating point moved to `hard` (24 jobs vs 16, 6
workers vs 5, 6 exclusions vs 4, 5 synthesis constraints vs 3, capacity slack
1.1 vs 1.2, value floor 0.72 vs 0.65). That move was made on the retracted
comparison, so it was unsupported when made — and §4.1b then failed the gate at
`hard` independently, which makes it the right move for the wrong reason.

### 4.1d `medium` re-measured — it fails too, and the bias ran the other way

The hole left by the §4.1 retraction, filled. 12 fresh team episodes on the
fixed harness (`configs/local-gpu-medium.yaml`, identical to the `hard` config
but for `name` and `difficulty`), against the original medium solo arm, which
needed no regenerating: a solo episode is 3 turns peaking near 5700 tokens, so
neither the 8192 truncation nor the 0.90 cap that destroyed the old team arm
could reach it.

**The first pass had 3 CUDA OOMs and looked like the best result in the
project:**

| metric | solo (n=12) | team (**n=9**) | gap | *d* | *p* |
|---|---|---|---|---|---|
| `fraction` | 0.879 | 0.907 | +0.028 | +0.27 | 0.560 |
| `strict` | 0.616 | 0.718 | +0.102 | +0.31 | 0.490 |
| `feasible` | 0.250 | **0.667** | **+0.417** | **+0.92** | 0.086 |

Teams producing a feasible schedule 2.7× as often, at *d* = 0.92 — by a wide
margin the largest effect ever measured here, and enough to restart the
ablation grid. It was not reported, because n=9 with three OOM dropouts is the
same defect that produced the withdrawn claim in §4.1b, and a defect does not
become acceptable when the number it produces is the one you want.

**The three OOMs turned out to be contention, not a ceiling.** They fired at
prefills of 4476, 7180 and 8002 tokens — far too small to fail against 22.8 GiB
at `memory_fraction` 0.95. The shared card (§3.9) was reclaimed by the other
account during the 45-minute run. Regenerated on an empty card, all three
completed with **zero** instrument failures.

**With 12/12, the effect collapses:**

| metric | solo (n=12) | team (n=12) | gap | *d* | perm *p* | 95% CI |
|---|---|---|---|---|---|---|
| `fraction` | 0.879 | 0.874 | **−0.005** | −0.05 | 0.915 | [−0.096, +0.084] |
| `strict` | 0.616 | 0.625 | +0.009 | +0.03 | 0.939 | [−0.262, +0.281] |
| `feasible` | 0.250 | 0.500 | +0.250 | +0.53 | 0.406 | [−0.167, +0.583] |

`feasible` is 3/12 against 6/12 — twice as often, three episodes versus six, and
nowhere near significance. **`medium` fails the gate as well.**

**Why this is the cleanest confound demonstration in the log.** The three
dropped episodes were not a random nine-twelfths:

| | fraction | feasible | mean chars |
|---|---|---|---|
| the 9 that survived OOM | 0.907 | 0.667 | 8381 |
| the 3 that OOMed | 0.775 | **0.000** | 8793 |

The episodes the card could not finish were **longer and worse**. Not one of
them produced a feasible schedule. §4.6 established that every resource limit
lands on the arm being measured; this adds the part that section got wrong by
omission — **the direction is not fixed**. OOM removes long episodes, and
whether that flatters or damages the arm depends entirely on whether long
episodes happen to score well. At `hard` it hid a real team episode and made
the team look worse. At `medium` it hid three bad ones and made the team look
dramatically better. The same mechanism, opposite signs, and neither is
detectable from the surviving data alone.

**The difficulty curve, complete.** Solo 0.879 at `medium` and 0.842 at `hard`;
team 0.874 and 0.871. A 50% increase in instance size moves one agent by 0.037
and four agents by 0.003, and at neither point does the team beat the solo
baseline on any metric. Two operating points, both failing, is a far stronger
negative than one — and it says the Phase 1 gate failure is a property of the
task and the model scale, not of a badly chosen difficulty.

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

![Batch-size sweep](figures/throughput.png)

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

### 4.6 Every resource limit lands on the arm being measured

Four separate failures tonight damaged the team arm and left the solo arm
untouched. That is not coincidence, and it generalises past this project.

**Quantified, once the ceiling in §3.10 was known.** The prefill ceiling on this
card sits near 6,500 tokens. Measuring every agent turn's context length in the
corpora already on disk:

| corpus | arm | turns | median | p90 | max | **over 6.5k** |
|---|---|---|---|---|---|---|
| `hard` | solo | 72 | 3,554 | 3,917 | 4,039 | **0.0%** |
| `hard` | team | 432 | 5,730 | 7,584 | 10,619 | **28.9%** |
| `medium` | team | 144 | 3,851 | 5,021 | 5,727 | **0.0%** |

**At `hard`, 28.9% of team turns sit above the hardware ceiling and 0% of solo
turns do.** Not a subtle bias — a censoring threshold that only one arm can
cross, and the arm it censors is the one under test. Regeneration cannot fix
it: the context length is a deterministic property of the instance and the
transcript, so an episode over the line is over it every time. The three
permanently-failing `hard` episodes were never a fluke.

**This re-weights the two operating points against each other.** `medium` is
completely clean — its longest team turn is 5,727 tokens, comfortably under the
ceiling for both arms — so §4.1d's null is measured on an uncensored
distribution and stands as written. `hard` is not: its team mean of 0.871 is
computed on a distribution with the upper tail removed.

**The direction of that censoring argues the negative result is if anything
understated.** At `medium`, the episodes lost to OOM scored 0.775 against the
survivors' 0.907 and produced zero feasible schedules (§4.1d) — long team
episodes are *worse*, not better. If that holds at `hard`, censoring the long
tail **flatters** the team arm, and the team still fails to beat solo. The
conclusion survives its own worst-case correction, which is the strongest form
it can take on this hardware.

**What it costs going forward.** Any operating point whose team contexts cross
6,500 tokens cannot be measured cleanly on this card, regardless of batch
settings. That rules out `xhard` entirely (§3.10) and puts `hard` on notice.
`medium` is the largest instance size this hardware can evaluate without
censoring the team arm — a statement about the equipment, not the hypothesis,
and one that belongs in the methods section rather than the results.

| limit | solo (3 turns, ~5.7k tok) | team (12 turns, ~10k tok) |
|---|---|---|
| `max_tokens` 512 | mild truncation | penalised the verbose agents |
| right-side truncation at 8192 | never triggered | deleted the answer contract |
| single-sequence OOM at 0.90 | 0/12 affected | 11/12 affected |
| single-sequence OOM at 0.95 | 0/12 affected | 3/12 affected |

The mechanism is the same each time: **on fixed hardware, whatever limit you
reach is reached first by the longest transcripts, and transcript length is a
function of how much the agents collaborate.** The confound is therefore
correlated with the independent variable by construction, and it points in one
direction — against the team.

Measured directly on the stage-1 corpus before the tainted episodes were
regenerated:

| | n | mean transcript | median |
|---|---|---|---|
| clean | 28 | ~3054 tok | 2986 |
| tainted | 8 | ~3478 tok | 3319 |

The episodes lost were 14% longer on the mean, and that understates it, because
a tainted transcript is cut off at the point of failure so its true length was
greater. The failing instance seeds recur across conditions -- `:6`, `:8`,
`:11` appear in both `baseline` and `fixed_order` -- confirming these are
instances that generate long transcripts rather than random victims.

**Consequence for the method.** A team-vs-solo comparison on constrained
hardware must report per-arm instrument-failure counts as standard output, not
inspect them when a number looks surprising. Dropping the affected episodes is
not sufficient either: it leaves a non-random subset. They have to be
regenerated, which is only possible because instances are deterministic in
`(seed, difficulty)`.

### 4.7 Propagation on real transcripts, and a gradient I could not establish

Run while the card was held by another user (§3.9), on transcripts already on
disk. Three results, in descending order of how much they can be trusted.

**1. Excision is untrustworthy on real data, as it was on the mock.** The
propagation index — the fraction of an agent's distinct content tokens that
reappear in later messages by *other* agents — is **0.633** across 132
agent-episodes (sd 0.216, range 0.150–1.000), and is flat across conditions
(baseline 0.625, fixed_order 0.627, symmetry 0.645). Nearly two thirds of what
an agent contributes is already duplicated elsewhere in the transcript by the
end of the episode.

This confirms on real transcripts what §4.5 found on the mock, and it settles a
question that was left open there: `frozen_excise` is not measuring an agent's
contribution on this corpus, because deleting an agent's messages does not
delete the agent's content. **`frozen_replay` is the mode to trust.** Had the
ablation grid run, reporting excision drops without this check would have
produced a corpus-wide "agents contribute nothing" artefact.

**2. The propagation index is strongly length-dependent — an instrument
caveat.** It correlates with the author's message length at **r = −0.70**
(unique words per message). This is close to mechanical: the index is a
*fraction* of distinct tokens, and a longer message has more distinct tokens
that must all be echoed to score highly. Anyone using this index as a measure
of influence is partly measuring brevity. It remains fit for its actual purpose
— deciding whether excision is safe — because that only needs the population
mean, not per-agent comparisons.

**3. A per-agent gradient that does not survive its own omnibus test.** Agents
appeared to differ by name: in randomised-order episodes, `A1`/`A2` scored
0.774/0.723 on propagation against `A3`/`A4` at 0.494/0.551, and wrote shorter
messages (64/73 unique words vs 86/85). Speaking position was fully balanced by
randomisation (0.489–0.511 across all four), so this is not the positional
confound, and the same rank order appeared independently in both symmetry
conditions.

It is still not a finding:

| test | result |
|---|---|
| omnibus max−min spread, agent labels permuted **within episode** | 21.3 words, *p* = **0.156** |
| trend: corr(agent numeral, verbosity) | *r* = +0.268, *p* = 0.019 |
| `A1+A2` vs `A3+A4` propagation split | +0.226, *p* < 0.0001 |

The two significant tests are both contrasts **chosen after seeing the
pattern**. The omnibus — which does not require choosing one — fails. This is
the same error as the withdrawn variance effect in §4.1b, encountered twice in
one day, and the second time it was more tempting because the effect is larger
and replicates across conditions.

**Recorded as a prediction rather than a result.** If minimal symmetry breaking
(a name and a seed) really does produce stable behavioural differences, then on
a fresh corpus verbosity should again increase with the agent numeral. That is
falsifiable, costs nothing beyond episodes already planned, and can be declared
before the data exists — which is the only thing that would make the *p*-value
mean anything. It is the first candidate for the preregistration in §7.7.

### 4.8 Phase 2 — no differentiation, and an instrument that could not have seen it

468 messages across all 48 episodes, coded by the local Qwen3-8B judge in about
six minutes on the freed card. **0 unparseable, 0 judge errors** — the run
itself was clean, which is what makes the rest of this section a finding about
the corpus rather than about the harness.

**The taxonomy collapsed to two labels.**

| label | n | share |
|---|---:|---:|
| `propose` | 294 | 62.8% |
| `verify` | 155 | 33.1% |
| `compute` | 8 | 1.7% |
| `other` | 6 | 1.3% |
| `agree` | 4 | 0.9% |
| `search` | 1 | 0.2% |
| `synthesize` | **0** | — |
| `organize` | **0** | — |

Not a solo-episode artefact: in the 276 team messages, where `organize`,
`synthesize` and `agree` are all possible, the same two labels take 98.1%.

**This breaks Phase 4 independently of the failed gate.**
`ACTION_TO_COMPONENT` maps exactly four actions onto graded components —
`compute`→arithmetic, `search`→search, `verify`→verification,
`synthesize`→synthesis. Three of the four have essentially no data. Even with
an ablation grid in hand, convergent validity would have been a test of
`verify` alone rather than of the mapping. (`converge` refused to run at all,
correctly: `missing ablation.jsonl`.)

**The one solid behavioural result: teams generate, lone agents audit.**
Propose as a share of propose+verify, per episode:

| | n | mean | sd |
|---|---:|---:|---:|
| solo | 12 | 0.403 | 0.132 |
| team | 36 | 0.674 | 0.101 |

Gap **+0.272**, episode-level permutation ***p* < 0.0001**. A single agent
spends most of its turns auditing its own draft; agents in a group spend most
of theirs putting assignments forward. This one is trustworthy where §4.1b's
variance effect and §4.7's gradient were not: the mechanism was stated before
the test was run, the permutation unit is the episode, and the separation is
wide relative to both standard deviations.

**And the central question comes back null.** On the only contrast with enough
volume to test:

| test | statistic | *p* |
|---|---|---|
| within-episode differentiation (do agents in an episode differ?) | mean within-episode sd of propose-share = 0.231 | **0.451** |
| cross-episode stability (is `A1` reliably one way?) | max−min across agents = 0.093 | **1.000** |

Agents within an episode differ no more than shuffling the same labels among
them would produce, and no agent identity carries a stable tendency across
episodes (A1 0.625, A2 0.676, A3 0.676, A4 0.718). Both nulls, at the right
permutation unit, on a pre-stated test.

**The caveat that matters more than the result.** A judge that never once
emitted `organize` or `synthesize` cannot detect differentiation *in organizing
or synthesizing*. So this corpus **cannot distinguish "the agents did not
differentiate" from "the instrument could not see differentiation"** — and
those have opposite implications for the thesis. PLAN.md's Phase 2 note, that a
local 7–8B is an inadequate judge and is for smoke tests only, was written as a
budget caveat; it is now the binding constraint on the result.

That makes judge quality the critical path rather than a validity footnote.

**The frontier subsample partly rescues the instrument.** One episode
(`baseline:hard:9`, 12 messages) coded by Gemini against the same codebook —
all the free daily quota allows:

| | labels used | distribution |
|---|---|---|
| local 8B | 3 of 8 | verify 5, propose 5, agree 2 |
| Gemini | 3 of 8 | propose 6, verify 5, synthesize 1 |

**κ = +0.596**, raw agreement 0.75, 95% bootstrap CI **[+0.24, +1.00]**. The
interval is nearly useless at n=12 — it contains the 0.78 of the prior-art
paper and also contains "barely better than chance".

But the *distribution* comparison is the one that was actually needed, and it
is more legible than κ: **the frontier judge also concentrates on
propose/verify**, using three labels on the same twelve messages. It did not
find the organizing and synthesizing that the 8B missed. That is evidence for
"these transcripts are genuinely uniform" over "the small judge is blind",
which shifts the reading of the null in §4.8 from *probably an artefact* to
*probably real*. Weakly — one episode.

Where the two disagree is still informative: both of the 8B's `agree` labels
were something else to Gemini (`synthesize`, `verify`). The small judge's
characteristic error looks like reaching for the contentless category, which is
exactly the direction that would suppress a differentiation signal.

**A third rater settles it: the local judge is good enough, and the collapse is
real.** 40 messages were sampled from team episodes, stratified to *oversample
the rare labels* — the sample is 6 `compute`, 4 `other` and 1 `agree` against
15 `propose` and 14 `verify`, deliberately loading it with the cases where a
weak coder should fail. The rater saw message text only: no agent id, no local
label, no episode context. The Gemini-coded episode was excluded so the two
validations stay independent.

| | κ | 95% CI | n |
|---|---|---|---|
| local 8B vs Gemini | +0.596 | [+0.24, +1.00] | 12 |
| **local 8B vs third rater** | **+0.684** | **[+0.50, +0.85]** | **40** |

κ = 0.68 is substantial agreement, the interval excludes everything below 0.50,
and it contains the 0.78 of the prior-art paper. **PLAN.md's assumption that a
local 7–8B is inadequate as a coder is not supported for this taxonomy on this
corpus.** That assumption had been treated as the binding constraint on §4.8's
null; it is not.

The label distributions:

| | propose | verify | compute | other | synthesize | organize | agree |
|---|---|---|---|---|---|---|---|
| local 8B | 15 | 14 | 6 | 4 | 0 | 0 | 1 |
| third rater | 16 | 12 | 6 | 4 | 1 | 1 | 0 |

The stronger rater used six labels to the 8B's five and did find one
`synthesize` and one `organize` the 8B missed (calling them `agree` and
`compute`). So the 8B *does* under-detect the interaction categories — but at
roughly 2 in 40. Extrapolated over 468 messages that is ~23 additional
messages, still far too thin to support a differentiation analysis in those
categories. **The taxonomy collapse is a property of these transcripts, not of
the judge.**

The four `other` labels agreed perfectly. They are empty messages, and both
raters declining to invent a category for them is the behaviour the
`other` class exists for.

**Consequence for §4.8.** The differentiation null was reported with the caveat
that this corpus cannot distinguish "the agents did not differentiate" from
"the instrument could not see it". That caveat is now substantially discharged:
two independent validations, one at n=40 with κ = 0.68, say the instrument sees
what a better instrument sees. **The null stands as a finding about the agents.**

Two honest limits. The third rater is a large language model, not a human — this
is model-vs-model agreement, and a shared family of failure modes would inflate
κ in a way no amount of *n* corrects. And the rater had seen the corpus-level
label distribution before coding, which could pull its labels toward the
majority classes. Both push κ *up*, so the true agreement is if anything lower
than 0.68 — but neither undermines the direction of the conclusion, since the
rare-label oversampling was designed to expose exactly the disagreement that
would matter.

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
| Batching | token-budgeted, length-sorted, 30000 padded tokens, `memory_fraction` 0.95 |
| Difficulty | `hard` — 24 jobs, 6 workers, 6 exclusions, 5 synthesis constraints, capacity slack 1.1, value floor 0.72 |
| Figures | `python scripts/figures.py` regenerates all three from the corpus |

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

~~1. **Does `hard` clear the Phase 1 gate?**~~ **Answered: no** (§4.1b). All
three metrics non-significant on near-equal arms.

~~2. Ablation grid.~~ **Cancelled by the gate**, per PLAN.md's stop condition.
Steps 2 and 3 below are what the grid would have produced and are not
recoverable at this operating point; they need either a task the team is better
at or a larger model.

3. **Behavioural coding with the local 8B**; κ against the frontier subsample.
   *In progress.* This is the live thread — behavioural differentiation does
   not require a performance benefit, so a failed gate does not touch it.
4. **Convergent validity:** do transcript labels predict causal profiles? Note
   that with no ablation grid there is no causal profile to correlate against
   at `hard`. Convergence can only be reported against *score* contribution,
   which is a weaker target and must be labelled as such.
5. **Date the retraction.** Re-run `medium`'s team arm on the fixed harness.
   The solo arm (0.879) is clean and does not need regenerating, so this is
   twelve episodes of card time and it converts §4.1 from "withdrawn, unknown"
   into a measured point on the difficulty curve.
6. Extend N via resume; Phase 4 robustness across model families and team sizes.
7. Preregister Phases 3–4 before running them.

**The honest summary of what Phase 1 cost and bought.** It bought one negative
result (§4.1b), a difficulty curve with one measured point and one withdrawn,
and an instrument that has now caught five distinct silent-corruption modes.
It cost the ablation grid, which was the project's headline deliverable. The
grid is not abandoned — it is blocked on finding an operating point where the
team contributes something to ablate.

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
