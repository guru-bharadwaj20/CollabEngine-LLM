# Preregistration — the `xhard` operating point

**Written and committed before any `xhard` episode was generated.** Commit this
file, then run. Its only purpose is to fix the hypothesis, the test, and the
falsification condition while the answer is still unknown, because three claims
in this project have already died of being chosen after the data was seen
(RESEARCH-LOG §4.1b variance, §4.7 agent gradient, §4.1d feasibility).

## Motivation

Two operating points have been measured. A single agent degrades as instances
grow; four agents do not:

| | `medium` (16 jobs) | `hard` (24 jobs) | change |
|---|---|---|---|
| solo | 0.879 | 0.842 | −0.037 |
| team | 0.874 | 0.871 | −0.003 |

If that difference in slopes is real rather than noise, the lines cross above
`hard`, and there exists an instance size at which collaboration pays. There is
a mechanism: `max_tokens` is fixed at 1024 and rounds at 3, so one agent has a
hard ceiling on how much it can enumerate, while four agents have four times
that budget *if they divide the work*.

**This is currently a weak signal and is not evidence.** The slope difference is
0.034 against per-arm standard deviations near 0.10. It is written here so the
test can be confirmatory rather than exploratory.

## Competing hypothesis, and how it is distinguished

Agents may not divide at all. The measured propagation index is **0.633** —
roughly two thirds of an agent's distinct content is restated by teammates
(§4.7). A team that duplicates rather than divides gains nothing from a larger
instance and should track solo downward. The mechanism test below separates the
two readings, and it is the reason this run is worth the card time whichever way
the primary test goes.

## The `xhard` tier

Scaled from `hard` along the same axes, with the per-agent budget deliberately
**unchanged** (`max_tokens` 1024, `rounds` 3, `n_agents` 4) so that total work
grows while individual capacity does not.

| knob | `hard` | `xhard` |
|---|---|---|
| jobs | 24 | 36 |
| workers | 6 | 8 |
| blocks | 4 | 5 |
| exclusions | 6 | 9 |
| synthesis constraints | 5 | 7 |
| planted errors | 4 | 6 |
| capacity slack | 1.10 | 1.05 |
| value floor ratio | 0.72 | 0.78 |

## Hypotheses

**H1 (primary, trend).** The team−solo gap on `fraction` increases with
instance size across the three operating points. Tested as the difference in
slopes over {medium, hard, xhard} using all 72 episodes, against a permutation
null that shuffles the condition label within each difficulty. One-sided,
α = 0.05.

The trend is the primary test rather than the `xhard` contrast alone because it
uses every episode collected and is the hypothesis actually being advanced. A
single-point contrast at n=12 per arm has a minimum detectable difference of
roughly 0.12 at 80% power, and no gap measured in this project has exceeded
0.03 — a per-point test would be underpowered by construction.

**H2 (secondary).** At `xhard`, team > solo on `fraction`, one-sided
permutation, α = 0.05.

**H3 (mechanism).** If the crossover is produced by division of labour, the
propagation index falls at `xhard` relative to `hard` (0.633), because agents
can no longer afford to restate the whole working answer. If the propagation
index holds or rises while H1 fails, the duplication reading is supported.

**H4 (manipulation check).** Solo at `xhard` < solo at `hard` (0.842). If solo
does *not* degrade further, the instance size is not biting and H1 is untestable
on this run rather than false.

## Amendment 1 — 2026-08-09 22:50, before any `xhard` result was analysed

**n rises from 12 to 24 episodes per arm** (seeds 0–23), and the `hard` arms are
extended from 12 to 24 to match.

Two things prompted this and neither is a result. First, the card became
available for an uninterrupted night, removing the resource constraint that set
n=12 in the first place. Second, the original `xhard` corpus was discarded
before any hypothesis was tested: 19 of 24 episodes were instrument failures
under GPU contention (90 errored turns), and the **only** analysis run on it was
the integrity audit. No score, gap, or *p*-value from that corpus has been
computed or seen.

This is recorded as an amendment rather than a silent edit because increasing n
after glimpsing a result is one of the standard ways a preregistration is
laundered. The protection here is that there is nothing to have glimpsed. The
power change: minimum detectable difference at 80% power falls from roughly
0.12 to 0.085 for a single-point contrast.

Nothing else changes — hypotheses, tests, metrics and the falsification
condition are all as written above.

## Amendment 2 — 2026-08-11, before any episode on the new instrument

**The instrument changes from Qwen3-8B at bf16, loaded in-process, to
Meta-Llama-3.1-8B-Instruct at Q4_K_M, served over llama.cpp.** Consequently
**all three tiers are regenerated on the new instrument**, and the Qwen3-8B
curve is superseded rather than extended.

### Why

`xhard` cannot run on this card at bf16 and the reason is not a tuning
parameter. The prefill materialises a logits distribution for every prompt
position over a 151,936-entry vocabulary, costing 1–1.5 MB per prompt token
above 15.3 GiB of resident weights; `xhard` prompts *start* near 5,200 tokens
(§3.10). Serving a 4-bit model removes the mechanism: weights drop to ~4.6 GiB,
prefill runs in `-ub`-sized micro-batches, and logits are materialised only for
the sampled position. The tier becomes runnable at an 18,432-token slot.

The alternative was to abandon the crossover test for want of hardware. Between
a hypothesis tested on different weights and a hypothesis not tested, this
registers the first.

### What it costs, stated plainly

**The three published Qwen3-8B numbers — solo 0.879 / 0.842 and team 0.874 /
0.871 — are not comparable to anything measured after this amendment.** They are
a different model family at a different precision. Specifically:

* **H1** is a trend across {medium, hard, xhard}. A trend whose third point came
  from a different instrument measures the instrument. All three points must be
  re-measured, hence `configs/llamacpp-medium.yaml`, `-hard.yaml`, `-xhard.yaml`
  at n=24 per arm.
* **H4** is stated against the literal value 0.842. On the new instrument it is
  restated as *solo at `xhard` < solo at `hard`, both measured here*. The
  numeral is retired, not the check.
* **H3** is stated against a propagation index of 0.633, also a Qwen3-8B
  measurement. Same treatment: the comparison is `xhard` against `hard` on the
  new instrument.

Under no circumstances are Qwen3-8B and Llama-3.1-8B episodes pooled, and the
config `name` fields differ (`llama31-8b-q4-*`) so the corpora land in separate
run directories and cannot be pooled by accident.

### Two confounds this introduces, named before the data

1. **Quantisation.** Q4_K_M is lossy, and it is not obviously neutral between
   the arms: if 4-bit weights degrade long-context instruction-following more
   than short-context, the team arm — whose contexts are four times longer —
   absorbs more of the loss. That biases *against* H1, which is the safe
   direction for a hypothesis this project is trying to falsify honestly, but it
   is a reason a null result here is weaker evidence than a null at bf16 would
   have been. It will be reported as such.
2. **Model family.** Llama-3.1-8B is a different generalist. Its instruction
   following, verbosity, and willingness to divide labour are all unmeasured
   here. This is why `medium` and `hard` are regenerated rather than assumed:
   without them there is no way to tell a crossover from a family effect.

### What does not change

Hypotheses H1–H4 as stated (with the two numerals retired above), the tests, the
metrics, the n=24 per arm, the exclusion-and-regenerate rule, the two-arm
design, and the falsification condition. The `xhard` `DifficultySpec` is
untouched: the instances are byte-identical to those the original
preregistration described, because they are deterministic in
`(seed, difficulty)` and no generator constant has been edited.

### Falsification, restated for the new instrument

If H1 fails and H4 holds across the three tiers measured here, the conclusion is
that no operating point in this task family rewards collaboration **at 8B, in
either of two model families, across a 2.25× range in instance size**. That is a
stronger negative result than the one Amendment 1 anticipated, not a weaker one
— the same claim, replicated across a second set of weights.

## Analysis, fixed in advance

- n = 24 episodes per arm (amended from 12), seeds 0–23, identical instances
  across arms.
- Arms: `solo` and `baseline` only. No symmetry sweep, no fixed-order control —
  they are not part of any hypothesis here.
- Instrument failures excluded via `analysis.integrity`, and **any episode lost
  to OOM is regenerated before analysis, not dropped.** Three results in this
  project have been distorted by OOM dropouts, in both directions (§4.1d).
- Metrics reported for all three of `fraction`, `strict`, `feasible`.
  `fraction` is the primary; the other two are secondary and labelled as such.
- Permutation tests at 20,000 draws; 95% percentile bootstrap intervals.

## Falsification

If H1 fails (p ≥ 0.05) and H4 holds, the conclusion is that **no operating point
in this task family rewards collaboration at 8B**, across three sizes spanning a
2.25× range in jobs. Phase 3 stays cancelled, and the project's result is the
negative one, reported at three points rather than two.

That outcome is not a failed run. It converts "we could not find a benefit" into
"we looked across the full usable range of the task and there is none", which is
the difference between an inconclusive study and a finding.
