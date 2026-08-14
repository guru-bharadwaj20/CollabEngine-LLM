# Preregistration — Phase 3, the causal ablation grid

**Status: drafted 2026-08-14 00:1x, awaiting sign-off. No confirmatory episode
has been generated.** The pilot corpus this document is sized from
(`runs/llama31-8b-q4-medium-ans/ablation.jsonl`, seeds 0–47) is
hypothesis-generating and is **excluded from every test below** by construction:
the confirmatory run uses seeds 1000–1149 and lands in a separate run directory.

Its only purpose is to fix the hypotheses, the tests, and the falsification
condition while the answers are still unknown. Four claims in this project have
already died of being chosen after the data was seen (§4.1b variance, §4.7 agent
gradient, §4.1d feasibility, §4.10 the token cap), and a fifth nearly did
last night — see *The crossover I am not going to test* below.

## Motivation

The pilot (§4.19) established two things and they point opposite ways.

| | pilot, n=48/agent |
|---|---|
| pooled live drop | **+0.055**, 95% CI [+0.023, +0.087] |
| agent × component interaction | **chi2 = 3.73, *p* = 0.928** |
| largest double-centred residual | 0.027, against ~0.06 expected under a true null |

Removing one of four agents measurably costs the team. That is **participation**,
and PLAN §0 is explicit it proves nothing: *"If you remove any competent
contributor from any team, output falls — that is true of four identical agents
with no division of labor at all."*

The claim needs the **interaction**, and the pilot found less of it than chance
generates. This run exists to convert that from an underpowered null into a
**bounded** one, and to establish what the participation effect actually is.

## The crossover I am not going to test

At the 148-episode checkpoint the pilot showed A3 at −0.052 on `verification`
against +0.032 on `synthesis`, with A4 the mirror image — the exact crossover
PLAN §0 names as the causal signature of specialization. At 192 it was −0.027
against +0.019, and falling.

It was the largest of sixteen cells at a sample size where the largest of
sixteen cells is about that big anyway. **No hypothesis below names A3, A4,
`verification` or `synthesis`**, and no agent-specific or component-specific
directional prediction is registered. Recorded here because the temptation to
promote it to a hypothesis was real, and a preregistration that quietly encodes
the pilot's noise is worse than none.

## Design

| | |
|---|---|
| Operating point | `medium`, the one tier where the team contributes anything (§4.16c) |
| Instrument | answer-budget, `configs/llamacpp-medium-ans.yaml` geometry, unchanged |
| Seeds | **1000–1149**, disjoint from the pilot's 0–47 |
| n | **150 episodes per arm** |
| Arms | `baseline`; `live:A1..A4`; `frozen_replay:A1..A4`; `capacity`; `random_message` |
| Generating arms | 10 × 150 = **1,500 episodes ≈ 12 GPU h** at the measured 2.1 ep/min |

`frozen_excise` is **not** an arm. Propagation on real transcripts is 0.589 and
its drops are ~0.000 (§4.18); it measures nothing here and its cost is zero
either way, so it may be computed but will not be reported as a contribution
estimate.

**Why n=150.** It is what makes the equivalence bound worth stating. Per-component
drop sd is 0.311, so a double-centred residual carries se ≈ 0.034 at n=48 and
**≈ 0.019 at n=150** — tightening the detectable interaction from ~0.06 to
**~0.038**. It is also two nights rather than a week, which is the difference
between a run that happens and one that is planned.

## Hypotheses

**H1 (primary). There is an agent × task-component interaction in live-ablation
drops.** Mixed-effects joint Wald test over the agent × component term, episode
as a random intercept (`analysis/mixed.py`), two-sided, α = 0.05.

*Predicted outcome: H1 fails.* Registering a predicted null is deliberate. The
pilot's evidence is not "we could not find it" but "there is less structure than
noise produces", and the value of this run is the bound, not the discovery.

**H1e (the bound, and the actual deliverable).** If H1 fails, report the largest
agent × component residual detectable at 80% power, and the 90% CI on every cell
of the 4 × 4 residual matrix. The reportable claim is *"no agent × component
interaction larger than X"*, X fixed by the achieved n and stated to two decimal
places.

**H2 (participation, confirmatory).** Pooled live-ablation drop > 0. One-sided
permutation, α = 0.05, 20,000 draws. The pilot estimate is +0.055 [+0.023,
+0.087]; this is a replication on fresh seeds, not a new question.

**H3 (capacity control).** Live drop > capacity drop. If removing agent *i* from
a four-agent team costs more than never having had a fourth agent, the loss is
attributable to *that agent's contribution* rather than to head-count. If the two
are indistinguishable, **H2's effect is head-count and nothing more**, which is
the deflating reading and must be reported as the headline if it holds.

**H4 (volume control).** Live drop > random-message drop, volume-matched. Separates
"this agent's contribution" from "less text in the context window". Same logic
as H3: if indistinguishable, the effect is context volume.

**H5 (fungibility, descriptive).** Δ(frozen_replay) − Δ(live) per agent, reported
with intervals, no threshold. Compensation is expected to make live drops the
smaller of the two; the gap is the redundancy metric PLAN §C1 defines. Registered
as descriptive because no prior value exists to predict against.

## Analysis, fixed in advance

- Metrics: `fraction` primary; `strict` and `feasible` secondary and labelled.
  Note `feasible` is 0.00–0.02 in every arm measured to date and will likely
  carry no information; it is reported for completeness, not interpreted.
- Drops paired on `instance_seed` against the same-seed baseline episode.
  Pairing is expected to buy nothing — measured *r* ≈ 0.00–0.36 across arms
  (§4.18) — and is used because it is the correct estimator, not for power.
- Instrument failures excluded via `analysis.integrity`; **any episode lost to
  OOM or server death is regenerated, not dropped** (§4.1d).
- Permutation tests at 20,000 draws; 95% percentile bootstrap intervals.
- No metric, arm, or subgroup is added after the first look at these seeds.

## Falsification, and why every branch is reportable

| outcome | conclusion |
|---|---|
| H1 holds | Specialization is causally real. The project's original thesis, confirmed. |
| **H1 fails, H2/H3/H4 hold** | **Expected.** Agents participate and are individually load-bearing beyond head-count and volume — but do not specialize. This is PLAN §7's claim in its strongest form. |
| H1 fails, H3 or H4 fails | The participation effect is head-count or context volume. The deflating reading, and it must lead the writeup. |
| H1 fails, H2 fails | The pilot's +0.055 did not replicate on fresh seeds. Report the failure to replicate and stop. |

**The expected branch is not a failed project.** PLAN §7 states the target claim
as *"apparent specialization is an artifact of reading structure into text"* where
the interaction is absent. A bounded null on the interaction, paired with a
measured participation effect that survives both controls, is that claim with
numbers attached rather than an absence of evidence.

What it still owes is the other half of the convergent-validity comparison —
transcript-derived role labels — and §4.17 has that blocked at κ = 0.29. **This
run does not unblock Phase 4 and does not claim to.**

## Amendment 1 — 2026-08-14 01:4x, before any confirmatory episode exists

**`capacity` becomes four arms, one per excluded agent, not one arm.**

`capacity_control` lowers `n_agents` by one, which always removes the *last*
agent, so it only ever produces the roster A1–A3. That roster matches `live:A4`
and nothing else: H3 as originally written quotes a single capacity number
against all four live cells, and for three of them there is no matched control
at all (§4.20).

The overnight controls also showed why this matters rather than being tidiness.
`live:A4` and `capacity:3` leave the *identical* roster on the identical seeds
and differ by **+0.052 — 74% of the drop attributed to removing A4** — at
*p* = 0.123, which is the wrong side of both significance and negligible. If
that gap is real, live-ablation drops measure the configuration as much as the
agent, and H2/H3 as written cannot tell the two apart.

**What changes.** `capacity:without-Ai` for each of the four agents, 150 episodes
each. That is 600 additional episodes, ~5 GPU h at the measured rate, taking the
grid from 1,500 to 2,100 episodes and ~12 to ~17 GPU h.

**H3 is restated** as: for each agent *i*, `live:Ai` drop > `capacity:without-Ai`
drop, tested per agent and pooled. The pooled test is primary; the per-agent
tests are secondary and will be underpowered by construction.

**H3b is added (procedural neutrality, primary for interpretation).** With the
same roster and seeds, `capacity:without-Ai` = `live:Ai`. This is the assumption
every ablation number in this project already rests on and it has never been
tested at adequate n. If it fails, the correct headline is that **the ablation
instrument is not neutral**, and every drop reported here needs restating
against the capacity arm rather than against the four-agent baseline.

Nothing else changes. No hypothesis direction is edited, and this amendment is
written from a control result, not from any test of H1–H5.

## Amendment 2 — 2026-08-14 23:3x, after H2 and before anything else

**H2 failed. The grid is halted, and the headline changes.**

Fresh seeds 1000–1149, all four agents, 599 usable ablation episodes against a
150-episode baseline: pooled live drop **+0.003, 95% CI [−0.015, +0.021]**, no
agent individually distinguishable from zero, A3 negative. The pilot's estimate
was +0.055 [+0.023, +0.087] (§4.22).

**The cause is identified rather than suspected.** The three-agent arms reproduce
across the two seed sets to within 0.002 over 791 episodes (*p* = 0.905); only
the four-agent baseline moved, 0.631 → 0.577. The participation effect was the
difference between a 48-episode baseline that drew high and ablated arms that
have not moved at all.

### What halts

**H1, H1e, H3, H4 and H5 are suspended, and the remaining `frozen_replay`,
`random_message` and capacity arms are not run.** Every one of them is expressed
as a drop from the four-agent baseline. With the pooled drop at +0.003 there is
no main effect to decompose: an agent × component interaction needs a
contribution to be differentially distributed, and the fungibility metric would
be a difference between two zeros. Running them would produce numbers with no
quantity behind them.

This is the preregistration's own branch rule — *"H1 fails, H2 fails → report the
failure to replicate and stop"* — and it is being followed rather than
renegotiated now that it has fired.

### What the project's result now is

Three nulls, and they are consistent with each other: no participation, no
specialization, no compensation. At `medium`, on this model, removing one of four
agents costs nothing measurable — which subsumes rather than contradicts
§4.19–4.20, since a flat interaction and zero fungibility are what a
zero main effect predicts.

### What replaces the grid

Not more ablation. The open question is now upstream of Phase 3 entirely:
**`medium` was selected as the one operating point where the team beats a
matched-budget single agent, and that selection rests on the same 48-episode
corpus** (§4.16b, +0.126 at *p* < 0.001 against C5). H2 has just shown that
corpus's four-agent baseline to be the high draw. The C5 contrast must be
re-measured on seeds 1000–1149 before any claim resting on `medium` stands.

That is one arm — `solo_long` at n=150, ~2 GPU h — and it is now the highest
value measurement available, because if it does not replicate either, the
project's remaining positive finding goes with it.

### What does not change

The fresh-seed rule, which is what caught this. No hypothesis direction is
edited. No result is deleted; §4.19 and §4.20 stand as recorded with §4.22
attached to them.

## What would make me amend this

Stated in advance, since amendments after a glimpse are how a preregistration is
laundered (cf. PREREG-xhard Amendment 1, which was clean only because nothing
had been looked at):

- The card becoming unavailable, reducing achievable n. The bound in H1e loosens
  accordingly and is restated; no hypothesis changes.
- Instrument failure above ~5% in any arm, which would mean regenerating before
  analysis rather than proceeding.
- Nothing else. In particular, not the direction of any observed effect.
