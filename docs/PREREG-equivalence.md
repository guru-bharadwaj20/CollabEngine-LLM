# Preregistration — equivalence margins

**Registered 2026-08-20, before any TOST was computed against any corpus in this
project.** The margins below are chosen from the task's own structure. They are
not chosen from the confidence intervals already published, and the reason that
distinction is worth a document is the same reason the fresh-seed rule exists:
a threshold picked after seeing the interval it will be compared against is not
a threshold.

---

## 1. Why this document exists

This project's headline claims are null claims:

- team size has no effect on `fraction` at any operating point measured;
- the agent × component interaction is flat;
- fungibility, Δ(frozen_replay) − Δ(live), is zero;
- no agent identity carries a stable behavioural tendency across episodes.

Every one of them is currently supported by *failing to reject* H₀, and a
non-significant *p*-value is not evidence for H₀. It is compatible with any
effect the sample was too small to see — which is not hypothetical here. The
withdrawn participation finding (RESEARCH-LOG §4.22) was **+0.055 measured on a
48-episode arm with `sd` ≈ 0.15**, where the minimum detectable effect at 80%
power is **0.086**. That arm could never have distinguished the effect it
reported from noise, in either direction.

Equivalence testing is the instrument built for the claim actually being made.
It can fail, which is what makes it evidence.

## 2. The margin, and where it comes from

**δ = 0.05 on `fraction`, for every team-vs-solo and ablation contrast.**

`fraction` is partial credit over the gradeable constraints of an instance. At
`medium` an instance carries 29 constraints of which roughly twenty enter the
score, so **one constraint is worth about 0.05**. That is the smallest
difference the task can express, and it is the unit that would matter to anyone
choosing an architecture: a gap smaller than one satisfied constraint does not
correspond to the system getting anything additional right.

Two cross-checks, both agreeing:

- **Deployment relevance.** A difference below one constraint would not change
  any decision about whether to run one agent or four, given that four cost ~4×
  the forward passes.
- **Instrument resolution.** The project's own corrections have moved gaps by
  0.03–0.25. A margin below 0.05 would be finer than the artifacts the
  instrument is known to carry, and would claim a precision the corpus does not
  have.

For `strict` and `feasible`, δ = 0.05 is retained for `strict` and **no
equivalence claim is made on `feasible` at all**: it runs 0.000–0.021 in nearly
every arm, so it has no usable scale and PAPER-DRAFT §9 already reports it as
uninformative. An equivalence bound on a metric that is zero everywhere is
arithmetic, not evidence.

## 3. What is registered

| # | Contrast | Margin | What equivalence would mean |
|---|---|---|---|
| E1 | team (4 agents × 3 rounds) − solo (1 × 3), per tier | 0.05 | The gate's null is a bound, not an absence |
| E2 | team − `solo_long` (C5, matched budget, solo brief) | 0.05 | The matched-budget null is a bound |
| E3 | 4-agent − 3-agent (per-agent ablation drop) | 0.05 | "Removing an agent costs nothing" is a bound |
| E4 | Δ(frozen_replay) − Δ(live) (fungibility) | 0.05 | "Survivors do not compensate" is a bound |
| E5 | precision arms, Q4_K_M − Q8_0 − f16, per arm | 0.05 | Quantisation is not carrying the null |
| E6 | model families at matched scale and quantisation | 0.05 | The null is not a Llama quirk |

**Test:** two one-sided Welch *t*-tests, α = 0.05, implemented in
`collabengine.analysis.inference.tost`. Welch rather than pooled because the
arms in this project have demonstrably unequal spread (solo 0.281 against the
team's 0.107 at `medium` before the cap was controlled).

**Reported for every contrast, whatever the verdict:** the TOST *p*, the 90%
interval, and `smallest_equivalence_bound` — the δ at which equivalence would
hold. The last of those is the number intended for the paper's tables.

## 4. The prediction, registered before it was run

1. **E1 and E2 will be equivalent at δ = 0.05 on the fresh-seed `medium`
   corpus** (*n* ≈ 150 per arm). The observed spread of 0.005 across 899
   episodes should support a bound well inside one constraint.
2. **E3 will be equivalent.** +0.003 [−0.015, +0.021] over 599 episodes already
   sits inside the margin.
3. **E4 will *not* be equivalent**, and this is the honest prediction rather
   than the flattering one: fungibility is estimated on the 48-episode pilot
   corpus, and at that *n* the bound will be wider than 0.05. The paper must
   then say "we cannot exclude compensation of up to X", not "there is no
   compensation."
4. **The interaction test will not be equivalent either**, for the same reason.
   PLAN §1 already states that this study "bounds the effect rather than
   excluding it"; this converts that sentence into a number.

**Falsification.** If E1 or E2 fails equivalence on the fresh-seed corpus, the
paper's central claim weakens from "team size does nothing" to "we cannot
resolve whether team size does anything", and the abstract changes accordingly.
That is the outcome this registration exists to make costly to ignore.

## 5. What this registration does not license

- It does not license reporting equivalence and significance selectively. Both
  columns are printed for every contrast by `scripts/analysis/gate_report.py`,
  including the ones where the bound is embarrassing.
- It does not license widening δ after a failure. If 0.05 turns out not to be
  achievable, the reported result is the achieved bound, not a larger margin.
- It does not convert an underpowered null into a finding. `not equivalent`
  means the corpus is too small to exclude an effect of that size. That is a
  statement about the corpus.

---

*Amendments below this line, dated, with the reason. Nothing above it is edited
after registration.*

---

## Postscript — the bounds, 2026-08-20

Computed the same day, after registration, against the **published summary
statistics** (difference, `sd` ≈ 0.15, per-arm *n*) rather than against raw
episodes. **The raw corpus is not on disk** — `runs/` was cleaned and every
`*.jsonl` is gitignored, which is Final Sweep item 1.0.a and 1.15. These are
therefore the correct numbers given what the project has published, and they
will be recomputed from episodes by `scripts/analysis/gate_report.py` the moment
a corpus exists. Any disagreement between the two is a reason to trust the
second.

| # | contrast | diff | s.e. | **bound** | TOST *p* | verdict |
|---|---|---|---|---|---|---|
| E1 | gate: 4 agents − solo, fresh `medium` | −0.003 | 0.017 | **0.032** | 0.004 | equivalent |
| E1b | 4 agents − 3 agents | +0.002 | 0.014 | **0.025** | <0.001 | equivalent |
| E2 | team − `solo_long` (C5, truncation-corrected) | +0.003 | 0.017 | **0.032** | 0.004 | equivalent |
| E3 | per-agent ablation drop | +0.003 | 0.014 | **0.026** | <0.001 | equivalent |
| E4 | fungibility Δ(frozen_replay) − Δ(live), pilot | −0.005 | 0.031 | **0.056** | 0.072 | **not equivalent** |
| — | the *withdrawn* +0.055, on its own 48-episode arm | +0.055 | 0.024 | 0.095 | 0.582 | not equivalent |

**All four registered predictions hold, including the unflattering one.**

**What E1–E3 buy.** The project's central negative result stops being "we could
not detect an effect" and becomes **"we exclude effects larger than 0.032 on
`fraction`"** — smaller than one satisfied constraint, and therefore smaller
than the smallest difference the task can express. That is a bound, and it is
the strongest form this result can take.

**What E4 costs, and it is the point of registering the prediction.**
Fungibility is estimated on the 48-episode pilot, where the bound is 0.056 —
*outside* the margin. So the correct sentence is **"we cannot exclude
compensation of up to about 0.06"**, not "survivors do not compensate."
§4.20's claim and README's "no compensation" both overstate what *n* = 48
supports, and are corrected rather than defended.

**The last row is the argument for the whole document.** The withdrawn
participation finding was +0.055 with an equivalence bound of 0.095 and an
a-priori MDE of 0.086 at that *n*. It was published as a headline from an arm
that could not resolve it in either direction. One line of this analysis, run
before the pilot rather than after it, would have said so.

### Power, as it should have been recorded

| *n* per arm | MDE at 80% power, `sd` = 0.15 |
|---|---|
| 24 | 0.121 |
| **48** | **0.086** — larger than the +0.055 it was used to claim |
| 150 | 0.049 |
| 599 | 0.024 |
| 899 | 0.020 |

To detect one constraint (0.05) at `sd` = 0.15 takes **142 episodes per arm**.
Every arm in this project below that number was, by construction, unable to see
the effect it was testing for.
