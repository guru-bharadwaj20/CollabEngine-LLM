# Results as measured, 2026-08-22

**Every number here was produced after its preregistration was written, and the
registered threshold is quoted beside each one.** Two registered predictions
failed. They are reported as failures, in the same table as the successes.

This document is the raw material for the paper. It states what was measured and
what the registration said to conclude; it does not draft prose.

**Corpora.** `runs/llama31-8b-q4-medium-h3b` (fresh-seed headline, 450+600),
`runs/llama31-8b-q4-code-medium` (450+600), `runs/llama31-8b-q4-cap{0512,1024,2048,3072,6144}`
(200 each), `runs/qwen25-7b-q4-medium` (450), `runs/llama31-8b-q8-medium` (450),
plus six restored corpora. About 3,700 episodes, **zero generation failures**.

---

## 1. The dose-response curve — `PREREG-cap-sweep.md`

**The registered contrast passed by 3.5x.** This is the strongest new result in
the study.

| cap | answer chars solo/team | apparent team advantage | 95% CI | excludes 0 | equivalence bound |
|---|---|---|---|---|---|
| 512 | 3.31 | **+0.390** | [+0.323, +0.456] | yes | 0.446 |
| 1024 | 4.08 | +0.100 | [+0.045, +0.156] | yes | 0.147 |
| 2048 | 3.84 | +0.069 | [+0.018, +0.123] | yes | 0.115 |
| 3072 | 4.89 | +0.026 | [-0.011, +0.064] | no | 0.057 |
| 6144 | 6.75 | +0.043 | [-0.003, +0.091] | no | 0.082 |

> **Registered (H-C1):** largest advantage at cap <= 1024 minus largest at cap
> >= 3072, **threshold >= +0.10**, with the short-cap interval excluding zero
> and truncation counts asymmetric.
>
> **Measured: +0.347.** Confirmed.

**What it licenses.** At a tight cap the team appears **+0.390** better — a
large, significant, entirely artefactual result. The same instrument at a
generous cap shows nothing. The artifact is not a fixed bias to subtract; it is
a function of the cap, and the curve is what makes the diagnostic *predictive*
rather than merely diagnostic: given a published system's per-turn cap and the
verbosity ratio between its arms, the curve estimates how much of its reported
advantage is instrument.

**A caveat to carry into the prose.** The `answer chars solo/team` column is not
monotone in the cap — 3.31, 4.08, 3.84, 4.89, 6.75, with the 1024 rung out of
order. The *advantage* is monotone in the cap; the *ratio* is not. Presenting
the ratio as a clean x-axis without saying so would overstate the tidiness of
the curve.

---

## 2. The second task family — `PREREG-code.md`

**The headline: the artifact class transfers, but a different member of it
dominates.** One registered clause is confirmed and one is falsified.

### 2.1 Interpretability gates, checked before the gate was read

| check | registered limit | measured | verdict |
|---|---|---|---|
| floor | team mean > 0.15 | **0.7275** | interpretable |
| parser dominance | malformed < 30% | solo **12.0%**, team **1.3%** | interpretable, 9x asymmetric |

### 2.2 Power, checked before C2 was read

| | assumed | realised |
|---|---|---|
| sd (`fraction`) | 0.15 | **team 0.289, solo 0.374** |
| MDE at n = 150 | 0.049 | **0.094 to 0.121** |

> `PREREG-code.md` §4 registered the response in advance: *"if the realised
> code-family sd is materially larger … this arm is underpowered for C2 and the
> honest report is a wide equivalence bound rather than a null."*
>
> **It is. Report the bound, not a null.** The registered margin of 0.05 sits
> below the MDE, so this arm cannot resolve the margin it was registered
> against. That was foreseeable only from a prior, which is why the prior was
> written down.

### 2.3 C1 — does the artifact transfer?

| clause | registered | measured | verdict |
|---|---|---|---|
| C1a verbosity ratio | > 1.5 (falsified < 1.2) | **1.78** answer-turn | **confirmed** |
| C1b truncation gap | >= 10 pp (falsified < 5 pp) | **0.0% vs 0.0% = 0 pp** | **FALSIFIED** |

**C1 is falsified on its truncation clause, and this is the most informative
outcome in the document.** The solo arm is still 1.78x more verbose in its answer
turn — the *behaviour* transfers — but at a 3,072-token budget nothing is cut, in
either arm. The cap did not bite on this task.

The whole-episode ratio is **0.26** (solo 1,664 characters, team 6,437), which is
the §4.18 direction: a 3-round solo arm writes three turns against the team's
twelve. Both ratios have to be reported. Quoting either one alone misleads, in
opposite directions.

### 2.4 C2 — the gate

| | team | solo | gap | 95% CI | equivalence bound |
|---|---|---|---|---|---|
| all episodes | 0.7275 | 0.6492 | **+0.0783** | [+0.014, +0.142] | 0.142 |
| **well-formed only** | 0.7373 | 0.7377 | **-0.0004** | [-0.059, +0.058] | **0.0585** |

**The entire apparent team advantage is the unparseable solo answers.** Excluding
episodes scored 0.000, the gap is **-0.0004** across 148 team and 132 solo
episodes.

TOST at the registered margin gives *p* = 0.0801 on well-formed episodes, so
**equivalence does not hold at 0.05** — consistent with the power finding above.
The defensible statement is the bound: *effects larger than 0.0585 are excluded.*

### 2.5 C3 — `solo_long`

| | measured | registered prediction |
|---|---|---|
| solo_long minus solo | **+0.0529** | negative, per scheduling's -0.063 |

**Falsified — the sign reverses.** More turns helped the single agent here.
`solo_long`'s malformed rate is 6.0% against solo's 12.0%, so some of this is the
same parser effect rather than better reasoning. That decomposition has not been
done and should not be asserted either way.

---

## 3. Model family and precision — `PREREG-families.md`

| arm | n (team/solo) | team | solo | gap | equiv bound | MDE | malformed solo / team |
|---|---|---|---|---|---|---|---|
| Llama Q4 (headline) | 150/149 | 0.6063 | 0.5250 | +0.0813 | 0.118 | — | **16.8% / 0.0%** |
| Llama Q8_0 | 150/149 | 0.6074 | 0.5538 | +0.0535 | 0.089 | 0.038 | **12.1% / 0.7%** |
| **Qwen2.5-7B** | 150/150 | 0.6107 | 0.6166 | **-0.0059** | **0.0288** | 0.037 | **0.0% / 0.0%** |

**F1 confirmed.** On Qwen the null replicates and the bound is **0.0288** — the
tightest equivalence bound in the study, on a second model family.

**F2 cannot be completed.** The registered test is the *interaction* — whether the
team-minus-solo gap grows with precision — and it needs the f16 rung, which was
cut on 2026-08-22 when analysis rather than generation became the constraint.
Q4 to Q8 moves the gap by -0.028 (0.0813 to 0.0535): the gap *shrank* at higher
precision, the opposite of the worry in PAPER-DRAFT §9. **One rung is not an
interaction.** Report as untested, not as reassurance.

### 3.1 A new finding: the parser artifact is Llama-specific

The malformed column above is the most surprising result of the day.

- Llama Q4: **16.8%** of solo answers scored 0.000, against **0.0%** of team answers.
- Llama Q8: **12.1%** against **0.7%**.
- Qwen: **0.0% against 0.0%.**

On Llama Q4, excluding malformed answers moves the gap from **+0.0813 to
-0.0245** — the single agent ends up *ahead*.

The fourth artifact (§4.24) is therefore not a universal property of
answer-format parsing. It is an interaction between one parser's conventions and
one model family's output habits. The parser was written against Llama's
conventions; Qwen satisfies it without special handling.

**This strengthens the class claim and narrows its fourth instance.**
Symmetric-by-specification conventions land asymmetrically, and *which* arm they
land on depends on the model — which is worse for the field than a fixed bias
would be, because it cannot be corrected by a constant and will not show up in a
single-model study at all.

---

## 4. Summary against every registered threshold

| # | Registered claim | Threshold | Measured | Verdict |
|---|---|---|---|---|
| H-C1 | cap sweep contrast | >= +0.10 | **+0.347** | **confirmed** |
| C1a | code verbosity ratio | > 1.5 | 1.78 | confirmed |
| C1b | code truncation gap | >= 10 pp | **0 pp** | **falsified** |
| C2 | code gate equivalent at 0.05 | *p* < 0.05 | *p* = 0.0801, bound 0.0585 | **underpowered; bound reported** |
| C3 | solo_long worse than solo | negative | **+0.053** | **falsified** |
| F1 | null replicates off Llama | 0.05 | bound **0.0288** | **confirmed** |
| F2 | precision interaction | gap growth < 0.05 | one rung only | **untested** |

**Three confirmed, two falsified, one underpowered, one untested.** Reporting
that distribution is the paper's strongest evidence that the preregistration was
real rather than decorative.

---

## 5. What is measured, and what is not

**Measured.** Score at a fixed turn-and-token budget, and how that score moves
with the per-turn cap, the task family, the model family and the quantisation.

**Not measured, and therefore not claimable:**

- **Wall-clock time, or time-to-fixed-quality.** Every arm received the same
  budget and was compared on score. Nothing here supports "teams finish faster",
  and the parallel-speedup literature is a separate and mature subfield.
- **That a single agent catches up given more time.** `solo_long` runs -0.063 on
  the scheduling family and +0.053 on the code family. The direction is
  task-dependent and currently unexplained; see `RELATED.md` on self-conditioning
  for a candidate mechanism.
- **Anything about ablation on qwen, Q8 or the cap rungs.** Those arms are
  `pipeline` only and say nothing about differentiation or fungibility.
- **14B and f16.** Cut by decision rather than failure on 2026-08-22. Configs,
  weights and `PREREG-14b.md` are intact and resume from disk.
- **Mistral.** Withdrawn: its chat template cannot express this study's briefs.
  See `PREREG-families.md` for the amendment.
