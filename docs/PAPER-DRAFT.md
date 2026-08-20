# Three Positives and a Cap

### Measurement artifacts in single-agent versus multi-agent LLM comparisons

**Draft — 2026-08-15.** Every number is reproducible from the released corpus via
`scripts/analysis/gate_report.py`; section references are to `docs/RESEARCH-LOG.md`.

---

## Abstract

We set out to test whether emergent role differentiation in same-model LLM agent
teams is causally real, using leave-one-out ablation rather than transcript
reading. We could not run that test, and the reason is the contribution.

Across three task difficulties, two serving instruments, and two disjoint seed
sets, we obtained **three separate statistically significant results favouring
multi-agent teams — and traced all three to measurement artifacts**. Two were the
same artifact: a per-turn token cap that is symmetric by construction and
asymmetric in effect, because a single agent must emit an entire solution in its
final turn while a team commits one its shared transcript already holds. The
third was a small-sample baseline: a four-agent reference estimated from 48
episodes that a 3× larger sample did not reproduce.

Removing all three, the effect of team size on this task family is flat. One
agent scores 0.579, three agents 0.574, and four agents 0.576 across 899
episodes — a spread of 0.005. The only manipulation that moves the score
reliably is giving a *single* agent more turns, which makes it **worse**
(−0.063, *p* = 0.012).

We report the artifact mechanism, the three diagnostics that caught it, and a
bounded negative result. We argue that the diagnostics are the transferable
part: any comparison between architectures that consume a resource differently
will land a shared limit asymmetrically, and the field's standard reporting does
not currently surface this.

---

## 1. Introduction

The claim under test came from two 2026 results that had never been combined.
Observational work established that same-model LLM teams *appear* to
differentiate — agents settle into critic-like and planner-like behaviour that
survives coding by an LLM judge. Separately, causal ablation work showed that
introspective judgements of agent contribution do not match what ablation
measures, but applied it only to *assigned* roles, as an engineering tool.

Nobody had pointed the causal instrument at the emergent phenomenon. That was
the intended contribution.

**The design correction that made it a real experiment.** A scalar performance
drop when an agent is removed does not demonstrate specialization; it
demonstrates participation. Remove any competent contributor from any team and
output falls — true of four identical agents with no division of labour. The
signature of specialization is an **agent × task-component interaction**:
ablating the emergent critic must damage verification-loaded components *more*
than planning-loaded ones, and ablating the planner must do the reverse. The
main effect proves nothing; the crossover is the result.

That test requires an operating point where the team actually outperforms a
single agent — otherwise the ablation grid measures its own noise floor. We made
that a stop condition. **Establishing that operating point is where the project
stopped, and why this paper is about measurement rather than emergence.**

---

## 2. Setup

| | |
|---|---|
| Model | Meta-Llama-3.1-8B-Instruct, Q4_K_M, one model serving every agent |
| Serving | `llama.cpp`, 18,432 tokens per slot, 7 slots |
| Task | Synthetic multi-constraint scheduling; tagged constraint classes (arithmetic, search, verification, synthesis); deterministic per-component grader |
| Difficulties | `medium` / `hard` / `xhard` — 16/24/36 jobs |
| Metrics | `fraction` (partial credit, primary), `strict`, `feasible` |
| Statistics | 20,000-draw permutation tests, 10,000-draw bootstrap intervals |
| Corpus | ~2,900 episodes across all conditions; all transcripts released |

**One model, many conversation contexts.** Identical weights across all agents
means observed differentiation cannot come from model heterogeneity — a control
the prior observational work lacked, since it used seven different LLMs.

**Preregistration.** Hypotheses, tests, and falsification conditions were
registered before the runs they govern (`docs/PREREG-xhard.md`,
`docs/PREREG-phase3.md`), with amendments dated and justified. Two disciplines
in those documents did the work reported below: a **complete-case sensitivity row
printed beside every headline number**, and a **fresh-seed rule** forbidding a
corpus from testing the hypothesis it generated.

---

## 3. The artifact

### 3.1 Mechanism

Agents are capped at `max_tokens` per turn. The cap is identical across
conditions, which makes it look like a controlled variable. It is not, because
the arms spend the resource differently:

- A **single agent** at *k* rounds must produce the complete final solution
  inside its own last turn. The answer competes with reasoning for one budget.
- A **team** at *k* rounds ends on a turn that commits an answer the shared
  transcript already contains. Its final turn can be a summary; the content was
  distributed across earlier turns and earlier agents.

So the same cap truncates an *answer* in one arm and a *summary* in the other.
And because harder instances need longer answers, **the artifact grows with
instance size in exactly the shape a genuine collaboration benefit would
predict**.

Measured, at `medium`: the single agent emits 28,319 characters against the
team's 13,877 on a matched turn budget — a factor of **2.04×**. "Matched budget"
matched turns, not generation.

### 3.2 Three positives, one cause

| # | finding | as measured | after correction |
|---|---|---|---|
| 1 | Team beats solo at `hard` | **+0.249, *d* = 1.09, *p* < 0.0001** | **−0.026, *p* = 0.500** |
| 2 | Team beats matched-budget solo at `medium` (C5) | **+0.126, *p* < 0.001** | **+0.003, *p* = 0.865** |
| 3 | Removing an agent costs the team | **+0.055 [+0.023, +0.087]** | **+0.003 [−0.015, +0.021]** |

**Finding 1** was the project's only significant Phase 1 result. Rebuilding the
instrument so the answer turn has its own 3,072-token budget — holding slot
geometry constant — moved the solo arm's mean by **+0.239** while the team's
barely moved (0.591 → 0.555). Truncated solo answers went from 10 to 1,
malformed from 11 to 1. Nothing about the models or the task changed; the
measurement had been reading its own cap.

**Finding 2** survived that fix and a 3× sample increase, arriving at +0.060
(*p* = 0.008) on fresh seeds. It does not survive the sensitivity row: with
answer-truncated episodes dropped it is **+0.003**. The single-agent arm carries
34% truncation, 22 malformed and 20 cut answers out of 150, against the team's
8%, 4 and 4. The answer budget fixed the *team* arm; a single agent writing
2.04× the text at twelve rounds still hits the cap.

**Finding 3** is a different artifact with the same consequence, and it is the
one we did not anticipate. See §4.

---

## 4. The baseline that did not reproduce

Our ablation pilot measured a per-agent participation effect of +0.055 on seeds
0–47. Our own preregistration forbade testing that on the corpus that produced
it, so we re-ran all four agents on seeds 1000–1149.

**Every arm reproduced except one.**

| arm | seeds 0–47 | seeds 1000–1149 | Δ |
|---|---|---|---|
| `solo`, 1 agent × 3 rounds | 0.585 | 0.579 | −0.006 |
| 3-agent ablated arms (pooled, *n* = 791) | 0.575 | 0.574 | **−0.002** (*p* = 0.905) |
| **team, 4 agents × 3 rounds** | **0.631** (*n* = 48) | **0.577** (*n* = 149) | **−0.055** (*p* = 0.032) |

The three-agent arms agree to within 0.002 across 791 episodes. The four-agent
baseline — the reference every positive finding in the project was measured
against — moved by 0.055, which is the size of the effect being claimed.

**This is not seed-range difficulty.** The generator is deterministic in
`(seed, difficulty)` and untouched across both runs; instances in the two ranges
are structurally identical with zero variance (16 jobs, 5 workers, 29
constraints, 3 planted errors). If the fresh range were harder, the three-agent
arms would carry the same penalty. They do not move at all.

It is sampling noise in a mean estimated from 48 episodes on an arm with
`sd` ≈ 0.15 — a standard error of ~0.021 against effects of ~0.05 being read off
it. **An unremarkable statistical fact that produced a headline finding**, and
one that a small-*n* pilot in this literature will reproduce routinely.

---

## 5. Results: what reproduced and what did not

This is the paper's central table.

### Failed to reproduce — every one a positive, all measured against the 48-episode four-agent baseline

| | pilot | fresh | *p* |
|---|---|---|---|
| four-agent baseline | 0.631 | 0.577 | 0.032 |
| gate: team − solo | +0.045 | **−0.003** | 0.865 |
| C5: team − matched-budget solo | +0.126 | +0.060 → **+0.003** truncation-corrected | 0.865 |
| participation: per-agent ablation drop | +0.055 | **+0.003** | — |

### Reproduced — every one a null or a negative

| | pilot | fresh |
|---|---|---|
| `solo`, 1 agent | 0.585 | 0.579 |
| 3-agent arms | 0.575 | 0.574 |
| within-agent budget penalty (3 vs 12 rounds) | −0.081 (*p* = 0.065) | **−0.063 (*p* = 0.012)** |
| generation asymmetry, solo ÷ team | 1.87× | 2.04× |
| agent × component interaction | *p* = 0.928 | (not re-run; no main effect to decompose) |
| fungibility Δ(frozen) − Δ(live) | −0.005 | — |

**The asymmetry is the result.** Everything that reproduced across independent
seed sets is a null or a negative. Everything that failed to reproduce was a
positive measured against a reference estimated from 48 episodes.

### The headline number

On fresh seeds, with the artifacts removed:

| arm | `fraction` |
|---|---|
| 1 agent × 3 rounds | **0.579** |
| 3 agents × 3 rounds | **0.574** |
| 4 agents × 3 rounds | **0.576** |

A spread of **0.005 across 899 episodes**. Team size does nothing on this task
family at 8B. The only manipulation that reliably moves the score is giving a
*single* agent four times the turns, which moves it **down**: 0.579 → 0.516,
−0.063 at *p* = 0.012.

---

## 6. What we can say about emergence, and what we cannot

Phase 2 coded 468 messages against an eight-action taxonomy. Agents within an
episode differ no more in what they do than shuffling labels among them produces
(*p* = 0.45), and no agent identity carries a stable tendency across episodes
(*p* = 1.00). The one robust behavioural result is between conditions rather
than within teams: **teams generate and lone agents audit** — propose as a share
of propose+verify is 0.674 for teams against 0.403 for solo, *p* < 0.0001.

The ablation side agrees. The agent × component interaction is chi2 = 3.73,
*p* = 0.928, with a double-centred residual matrix holding **less structure than
a true null would be expected to produce** (largest residual 0.027 against ~0.06
expected). Blocking compensation does not increase the damage either:
Δ(frozen_replay) − Δ(live) = **−0.005**.

**But the scope must be stated precisely.** These are measured at an operating
point where the team has no advantage over a single agent. A flat interaction
and zero fungibility are what a zero main effect *predicts*; they are not
independent confirmations of it. **This is evidence that this task, at this
scale, does not produce the team advantage that specialization would have to
explain — not evidence that emergent specialization is absent in multi-agent LLM
systems generally.** The difference is between a bounded null and an
unsupported universal.

### Two instrument defects, reported rather than quietly fixed

**Frozen-transcript excision does not work on real transcripts.** Deleting an
agent's messages and re-grading gives drops of ~0.000 against live drops of
0.03–0.23, because agents restate the working solution constantly: the
propagation index — the fraction of an agent's content echoed by others later —
is **0.589**. The contribution survives in the copies. Read naively this reports
"the agent contributed nothing" when the truth is "this measurement does not
work here."

**Our capacity control controlled one of the four cells it was quoted against.**
Lowering `n_agents` by one always removes the *last* agent, so only one live
ablation cell shares its roster. On that cell, 74% of the attributed drop was
unexplained by head-count (*p* = 0.123, later +0.034 at *p* = 0.073) — too large
to ignore, too under-powered to act on, and moot once the main effect went to
zero. Both defects are released with the code.

---

## 7. Recommendations

For anyone comparing single-agent and multi-agent LLM systems:

1. **Print an answer-turn truncation count beside every headline number.** Not
   in an appendix. A gate evaluated without it passed at two of three operating
   points on an artifact here.
2. **Report a complete-case sensitivity row.** Dropping truncated episodes is
   post-hoc and biased; so is ignoring them. The two bracket the truth. Ours was
   right three times, including against a *p* < 0.0001 headline.
3. **A matched-budget arm is part of the comparison, not an extra.** Without
   one, "four agents beat one" and "more tokens beat fewer" are the same
   measurement.
4. **Match generation, not just turns.** Our matched-budget arm let the single
   agent emit 2.04× the team's text. Turn count is not token count.
5. **A single-agent baseline needs a single-agent prompt.** Running *n* = 1
   through a brief that says "the others" and "the group's last message is
   scored" measures the harness, in the direction that flatters the team.
6. **Never test a hypothesis on the corpus that generated it.** This rule cost
   us one extra run and overturned our headline. Without it we would have
   published a 2,100-episode ablation grid decomposing an effect that does not
   exist — and every cell would have been internally consistent.
7. **Any per-turn resource limit lands asymmetrically when the arms use the
   resource differently.** This generalises past token caps to context windows,
   wall-clock budgets, and tool-call quotas.

---

## 8. Limitations

- **One model family, one scale.** Llama-3.1-8B at Q4_K_M. Quantisation is
  lossy and not obviously neutral between arms — if 4-bit weights degrade
  long-context instruction-following more, the team arm absorbs more of the
  loss. That biases *against* a team advantage, so it weakens a null more than
  it would have weakened a positive, and we report it as such.
- **One task family.** Synthetic constrained scheduling. Chosen because it is
  programmatically gradable with per-component scores and uncontaminated, but a
  single task cannot support a claim about collaboration in general.
- **`feasible` is uninformative.** 0.000–0.021 in nearly every arm; almost no
  episode produces a fully feasible solution. All comparisons here are between
  degrees of partial credit.
- **The interaction test was not re-run on fresh seeds**, because H2 removed the
  main effect it would decompose. Our bound on it stands on the pilot corpus.
- **Behavioural coding is bounded by judge quality** — see §9.

---

## 9. Future work

Named explicitly as extensions, none of which blocks the result above. The three
below are the ones that change what the paper can claim; the full sweep of what
a top-venue submission still needs — second task family, second model family,
equivalence testing, multiplicity correction, related work, and the
reproducibility package — is tracked with per-item status in
[`Final Sweep.md`](../Final%20Sweep.md).

**A 14B model.** The pre-registered response to "7–8B may be too weak to
collaborate meaningfully." A 14B fits the same 24 GB card at 4-bit. If a team
advantage appears at 14B where none exists at 8B, that locates a capability
threshold for collaboration — a more interesting finding than either of ours,
and directly enabled by the artifact controls reported here. **Our negative is
explicitly scoped to 8B and does not predict the 14B outcome.**

**Role-label validation at usable agreement.** Our behavioural coding is capped
by a local judge agreeing with a human at **κ = 0.29** after four codebook
revisions (Phase 2 needs ~0.6; convergent validity needs ~0.78). Four books
moved κ from 0.07 to 0.29 and stopped, so this is a model-capacity limit, not a
wording problem. Until it is fixed, the transcript-derived half of the
convergent-validity comparison cannot be interpreted — we have causal
contribution profiles and no trustworthy labels to correlate them against.
Testable on the same 40 messages the moment a larger judge is on disk.

**A task family where collaboration demonstrably pays.** The strongest version
of the original experiment needs an operating point that exists. Finding one —
by scale, by task, or by protocol — is the prerequisite the whole ablation
programme was blocked on.

---

## 10. Conclusion

We did not measure whether emergent roles in LLM teams are causally real. We
measured that three independent significant results favouring teams were
artifacts — two of a token cap that is symmetric in specification and asymmetric
in effect, one of a 48-episode baseline — and that with those removed, team size
has no effect on this task family at 8B.

The negative result is bounded and honest. The mechanism is the contribution:
**a shared per-turn limit is not a shared constraint when the arms spend it
differently**, and the diagnostics that expose it are cheap, general, and
currently absent from standard practice.

Every artifact we removed made the gap smaller. Not once did one make it larger.
