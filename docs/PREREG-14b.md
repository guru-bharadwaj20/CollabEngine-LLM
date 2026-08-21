# Preregistration — the 14B arm

**Registered 2026-08-21, before a single 14B episode existed.** The weights were
on disk when this was written and no inference had been run against them.

This is the most publication-favourable arm in the study, which is precisely why
it is registered. A positive here converts a negative result into a
capability-threshold finding; that incentive is the reason the prediction, the
threshold and the stopping rule are fixed in advance rather than read off the
output.

---

## 1. Why this arm exists

`PLAN.md` §6 lists "7–8B too weak to collaborate meaningfully" as a *medium*
likelihood risk and names the response: "step up to 14B (still fits 24 GB at
Q4)". `PAPER-DRAFT.md` §9 repeats it as future work and states plainly that
**our negative is scoped to 8B and does not predict the 14B outcome.** This
document is that step-up, run under the artifact controls the project spent
seven corrections building.

The question is not "does 14B score higher". It obviously will. The question is
whether **the gap between a team and a correctly-briefed single agent** is
different at 14B than at 8B, once the token-cap artifact is removed from both.

## 2. Instrument

| | |
|---|---|
| Model | Qwen2.5-14B-Instruct, Q4_K_M, sharded GGUF |
| Serving | `llama.cpp` b10369 (6e62ba538), **3 slots × 18,432 tokens** |
| Config | `configs/llamacpp/medium-14b.yaml` |
| Seeds | 1000–1149, the fresh-seed range, disjoint from every pilot |
| Arms | `baseline` (4 agents × 3 rounds), `solo` (1 × 3), `solo_long` (1 × 12, `SOLO_BRIEF`) |
| Answer budget | `answer_max_tokens = 3072`, unchanged from the 8B instrument |

**Three slots rather than four is a memory fact, not a choice.** Qwen2.5-14B is
48 layers × 8 KV heads × 128, so KV costs 192 KiB/token against Llama-8B's 128,
and the weights are 8.9 GiB against 4.6. Four slots would want ~22.4 GiB on a
shared 24 GiB card, and Windows pages rather than raising OOM (§3.4). Per-slot
context is held at 18,432 — the one quantity that must not vary across arms.
Throughput changes; the measurement does not.

**A model-family change rides along with the scale change and is declared
here.** The only 14B that fits this card at Q4 is not a Llama, so this arm
varies scale *and* family together. It therefore cannot separate them on its
own. `configs/llamacpp/medium-qwen.yaml` (Qwen2.5-**7B**, same quantisation,
same instrument) is what separates them, and **it must be run before this arm is
interpreted** — Qwen-7B against Llama-8B is the family contrast at fixed scale,
and Qwen-7B against Qwen-14B is the scale contrast at fixed family. Reading this
arm without that one is the confound this document exists to refuse.

## 3. Power, computed before the run

`sd` prior = 0.15 on `fraction`, from the 8B four-agent arm (RESEARCH-LOG §4.22).

| arm | planned *n* | MDE at 80% power |
|---|---|---|
| each of `baseline`, `solo`, `solo_long` | 150 | **0.049** |

So this arm can resolve a difference of about one satisfied constraint and no
smaller. **It cannot resolve the 0.005 spread observed across team sizes at
8B** — that would take ~14,000 episodes per arm — and no amount of staring at
the output will change that. The registered analysis is therefore an
equivalence test, not a search for significance.

## 4. Hypotheses

**H14-1 (primary, confirmatory).** At `medium`, the team–`solo_long` gap on
`fraction` at 14B is **larger than at 8B**.
- 8B reference: +0.003, equivalence bound 0.032 (`PREREG-equivalence`, E2).
- **Threshold for a capability effect: the 14B gap exceeds +0.05** — one
  satisfied constraint — *and* its 95% bootstrap interval excludes zero *and*
  it survives the answer-turn truncation sensitivity row.
- All three clauses are required. Any one alone has already produced a
  withdrawn finding in this project.

**H14-2 (primary, confirmatory).** If H14-1 does not fire, the 14B gap is
**equivalent to zero** at δ = 0.05 (TOST, `PREREG-equivalence` margin, unchanged).

**H14-3 (secondary).** The generation asymmetry (solo ÷ team characters on a
matched turn budget) at 14B is in the same direction as at 8B, where it is
2.04×. Reported with the answer-turn truncation counts beside it, per
recommendation 1 of the paper.

## 5. What would falsify what

| outcome | reading |
|---|---|
| H14-1 fires, all three clauses | **A capability threshold for collaboration lies between 8B and 14B.** The paper gains a positive result and its scope claim tightens to "at 8B" rather than "at this scale" |
| H14-2 holds, H14-1 does not | The negative extends to 14B with a bound. The paper's scope widens and the claim is *strengthened*, not weakened |
| Neither: gap positive but under threshold, or interval includes zero | **Underpowered, and reported as underpowered.** No claim either way. The honest sentence is the equivalence bound actually achieved |
| Truncation counts differ sharply between arms | The artifact reproduces at 14B, which is itself a Tier-1 result for the paper's central mechanism |

## 6. Rules fixed in advance

1. **The fresh-seed rule applies.** Seeds 1000–1149. Nothing from this corpus
   may test a hypothesis this corpus generated.
2. **No arm is dropped for being slow.** Three slots makes this arm roughly
   twice the wall clock of the 8B one. That is not a reason to cut *n* to 48,
   which is the sample size that produced the project's withdrawn headline.
3. **The sensitivity row is printed beside every number**, per
   `PREREG-xhard`'s discipline, which has been right three times.
4. **Multiplicity.** H14-1 and H14-2 join the Phase 1 gate family for Holm
   adjustment (`analysis.inference`). Registering two hypotheses and reading
   each at 0.05 is what this project already did wrong once.
5. **`medium` only, first.** `hard` is run only if `medium` shows movement.
   Running both and reporting the better one is not a plan.

## 7. The prediction, on the record

**I expect H14-2, not H14-1.** The 8B result is not that the team is slightly
behind — it is that team size does nothing, with one agent, three agents and
four agents inside 0.005 of each other across 899 episodes, and with the only
reliable effect being that *more turns make a single agent worse*. Nothing in
that pattern looks like a model straining against a capability ceiling.

Recorded because it is the unflattering prediction, and because if H14-1 does
fire, this paragraph is what makes it worth believing.

---

*Amendments below this line, dated, with the reason. Nothing above it is edited
after registration.*
