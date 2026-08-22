# Preregistration — the model-family and precision arms

**Registered 2026-08-22 16:20.** Covers four gate-only arms: Qwen2.5-7B,
Mistral-7B-v0.3, Llama-3.1-8B at Q8_0, and the same at f16.

**Registration status differs per arm and is stated per arm rather than
averaged.** One of the four has already generated:

| Arm | Episodes at registration | Scored at registration |
|---|---|---|
| qwen 7B | complete (450) | **no** |
| mistral 7B | generating | no |
| llama Q8_0 | not started | no |
| llama f16 | not started | no |

> **Amendment, 2026-08-22 16:16 — the mistral arm is withdrawn, and not for a
> reason that a rerun fixes.** Preflight refused it four minutes after this
> document was written: `Mistral-7B-Instruct-v0.3`'s chat template raises
> `Conversation roles must alternate user/assistant/...` and supports no
> `system` role. `orchestrator/episode.py` sends a system message and, in the
> team arm, consecutive turns from the same role — so the study's brief
> structure is not expressible in that template at all.
>
> **Running it anyway would have confounded family with prompt assembly.** The
> fix is either to serve Mistral under a template it was not trained on, or to
> rewrite message assembly for one family only. Both change the instrument in
> exactly the arm meant to isolate the model, which is §4.1c's mistake with a
> different label. The arm is dropped rather than repaired.
>
> **What is lost is small and should be stated as such.** F1 asks whether the
> null is a Llama quirk; qwen answers it with a second family, and mistral was
> always the more redundant of the two. What is gained is a real limitation:
> **the study's multi-agent brief is not portable across chat templates**, which
> constrains how any harness built this way can be run across model families —
> and is worth a sentence in the paper's limitations rather than a silent gap in
> a table.
>
> The preflight did its job: it cost one probe request instead of a card-night,
> and `queue-tier1.sh` moved to the next arm rather than stopping, because the
> arms answer unrelated questions.

Nothing in any of the four has been scored. `queue-tier1.sh` prints
`GATE IS READABLE` and computes nothing, so no mean, gate or truncation count
from these arms has been seen by anyone. The qwen arm is therefore registered
in the same weaker position as `PREREG-code.md`: design frozen before
generation, thresholds fixed before analysis, but not before the episodes
existed. The other three are registered before generation in the ordinary way.

Written now rather than after the queue drains, because the code family reached
completion with no registration at all and that is the second time in one day
that apparatus outran its own paperwork.

---

## 1. Why these arms exist, and what they cannot answer

Two questions, both narrower than the ones the main corpus answers:

- **Is the null a Llama quirk?** — qwen 7B, mistral 7B.
- **Is the null an artifact of 4-bit weights?** — Q8_0, f16.

`PAPER-DRAFT.md` §9 raises the second explicitly: *if 4-bit weights degrade
long-context instruction-following more, the team arm absorbs more of the loss.*
That sentence is currently argued. These arms make it measured.

**What they cannot answer.** They are `pipeline` only — `baseline`, `solo`,
`solo_long`, no ablation grid. They therefore say nothing about role
differentiation, agent fungibility, or anything else the ablation measures.
That is deliberate: six ablation grids would cost roughly fifteen further
card-hours to re-measure a null the headline corpus already bounds, and the
question here is a gate question. **If a gate moves in any of these four arms,
its ablation becomes the obvious follow-up and is not in scope for this
document.**

## 2. Instrument

| | |
|---|---|
| Serving | `llama.cpp` b10369 (6e62ba538), 18,432 tokens per slot in every arm |
| Seeds | 1000–1149, the fresh-seed range, identical across all four |
| Arms | `baseline` (4 × 3), `solo` (1 × 3), `solo_long` (1 × 12) |
| Configs | `medium-qwen`, `medium-mistral`, `medium-q8`, `medium-f16` |

Each config differs from `medium-h3b.yaml` by `backend.model` and nothing else
that touches the model. **Slot count varies and per-slot context does not** —
qwen and mistral take 4 slots, Q8_0 takes 4, f16 takes 2, because KV cost and
weight size differ. Throughput changes; the measurement does not. Holding
per-slot context at 18,432 is the one quantity that must not vary, since it is
the cap whose asymmetry the whole study is about.

## 3. Hypotheses and thresholds

**F1 — the solo-vs-team null replicates off Llama.** For qwen and mistral
separately, team and correctly-briefed solo do not differ on `fraction` beyond
the registered margin.

> *Threshold:* equivalence at **δ = 0.05**, the margin fixed in
> `PREREG-equivalence.md` and deliberately not re-chosen here.
>
> *Predicted:* equivalence holds in both. A team advantage clearing +0.05 in
> either family would mean the paper's null is Llama-specific, and the scope
> sentence in §9 would have to name the family rather than the scale.

**F2 — the null replicates across precision.** For Q8_0 and f16 against the
Q4_K_M headline, same margin, same test.

> *Predicted:* equivalence holds. **The interaction is the claim, not the main
> effect.** Higher precision will very likely raise both arms; that proves
> nothing. What §9 asserts is that 4-bit costs the *team arm specifically*
> more, so the quantity registered here is the **change in the team-minus-solo
> gap** across precision rungs, not the change in either arm's mean.
>
> *Falsified if:* the team-minus-solo gap grows by more than **0.05** from
> Q4_K_M to f16. That would support §9's own limitation and is the one outcome
> in this document that would damage the paper's headline.

**F3 — the cap asymmetry is instrument-wide, not model-specific.** The
generation ratio between solo and team exceeds 1.5 in every arm.

> *Predicted:* yes, in all four. This is the same mechanism the scheduling and
> code families show, and a model that did not exhibit it would be the
> interesting case.

## 4. Power

`sd` = 0.15 from the scheduling family's realised spread. At *n* = 150 per arm,
MDE ≈ **0.049** at 80% power, α = 0.05 — adequate for a δ = 0.05 margin and
nothing finer. The precision *interaction* in F2 is a difference of differences
and is correspondingly less powered; **if its bound is wider than 0.05 the
honest report is that this study cannot resolve the interaction**, not that the
interaction is absent. §4.22's lesson, again: the 48-episode pilot's MDE was
0.086 against the +0.055 it reported.

## 5. Analysis, fixed in advance

1. `is_instrument_failure` filtering, unchanged.
2. Truncation, generation ratio and malformed-answer counts printed beside
   every mean, per §4.24 — each of these models has its own answer-format
   behaviour and the fourth artifact was a parser convention.
3. TOST at δ = 0.05 plus the smallest δ at which equivalence holds, per arm.
4. Holm correction across F1's two families and F2's two rungs as one family of
   four.
5. `fraction` primary, `strict` secondary and reported alongside.

## 6. What would make an arm uninterpretable

- Team mean `fraction` below **0.15** — a floor, reported as a floor.
- A malformed rate above 30% in either arm, making the parser the dominant term
  rather than the model. **This is the likeliest failure here**: the
  answer-format parser was written against Llama's output conventions, and
  §4.24 already found it scoring 25 well-formed-but-unconventional solo answers
  at zero. Qwen and Mistral have no reason to share Llama's habits, and a
  parser that penalises them is a measurement of the parser.
