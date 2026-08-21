# Advisor briefing packet

**For the sign-off conversation. Written 2026-08-21.**

Final Sweep 8.1 says to start this now rather than at submission, because it
gates nothing and takes calendar time. This packet is what to bring. It exists
so that the meeting spends its hour on the four decisions at the bottom rather
than on reconstructing what the project did.

---

## 1. The claim, in one paragraph

We set out to test whether emergent role differentiation in same-model LLM agent
teams is causally real, using leave-one-out ablation. We could not run that test.
Across three task difficulties, two serving instruments and three disjoint
corpora we obtained four statistically significant results favouring multi-agent
teams, and traced all four to measurement artifacts in our own harness. With all
four removed, team size does nothing on this task family: one agent scores 0.579,
three agents 0.574, four agents 0.576 across 899 episodes. The only manipulation
that reliably moves the score is giving a *single* agent more turns, which makes
it worse (−0.063, *p* = 0.012). The contribution is the artifact class, the
diagnostics that catch it, and a bounded negative result.

**The artifact class, which is the transferable part:** a limit that is symmetric
by specification lands asymmetrically when the arms consume the resource
differently. A per-turn token cap is identical for both arms, and a single agent
must emit an entire solution in its final turn while a team commits one its
shared transcript already holds. Three of the four artifacts are that shape; the
fourth is an answer-format parser applying one convention to two arms that meet
it at different rates.

---

## 2. What to read, in this order

Reading all of it is not the point. Reading these four in this order is.

| Order | Document | Why this one | Time |
|---|---|---|---|
| 1 | `docs/PAPER-DRAFT.md` — abstract, §1, §5, §5.1 | The claim and the four artifacts | 20 min |
| 2 | `docs/PREREG-equivalence.md` | The discipline the credibility rests on: margin registered before the analysis that used it | 5 min |
| 3 | `docs/RESEARCH-LOG.md` §4.10, §4.13, §4.14, §4.23, §4.24 | The five moments where a positive result was found and then dismantled | 25 min |
| 4 | `Final Sweep.md` §0 | What is done, what is not, and what the remaining GPU budget buys | 5 min |

`docs/RESEARCH-LOG.md` in full is the transparency artifact, not the argument.
It is written in first person and dated; the paper is not. `docs/SUPPLEMENTARY.md`
declares which document is which, and a build-time assertion keeps the two
registers from mixing.

---

## 3. What is defensible, stated plainly

These are the parts to lean on, because they are unusual rather than merely
adequate.

- **Preregistration written before the episodes existed**, amended in dated
  public steps, with a fresh-seed rule that overturned the project's own
  headline. The rule cost this project its positive result; that is the evidence
  that it was real.
- **Every correction moved the result against the author's interest.** Four
  findings favouring teams, four withdrawn. None moved the other way.
- **An instrument that has caught seven silent corruption modes**, the largest of
  which would have published *d* = 1.09 at *p* < 0.0001.
- **The nulls are bounded, not merely unrejected.** TOST against a margin
  registered in advance excludes team-size effects larger than 0.032 on the main
  gate — smaller than one constraint on the task.
- **427 tests from a fresh unauthenticated clone**, in seconds, with no GPU.

---

## 4. The four objections a reviewer will raise, and where the answer lives

Worth rehearsing, because three of the four have real answers and the fourth
does not.

**"A negative result on one task family and one model family."**
Correct, and the paper scopes itself to that explicitly rather than quietly.
Second task family (code generation) and second/third model families are built,
preregistered and unrun — apparatus complete, roughly 30–40 GPU-hours of
generation outstanding. `Final Sweep.md` 1.1–1.4.

**"Failing to reject the null does not support the null."**
Agreed, which is why the paper does not do that. Every null claim carries a TOST
result against a margin registered before the test, and the smallest δ at which
equivalence holds is reported beside it. One claim — fungibility — does *not*
clear the margin and has been corrected to a bound of 0.056 rather than left as
a null. `docs/PREREG-equivalence.md`.

**"You ran many tests."**
Holm-adjusted and BH-FDR values are printed alongside the uncorrected ones for a
declared nine-member family. No conclusion changes, which is the point of doing
it. `src/collabengine/analysis/inference.py`.

**"The κ = 0.29 judge."** *This one has no good answer yet.* Four codebooks moved
inter-rater agreement from 0.07 to 0.29 and stopped, against a κ = 0.78 standard
in the cited prior work. Three resolutions are open, one of which is free: re-run
the 40-message hand-coded validation set through the 14B already on disk, which
settles whether κ is a capacity limit or a wording one. Queued. If it is a
capacity limit, the honest move is demoting Phase 2 to the appendix — the
ablation-side nulls do not depend on it. **This is decision (c) below.**

---

## 5. The four decisions that need a second opinion

Everything above is reporting. These are the questions the meeting is for.

**(a) Venue and track.** This is a measurement and negative-result paper about
evaluation practice. The argument in `docs/VENUE.md` is that it fits a
datasets-and-benchmarks track better than a main track, and an evaluation
workshop very well as a fallback that still yields a real publication. The
counter-argument is that the artifact class is a general claim about how
architectures are compared, which is main-track material, and that aiming low
early forecloses it. **Needed: a call on ambition versus certainty.**

**(b) Whether to spend the remaining 30–40 GPU-hours before submitting, or
submit scoped.** The second task family is the single biggest weakness and the
apparatus for it is finished. Running it converts "a bug in one harness" into "a
class of measurement error". Not running it means the title and abstract must
stay scoped to one task family and one model family — which they already do, and
which must not be quietly relaxed under deadline pressure. **Needed: a read on
whether the scoped version is submittable at the chosen venue.**

**(c) The κ = 0.29 judge.** Replace it, demote Phase 2 to the appendix, or find
a paid frontier key. The free 14B check narrows this but does not decide it.
**Needed: a judgement on whether a κ = 0.29 instrument may appear in a body
section at all.**

**(d) The title.** "Three Positives and a Cap" is now wrong on both halves —
§5.1 makes it four, and the fourth is a parsing convention rather than a cap.
Candidates and their trade-offs are in `docs/TITLE-OPTIONS.md`, built around the
generalisation *symmetric by specification, asymmetric in effect*. Flagged in the
draft rather than changed unilaterally. **Needed: a decision.**

---

## 6. What is not being asked

Not asked: whether to redo the science. The result has survived three
instruments, five operating points and two disjoint seed sets, and everything
outstanding in `Final Sweep.md` is breadth or packaging. A conversation that
reopens the experimental design has misread the state of the project — the
disciplines in §3 are the expensive part and are already paid for.
