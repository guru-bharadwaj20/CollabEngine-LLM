# Retitling the paper

**Final Sweep 3.4. Assembled 2026-08-21. This is an author decision and it has
not been made here — `docs/PAPER-DRAFT.md`, `README.md` and
`scripts/analysis/build_paper.py` are untouched.**

---

## Why "Three Positives and a Cap" is now wrong, on both halves

It was accurate on 2026-08-15. It is not accurate now, and it failed in the two
ways a title with a fact in it can fail.

**The count moved.** `docs/PAPER-DRAFT.md` §5.1 adds a fourth artifact, found
when the corpus was rebuilt from seeds after `runs/` was cleaned
(`docs/RESEARCH-LOG.md` §4.24). Read as scored, the rebuilt gate reverses the
paper's headline: `fraction` +0.081, *d* = 0.43, *p* < 0.001 in the team's
favour. What it is instead: **25 unparseable single-agent answers scored 0.000
and counted, against 0 in the team arm.** Twenty-five zeros in one arm move that
arm's mean by 0.106, which is more than the entire gap. Restricted to episodes
where both arms produced a parseable answer, the single agent leads by 0.025 and
the published null reproduces. The abstract now says four; the title says three.

**The mechanism moved too, and this is the more serious half.** The fourth
artifact is not a token cap. It is the answer-format parser. `TEAM_BRIEF` tells
an agent that the group's last message is what gets scored, so a team's final
turn is written *as* a submission; a single agent under `SOLO_BRIEF` is under no
such pressure and sometimes emits a complete, correct assignment in prose.
`solo:medium:1013` contains all sixteen job assignments, `finish_reason = stop`,
and scores exactly zero. A title that ends in "and a Cap" names the cause of
three of four artifacts and asserts, wrongly, that the resource limit is the
whole story. §7 of the draft makes the opposite point explicitly:
recommendation 7 generalises past resource limits to **any convention identical
in specification**.

**The lesson to carry into the replacement: do not put the count in the title.**
It has broken once, the diagnostic that caught the fourth artifact is still
running against a corpus in progress, and a fifth is not hypothetical — §4.24
leaves two instrument questions open. "Four Positives and a Convention" would be
the same trap rebuilt one number to the right.

---

## The generalisation that covers all four

**Symmetric by specification, asymmetric in effect.**

Each of the four is one limit or convention, identical in the specification for
both arms, landing on one arm only, and in the direction that flatters the team:

| # | The shared thing | Why it lands asymmetrically |
|---|---|---|
| 1, 2 | Per-turn token cap (§4.10, §4.14) | Solo must emit an entire solution in its final turn; a team commits an answer its shared transcript already holds |
| 3 | The 48-episode four-agent baseline | Not a shared limit but a shared estimator, and a 3× larger sample did not reproduce it |
| 4 | The answer-format parser (§5.1, §4.24) | `TEAM_BRIEF` makes a submission block the natural final turn; `SOLO_BRIEF` does not |

Artifact 3 is the honest wrinkle in the generalisation: a small-sample baseline
is not a specification landing asymmetrically, it is an underpowered estimate.
Any title built on the symmetry phrase covers three of four exactly and the
third by family resemblance. That is a reason to keep the subtitle broad, not a
reason to abandon the phrase — the paper's own §7 recommendations and its
concurrent-work comparison in §1 are both organised around it.

---

## Candidates

Each keeps the existing subtitle, **"Measurement artifacts in single-agent versus
multi-agent LLM comparisons,"** unless noted, since that line is already
accurate and already in the docx.

### A. Symmetric by Specification, Asymmetric in Effect

*Promises:* the mechanism, in the paper's own words, with no count to go stale.
A reviewer who reads only the title knows the claim is about measurement rather
than about agents, and knows the shape of the failure mode before the abstract.

*Risks:* abstract on its own — nothing in it says LLM, agent or evaluation, so
it depends entirely on the subtitle carrying the domain. Nine words of Latinate
vocabulary before the reader learns what field this is.

### B. Four Positives and a Convention

*Promises:* continuity with the current title and the same concrete hook, updated
to the correct count and the correct fourth cause.

*Risks:* it rebuilds the exact failure being repaired. The count is a fact in a
title that has already changed once, and the diagnostics are still running.
"Convention" is also weaker than "cap" as an image — a reader will not know what
it refers to until §5.1.

### C. The Same Limit Is Not the Same Limit

*Promises:* memorable, and it states the paradox in seven plain words.

*Risks:* too cute for a measurement paper whose credibility rests on being
unshowy, and it buries the mechanism rather than naming it. It also says "limit,"
which is precisely the over-narrow reading that recommendation 7 corrects.

### D. When a Shared Limit Is Not a Fair One: Arm-Asymmetric Measurement Artifacts in Multi-Agent LLM Evaluation

*Promises:* domain and mechanism in one line, no subtitle needed, and
"arm-asymmetric" is the exact technical handle a reviewer would search for.

*Risks:* "fair" imports a fairness-literature reading the paper does not make and
does not want. Long enough that it will be truncated in listings, and the
truncation drops the half that carries the mechanism.

### E. Measurement Artifacts in Single-Agent versus Multi-Agent LLM Comparisons

The deliberately plain option: promote the current subtitle and delete the
headline entirely.

*Promises:* it cannot become wrong. It states the topic, the comparison and the
domain, and it is the title a reviewer would write for this paper if asked to
describe it neutrally.

*Risks:* indistinguishable from a survey. It promises a catalogue and delivers a
mechanism plus a null, which undersells both. Nothing in it is memorable, and at
a venue where the same session holds a dozen evaluation papers, being forgettable
has a cost.

### F. Symmetric by Specification, Asymmetric in Effect: How Shared Limits Manufacture Multi-Agent Advantage

*Promises:* mechanism plus consequence, and "manufacture" is the strongest verb
available for what actually happened — four significant results produced by the
instrument.

*Risks:* over-claims on evidence the paper does not have. "Manufacture
multi-agent advantage" asserts a general property of the literature from one
task family, one model family and one scale, and §8 limits the claim explicitly
to a bounded null. It also reads as an accusation against the five frameworks
named in §2, which the paper is careful not to make.

---

## Recommendation

**A — "Symmetric by Specification, Asymmetric in Effect," keeping the existing
subtitle.**

Three reasons.

**It cannot go stale.** The count broke this title once and the corpus work that
would break it again is still running. A title built on the mechanism survives a
fifth artifact; a title built on a count does not.

**It is already the paper's own thesis sentence.** The abstract uses the phrase
verbatim for the token cap, §5.1 uses it verbatim for the parser, and §7
recommendation 8 generalises it to context windows, wall-clock budgets and
tool-call quotas. Nothing has to be written to justify the title — the argument
for it is the paper.

**It claims exactly what is supported and nothing more.** It asserts a mechanism
about instruments, not a verdict about multi-agent systems, which is the line §8
draws and the line D and F both cross. The subtitle carries the domain, so the
title's abstractness — the real objection to A — is answered on the same line.

The plain option E is the correct fallback if a reviewer or an advisor finds A
too abstract to stand alone. It loses memorability and loses nothing else, and
between the two the risk is asymmetric: A risks being read as vague, E risks
being read as a survey, and vague is the cheaper failure at an evaluation track
where the abstract is read.

Rejected outright: **B**, for rebuilding the fault being fixed, and **F**, for
over-claiming past §8.

**This is the author's call.** Nothing in `PAPER-DRAFT.md`, `README.md` or
`build_paper.py` has been changed. When it is made, the title appears in
`docs/PAPER-DRAFT.md` line 1, the note beneath it flagging the mismatch comes
out, and `Final Sweep.md` item 3.4 closes.

---

## Recommendation, 2026-08-22 — revisited after the results

The measurements changed what the title has to survive, so the pick is restated
here rather than left as written before the data existed.

**What changed.** The code family showed that the *token cap* does not transfer
(0% truncation there) while the *class* does — the artifact reappeared as an
answer-format convention instead. And the parser artifact turned out to be
**Llama-specific**: 16.8% of Llama's solo answers unparseable against 0.0% of
Qwen's. So the count of artifacts is now unstable across task families and model
families both. **Any title built on a number, or on the word "cap", will be
wrong again within one experiment.**

### The pick

> # Symmetric by Specification, Asymmetric in Effect
> ### Measurement artifacts in single-agent versus multi-agent LLM comparisons

**Why this one survives where the others do not:**

1. **It cannot go stale.** It names the mechanism class, not an instance count.
   "Three Positives and a Cap" broke twice in a week — first when §5.1 made it
   four, then when the fourth turned out not to be a cap.
2. **It is already the paper's own thesis sentence**, verbatim, in the abstract,
   §5.1 and §7. The title should be the claim, not a label stuck on top of it.
3. **It claims a property of instruments, not a verdict on multi-agent systems.**
   That is the line §8 draws and the line two concurrent papers make it important
   to hold: Tran & Kiela and Ringelmann (arXiv:2606.02646) both already argue
   multi-agent does not help. This paper's distinct claim is *why the literature
   keeps reporting that it does*.
4. **It generalises exactly as far as the evidence does.** The cap sweep, the
   code family and the Llama/Qwen split are three instances of one class and
   nothing more.

**Fallback** remains the plain descriptive option E, if a reviewer or an advisor
finds the phrase too abstract for a first encounter.

**Rejected, and now more firmly than before:**

- *"Four Positives and a Convention"* — rebuilds the exact fault that broke the
  current title, and the count is now known to be task- and model-dependent.
- Anything containing *"cap"* — falsified as a general mechanism by the code
  family's 0% truncation.
- Anything asserting multi-agent systems do not work — over-claims past §8, and
  walks directly into two concurrent papers that own that conclusion.

**Still the author's call.** Nothing in the build has been retitled.
