# Preregistration — the code task family

**Registered 2026-08-22 12:50, after the episodes were generated and before any
of them were scored.** That is weaker than the project's other preregistrations
and the difference is stated here rather than glossed: `PREREG-14b.md` was
written while the weights sat unused on disk, and this one was not.

**What is still true of it.** The 1,050 episodes exist on disk and no gate, no
mean and no truncation count has been computed from them. `queue-tier1.sh`
prints `GATE IS READABLE` and computes nothing; the per-rung gates that
`cap-sweep.sh` printed have no equivalent here. So the hypotheses, the
threshold and the falsification condition below are fixed while the answers are
genuinely unknown, which is the property that makes a registration worth
anything. What is lost is the guarantee that the *design* was not tuned to the
data — and since the design was frozen in `configs/llamacpp/code-medium.yaml`
and 24 tests before generation started, that loss is small but real.

**Why it was late.** Final Sweep 1.1 was built and run as apparatus while the
registration was tracked as prose to write later, and the queue reached it
first. Final Sweep §11 says plainly that a new task family does not get to skip
its preregistration because the deadline is close. It nearly did.

---

## 1. Why this arm exists

One synthetic scheduling task is the study's single biggest scope weakness. The
paper's claim is not about scheduling: it is that **a limit which is symmetric
by specification lands asymmetrically when the arms consume the resource
differently**. If that is a property of instruments rather than of one harness,
it must appear in an unrelated domain with an unrelated grader.

Code generation is the most adversarial available venue for the claim, which is
why it was chosen over math or multi-hop QA. A single agent must emit an entire
module inside its final turn while a team commits one its transcript already
holds. Math suppresses the mechanism because answers are short; code maximises
it.

## 2. Instrument

| | |
|---|---|
| Model | Meta-Llama-3.1-8B-Instruct, Q4_K_M |
| Serving | `llama.cpp` b10369 (6e62ba538), 4 slots × 18,432 tokens |
| Config | `configs/llamacpp/code-medium.yaml` |
| Seeds | 1000–1149, fresh-seed range, disjoint from the 0–47 pilot |
| Arms | `baseline` (4 agents × 3 rounds), `solo` (1 × 3), `solo_long` (1 × 12) |
| Ablation | live, A1–A4 |
| Answer budget | `answer_max_tokens = 3072`, unchanged from the scheduling instrument |
| Grading | hidden test suite, four independent components, sandboxed |

**`team.task` is the only line that differs from `medium-h3b.yaml` in any way
that touches the model.** Same weights, same quantisation, same slot geometry,
same seeds, same rounds, same per-turn cap, same answer budget. Changing the
task and anything else in one run is §4.1c's mistake.

## 3. Hypotheses and thresholds

**C1 — the artifact transfers.** The single-agent arm generates more text per
episode than the team arm at the same turn budget, and is truncated at the
answer turn more often.

> *Registered threshold:* generation ratio (solo characters ÷ team characters)
> **> 1.5**, and answer-turn truncation at least **10 percentage points**
> higher in `solo` than in `baseline`. The scheduling family showed 2.04× and
> 34% vs 8%; this predicts the same direction, not the same magnitude.
>
> *Falsified if:* the ratio is below 1.2, or the truncation gap is under 5
> points, or the sign reverses. **A falsification here is the most informative
> outcome available in this document** — it would mean the artifact is a
> property of the scheduling task's answer shape rather than of the instrument,
> and the paper's generalisation from four artifacts to a class would have to
> be withdrawn to a claim about one task family.

**C2 — the gate is null once the artifact is controlled.** Team and correctly
briefed solo do not differ on `fraction` beyond the registered margin.

> *Registered threshold:* equivalence at **δ = 0.05**, the margin already fixed
> in `PREREG-equivalence.md` and not re-chosen for this family.
>
> *Predicted:* equivalence holds. *If a team advantage clears +0.05 and
> survives the truncation correction*, that is a real finding and the paper's
> null becomes task-family-dependent — which is a different and more
> interesting paper than the one currently drafted.

**C3 — `solo_long` does not rescue the single agent.** More turns for one agent
does not raise its score above the team's.

> *Predicted:* the scheduling family's −0.063 (*p* = 0.012) direction repeats.
> No magnitude is registered; this is directional only.

## 4. Power, from a prior rather than from these episodes

`sd` on `fraction` is taken from the scheduling family's realised value,
**0.15**, because no code-family spread has been computed. At *n* = 150 per arm
that gives an MDE of about **0.049** at 80% power and α = 0.05 — adequate for
the δ = 0.05 margin in C2 and for nothing finer.

If the realised code-family `sd` is materially larger — plausible, because a
hidden test suite is coarser than a fraction-of-constraints score and may be
closer to bimodal — then this arm is underpowered for C2 and the honest report
is a wide equivalence bound rather than a null. **That check runs before C2 is
read**, and the bound is reported either way. This is the §4.22 lesson: the
48-episode pilot's MDE was 0.086 against the +0.055 it reported, and one
pre-run line would have said so.

## 5. Analysis, fixed in advance

1. `is_instrument_failure` filtering, unchanged.
2. Truncation and generation accounting **printed beside every mean**, per the
   standing rule that a calibration number without it measures the cap.
3. Malformed-answer counts per arm, per §4.24 — the fourth artifact was an
   answer-format convention and this family has its own parser.
4. TOST at δ = 0.05, plus the smallest δ at which equivalence holds.
5. Holm correction across C1–C3 as one family.

No metric is added after seeing the output. `fraction` is primary; `strict` is
secondary and reported alongside.

## 6. What would make this arm uninterpretable

Stated in advance so it cannot be argued afterwards:

- A floor effect. If the team arm's mean `fraction` is below **0.15**, the 8B
  is too weak on this task for a solo-vs-team contrast to mean anything, and
  the arm is reported as a floor rather than as a null. PLAN §3 warns about
  exactly this for code tasks.
- A malformed rate above 30% in either arm, which would make the parser rather
  than the model the dominant term.

---

## Postscript — outcome, 2026-08-22

Scored the day it was registered, in the order this document fixes: the §6
interpretability gates and the §4 power check first, C1 second, C2 third.

| clause | registered | measured | verdict |
|---|---|---|---|
| §6 floor | team > 0.15 | 0.7275 | interpretable |
| §6 parser | malformed < 30% | solo 12.0%, team 1.3% | interpretable |
| §4 power | sd 0.15, MDE 0.049 | **sd 0.29–0.37, MDE 0.094–0.121** | **underpowered** |
| C1a | ratio > 1.5 | **1.78** | confirmed |
| C1b | truncation gap >= 10 pp | **0 pp** | **falsified** |
| C2 | equivalent at 0.05 | *p* = 0.0801, bound 0.0585 | **bound, not null** |
| C3 | solo_long negative | **+0.053** | **falsified** |

**C1b is the outcome this document named as most informative, and it happened.**
The verbosity asymmetry transfers — the single agent still writes 1.78x more in
its answer turn — but at a 3,072-token budget nothing is truncated in either
arm. The cap did not bite on this task, so the mechanism that produced three of
the four scheduling artifacts is absent here.

**And yet the arm still shows a +0.0783 apparent team advantage, all of it the
fourth artifact.** Excluding unparseable answers the gap is **-0.0004**. The
class transfers; the member does not. That is a more interesting result than a
clean replication would have been, and it is not the result this document
predicted.

**The power clause did its job.** §4 said in advance that a materially larger
realised `sd` would mean reporting a bound rather than a null. The realised sd
is 2 to 2.5x the assumed value, the MDE is above the registered margin, and the
bound is what is reported. Written after the fact this would be an excuse;
written before, it is a finding about how coarse a hidden-test-suite score is
compared with a fraction-of-constraints score.

**C3's reversal is unexplained and left that way.** `solo_long` has half the
malformed rate of `solo` (6.0% against 12.0%), so some of +0.053 is parser
rather than reasoning. The decomposition has not been run and is not asserted.
