# Venue and deadline

**Final Sweep 8.3. Assembled 2026-08-21. This is decision material, not a
decision — nothing here is wired into `build_paper.py` and no limit is set.**

Every date below was checked against the venue's own page on 2026-08-21 and is
sourced at the bottom of this file. Anything not confirmed on a primary source
is labelled **unverified** and stays that way until someone checks it. This
project has already caught two invented citation titles; a plausible-looking
deadline is the same failure with a worse consequence.

---

## What is being submitted

A measurement paper with a negative result at its centre. The intended
contribution — a causal test of emergent role differentiation — was never run,
because the operating point it needs does not exist: four separate significant
results favouring multi-agent teams all resolved into artifacts of the
measurement, and with them removed the effect of team size is flat to 0.005
across 899 episodes. What is left to publish is the artifact mechanism, the
three diagnostics that caught it, and a bounded null.

That shape decides the venue more than the topic does. It is a paper about
evaluation practice and it has no method to propose. At a main track it competes
in a form built to reward new methods, with one task family, one model family
and one scale — the Final Sweep's own Tier 1 calls scope the biggest weakness,
and a main-track reviewer will reach for it in the same place. At an evaluation
track the identical text is on-topic by construction.

**The first finding of this memo is that the venue the docx is named after has
already closed.** `CollabEngine-NeurIPS2026.docx` is a filename, not a plan.

---

## The dates

### Closed for this cycle

| Venue | Track | Deadline | Page limit | Status |
|---|---|---|---|---|
| NeurIPS 2026 | Main | abstract 4 May 2026, full paper 6 May 2026 AoE | 9 pages of content; acknowledgements, references, checklist and appendices excluded | **Closed.** Notifications 24 Sep 2026 |
| NeurIPS 2026 | Evaluations & Datasets | abstract 4 May 2026, full paper 6 May 2026 AoE | not stated in the track CFP or its FAQ; the FAQ refers authors to the main-track formatting instructions, so 9 pages by inference — **unverified as a track-specific number** | **Closed.** Notifications 24 Sep 2026 |
| ICML 2026 | Main | 28 Jan 2026 | 8 pages main text; references, impact statement and appendices unlimited | **Closed** |
| COLM 2026 | Main | abstract 26 Mar 2026, paper 31 Mar 2026 | not checked — **unverified** | **Closed** |
| EMNLP 2026 | via ARR May cycle | ARR submission 25 May 2026; EMNLP commitment 2 Aug 2026 | not checked — **unverified** | **Closed.** The commitment window ended nineteen days ago |

The Datasets & Benchmarks track has been renamed **Evaluations & Datasets** for
NeurIPS 2026, with its scope widened to treat evaluation as an object of study
rather than a service to methods. Its FAQ says, verbatim: *"Negative results, as
long as they bring new insights and are thoroughly demonstrated via empirical
evaluations, are welcome in ED track,"* and lists *failure modes of current
benchmarks* among the examples. It also states that a dataset is not required —
a submission may be empirical insight about evaluation with no new data. That is
this paper's abstract restated as a call for papers, which is why it anchors the
recommendation below even though the 2026 instance is gone.

### Open

| Venue | Track | Deadline | Page limit | Fit | Risk |
|---|---|---|---|---|---|
| **NeurIPS 2026 workshop** | TAE, *Can We Trust AI Evaluation?* | **29 Aug 2026 AoE**; reviews 14 Sep, notification 22 Sep 2026 | **unverified** — the CFP page returned HTTP 503 on 2026-08-21 and no limit was confirmed | Best topical match found anywhere. Its listed topics include robustness and stability of evaluation conclusions across conditions, and measurement validity and causal assumptions in evaluation protocols. Both name this paper | Non-archival. Eight days out. The limit must be read off the site before anyone writes to it |
| **NeurIPS 2026 workshop** | IAEval, *Evaluation of Interactive Agents* | **29 Aug 2026 AoE**, notification 29 Sep 2026 | 9 pages full, 4 short, excluding references and appendices; NeurIPS 2026 style; double-blind | Topics include collaborative agents, trajectory-level evaluation and grader design. The 9-page limit is the draft's current shape, so the full paper goes in without restructuring | Non-archival. An agent-capability audience can read a null on one synthetic task as narrow |
| **NeurIPS 2026 workshop** | JUDGe, *Can We Trust the Judge?* | 29 Aug 2026 AoE — an aggregator listed 30 Aug, the workshop's own site says 29, so **treat 29 as the deadline**; notification 29 Sep 2026 | 6 pages + references full, 4 short, 2 junior spotlight | Evaluator reliability as a systems problem. The κ = 0.29 judge and the answer-format parser both belong here | Non-archival. Six pages forces a rewrite rather than a trim. Work already published at a major ML venue is ineligible |
| **ICLR 2027** | Main | abstract **18 Sep 2026** AoE, paper **25 Sep 2026** AoE; reviews 5 Nov, decisions 16 Dec 2026 | 9 pages main text at submission, 10 at camera-ready; references, appendices, ethics, reproducibility and AI-use statements all excluded | The only archival main-conference deadline reachable this cycle. The limit already matches the draft, and the reproducibility statement is free of it, which suits a paper whose corpus is the argument | ICLR has no evaluations track. A bounded null on one task, one model family and one scale is scored against method novelty. Five weeks does not close the Tier 1 scope items a reviewer will name first |
| **ARR October 2026** | → NAACL 2027 / COLING 2027 | submission **12 Oct 2026**; cycle closes and commitment due 20 Dec 2026 | ARR long-paper limit not checked — **unverified** | Reviews arrive detached from a venue decision, so the scores can be carried forward rather than spent. Seven weeks of runway rather than five | NLP reviewing rewards task and language breadth, and a synthetic scheduling corpus is an odd fit for it. Commitment lands over the December holidays |
| **ARR January 2027** | → ACL 2027 | January 2027, **exact date unverified** | **unverified** | As above with two more months of runway | As above, plus nothing about the ACL 2027 timeline is published yet |
| **NeurIPS 2027** | Evaluations & Datasets | **unverified.** Not announced. The 2026 instance ran abstract 4 May / paper 6 May, so ~May 2027 is a pattern, not a fact | **unverified** | The one venue whose call for papers describes this paper | Nine months away, and none of it is announced |
| **ICML 2027** | Main | **unverified.** Not announced; ICML 2024 and 2026 closed 1 Feb and 28 Jan, so late January 2027 is a pattern | 8 pages at ICML 2026; 2027 **unverified** | The same main-track mismatch as ICLR | The same as ICLR, four months later |
| **COLM 2027** | Main | **unverified.** Not announced; COLM 2026 closed 31 Mar 2026 | **unverified** | A language-model-native audience receptive to evaluation critique | Nothing confirmed |

---

## Recommendation

**Do two things. They do not conflict, because NeurIPS workshops are
non-archival.**

**1. Submit the current draft to a NeurIPS 2026 evaluation workshop by 29 August
2026 — eight days.** First choice **TAE**, whose topic list is this paper's
contribution written out by someone else; second choice **IAEval**, whose 9-page
limit takes the existing draft with no restructuring. This is the highest-value
eight days on the calendar: it buys expert review of exactly the artifact claim,
from people who evaluate evaluations for a living, *before* the Tier 1 GPU work
is spent building an archival version around an unreviewed thesis.

**Correct one thing in Final Sweep 8.3 while doing it.** That section calls a
workshop "a fallback with a real publication." For these workshops it is not.
All are non-archival; accepted papers do not enter proceedings, and IAEval and
JUDGe both state it on their own pages. The upside is exactly that property:
submitting costs nothing at ICLR, ARR or NeurIPS 2027, because there is nothing
to double-submit. Treat it as reviewed exposure and a talk, not as a line on a
CV.

**2. Target the NeurIPS 2027 Evaluations & Datasets track as the archival
home.** Its scope statement is the closest thing to a written invitation this
paper will get, negative results are welcome by explicit policy, and no dataset
is required. **Its dates are unverified and unannounced**; ~May 2027 is inferred
from the 2026 cycle and must be re-checked the day the CFP posts.

The cost of that anchor is nine months, and the danger is named in the paper's
own introduction: Tran and Kiela (arXiv:2604.02460, April 2026) already reach a
neighbouring conclusion by a different route. **A second near-scoop during a
nine-month wait takes the contribution.** The mitigation is dated priority
rather than speed — an arXiv preprint posted at the workshop deadline, plus the
workshop paper itself, timestamps the mechanism and the diagnostics to August
2026, and the archival version then cites its own preprint. Do not let the 2027
anchor stand without the preprint. Without it the plan is a bet that nobody else
publishes differential final-answer truncation for nine months.

### Second choice

**ICLR 2027 — abstract 18 September, paper 25 September 2026.** It is the only
archival main-conference deadline reachable this cycle, the page limit already
matches the draft, and a decision arrives on 16 December 2026 instead of in
autumn 2027.

Take it only if the Final Sweep's Week 0–4 critical path actually lands, and
understand that what makes it second is the reviewing form rather than the
calendar. ICLR has no evaluations track. The paper arrives as a null on one
synthetic task family, one model family and one scale, into a process that
scores method novelty, and the open Tier 1 items — second task family, second
model family, the 14B grid — are precisely what a reviewer reaches for first.
Five weeks closes some of them, not all. Spending the strongest available
version of this paper on its least favourable reviewing form is a worse outcome
than waiting, and the scoop argument that would override that is already handled
by the preprint.

**Third, if ICLR is skipped and nine months is judged too long: ARR, 12 October
2026.** Its reviews come back without committing to a venue, so the scores
inform the NeurIPS 2027 submission even if nothing is ever committed to NAACL or
COLING.

---

## What this decision unblocks, and the one line of code waiting on it

`scripts/analysis/page_check.py` takes `--limit` and **has no default**. That is
deliberate, and the file says so in its own docstring: venue limits change every
cycle, and a default is a number that goes stale silently between the day it is
typed and the day it is trusted. With no `--limit` the script prints a true page
count read from Word, splits body from back matter, and enforces nothing.

Once the venue is fixed, the limit gets wired into the build and the check turns
from a report into a gate:

| If the venue is | the invocation becomes |
|---|---|
| NeurIPS 2027 ED | `--limit 9 --body-ends-at References` — **9 is inferred from the main-track instructions, not read off the ED call. Re-verify when the 2027 CFP posts** |
| ICLR 2027 | `--limit 9 --body-ends-at References` at submission, `--limit 10` at camera-ready |
| IAEval workshop | `--limit 9 --body-ends-at References` |
| JUDGe workshop | `--limit 6 --body-ends-at References` |
| TAE workshop | **unverified** — read the limit off the site before writing to it |
| ARR | **unverified** |

Fixing the venue also unblocks Final Sweep 5.3, which cannot be closed against
an unknown limit, and 8.2: every venue in the open table reviews double-blind,
and `build_paper.py --anonymous` has still never been run.

---

## Sources

All fetched 2026-08-21.

- [NeurIPS 2026 Call for Papers](https://neurips.cc/Conferences/2026/CallForPapers)
- [NeurIPS 2026 Evaluations & Datasets Track Call for Papers](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets)
- [NeurIPS 2026 Evaluations & Datasets FAQ](https://neurips.cc/Conferences/2026/EvaluationsDatasetsFAQ)
- [Introducing the Evaluations & Datasets Track at NeurIPS 2026](https://blog.neurips.cc/2026/03/23/introducing-the-evaluations-datasets-track-at-neurips-2026/)
- [Announcing the NeurIPS 2026 Workshops](https://blog.neurips.cc/2026/08/10/announcing-the-neurips-2026-workshops/)
- [TAE — Can We Trust AI Evaluation?](https://tai-eval.github.io/)
- [IAEval — Evaluation of Interactive Agents @ NeurIPS 2026](https://eval-interactive-agents-workshop.github.io/)
- [JUDGe 2026 — Can We Trust the Judge?](https://judge2026.github.io/)
- [ICLR 2027 Call for Papers](https://iclr.cc/Conferences/2027/CallForPapers)
- [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
- [ICML 2026 Call for Papers](https://icml.cc/Conferences/2026/CallForPapers)
- [ACL Rolling Review — Dates and Venues](https://aclrollingreview.org/dates)
- [EMNLP 2026 Call for Main Conference Papers](https://2026.emnlp.org/calls/main_conference_papers/)
- [COLM 2026 Key Dates](https://colmweb.org/dates.html)
- [AI Workshop Tracker — NeurIPS 2026 workshop deadlines](https://aiworkshoptracker.com/conference/neurips/)

---

## Recommendation, 2026-08-22 — after the results landed

**Updated because the evidence changed.** Three things happened after this memo
was written: the cap sweep confirmed its registered contrast at **+0.347**, the
code family produced a mixed result that is *more* interesting than a clean
replication, and a **second near-scoop** appeared (Ringelmann, arXiv:2606.02646)
alongside Tran & Kiela.

### The pick: TAE workshop on 29 August, and ICLR 2027 on 25 September

Both. They do not conflict, because the NeurIPS 2026 evaluation workshops are
non-archival.

**1. TAE (`Can We Trust AI Evaluation?`), 29 Aug 2026 — seven days.**
The paper as it stands is on-topic by construction: it is an evaluation-validity
result about a widely-used comparison. Submitting costs nothing archival and buys
reviewed exposure plus a timestamp against two concurrent papers. The results in
`RESULTS.md` are sufficient for the workshop length without the 14B or f16.

**2. ICLR 2027, abstract 18 Sep / paper 25 Sep 2026 — the archival target.**
Five weeks is enough to write the sweep and the code family properly, and — if
wanted — to restore the 14B, which resumes from disk. 9 pages of main text, which
the current draft exceeds by ~0.5 and which §5.3's pre-decided remedy fixes.

**Not NeurIPS 2027 Evaluations & Datasets, despite the fit.** It is the
best-fitting venue and roughly nine months out. With two concurrent papers
already published on the neighbouring conclusion, a nine-month wait risks a third
arriving on the *mechanism* itself. Post the arXiv preprint at the workshop
deadline to establish priority, then submit archivally at ICLR.

### Why the two near-scoops argue for going sooner, not later

Tran & Kiela normalise a global compute budget; Ringelmann model coordination
overhead. Both conclude multi-agent does not help. Neither explains **why the
published literature keeps reporting that it does** — and that question is now
open in a way it was not in July. This paper answers it: a per-turn cap,
symmetric by specification, manufactures apparent advantage, and the curve says
how much. That contribution is *more* valuable because of the scoops and *more*
perishable for the same reason.

### What must be true before submitting anywhere

| | |
|---|---|
| Page limit | body is 9.5 against a 9-page limit; §5.3's remedy is pre-decided |
| Title | `TITLE-OPTIONS.md`, author's call |
| Results written up | `RESULTS.md` is the raw material; the prose is not drafted |
| Corpus release | still unreleased, and the paper claims otherwise — §1.0.a |

**The last row is the one that is also a correctness issue**, and it grew today:
the study now rests on eleven corpora rather than six.
