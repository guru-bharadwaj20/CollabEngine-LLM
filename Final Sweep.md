# Final Sweep

**What stands between the current corpus and a NeurIPS/ICML/ICLR/ACL-class submission.**

Written 2026-08-20 against commit `653f7db`. Every status below was checked
against the repository rather than recalled, and the evidence column says where.
Nothing here is a new research idea — the science is done and the result is
[the bounded negative one](docs/PAPER-DRAFT.md). What is left is breadth,
statistics that match the shape of the claim, and the packaging a top venue
requires.

**How to read the marks.** ✅ means the thing exists, works, and can be pointed
at. ❌ means it does not, *including* the cases where scaffolding exists but no
measurement has been made through it — a downloader that has never been run is
not a 14B result. Partial states are described in the notes, never in the mark.

---

## 0. Status at a glance

### Tier 1 — non-negotiable

| # | Item | Status | Where it stands |
|---|---|---|---|
| 1.1 | Second task family | ❌ | One synthetic scheduling generator; no second domain exists in `src/collabengine/tasks/` |
| 1.2 | Second model family | ❌ | Configs exist for Qwen3-8B (`configs/hf-local/`, `configs/vllm/`); every *reported* number is Llama-3.1-8B Q4_K_M, and the bf16 Qwen corpora were deleted (LOG §7) |
| 1.3 | Run the 14B | ❌ | `scripts/ops/fetch_14b.py` exists and Appendix C says "not yet used for a measurement" |
| 1.4 | Precision/quantization control | ❌ | No arm holds task, tier and seeds fixed while varying precision |
| 1.5 | Multiple-comparisons correction | ❌ | No Holm/BH anywhere in `src/` or `scripts/` |
| 1.6 | Equivalence testing (TOST) | ❌ | Absent. Every headline claim in the paper is a null claim made without it |
| 1.7 | A priori power analysis | ❌ | One 80%-power sizing line in `docs/PREREG-phase3.md:81`; nothing systematic, nothing pre-run for Phase 1 |
| 1.8 | Effect-size CIs in main tables | ✅ | 10,000-draw percentile bootstrap in `scripts/analysis/gate_report.py:133`; used for headline contrasts. Note: not yet carried into *every* paper table |
| 1.9 | Related work beyond two papers | ❌ | Five references total in `scripts/analysis/build_paper.py:844-852`; no multi-agent frameworks, no debate, no MoA, no measurement-validity literature |
| 1.10 | Academic prose, not log register | ✅ | `build_paper.py` emits the submission in standard register; the log survives as the transparency artifact |
| 1.11 | Standard section structure | ❌ | Sections 1–11 + References + Appendix + checklist exist; **Broader Impact and Reproducibility are checklist answers only, not body sections** |
| 1.12 | Page-limit compliance | ❌ | Nothing measures or enforces a page count; venue not yet fixed |
| 1.13 | Public repo, actually cloneable | ❌ | `api.github.com/repos/guru-bharadwaj20/CollabEngine-LLM` → **404**. The remote is private or absent |
| 1.14 | Pinned environment | ❌ | `pyproject.toml` declares floors (`numpy>=1.26`); no lockfile, no container, `torch`/`transformers` undeclared by design |
| 1.15 | README reproduces the headline end-to-end | ❌ | `gate_report.py` reproduces every gate number **from a corpus that `.gitignore` excludes**. A clone cannot run it |

### Tier 2 — materially raises the odds

| # | Item | Status | Where it stands |
|---|---|---|---|
| 2.1 | Audit a public multi-agent system for the same artifact | ❌ | Not started |
| 2.2 | Release the diagnostic as a usable tool | ❌ | `analysis/integrity.py` implements the accounting; nothing points it at a foreign transcript |
| 2.3 | Replace or demote the κ = 0.29 judge | ❌ | Four codebooks moved κ from 0.07 to 0.29 and stopped (§4.17) |
| 2.4 | Dose-response cap sweep | ❌ | The artifact is measured on/off at one cap, never as a curve |

### Tier 3 — worth doing, lower urgency

| # | Item | Status | Where it stands |
|---|---|---|---|
| 3.1 | Advisor / CCBD-CDSAML sign-off | ❌ | Not started |
| 3.2 | Anonymize per double-blind policy | ❌ | `build_paper.py --anonymous` exists and has never been exercised; repo link and affiliation are still in the docx |
| 3.3 | Decide venue and check the real deadline | ❌ | The docx is titled NeurIPS 2026 by default and no deadline has been checked |

**Six ✅ out of twenty-two — and two of the six are qualified.** That is not a
bad position: the ✅s are the expensive ones (a working instrument, an honest
corpus, prose that is already publishable in register), and most of the ❌s are
mechanical rather than uncertain.

---

## 1. Three corrections that land first, because everything else depends on them

These are cheap, they are prerequisites, and one of them is a claim in the paper
that is currently **not true**.

### 1.0.a The paper says the transcripts are released. They are not.

`docs/PAPER-DRAFT.md` §2 lists "~2,900 episodes across all conditions; **all
transcripts released**". `.gitignore` excludes `runs/`, `*.jsonl` and
`*.jsonl.gz`, and the GitHub remote 404s. Three separate reproducibility
checkboxes (1.13, 1.15, and the NeurIPS checklist's data item) rest on this
sentence.

**Do this first, in this order:**

1. Measure the corpus: `du -sh runs/` and a per-condition episode count. If it
   is under ~2 GB compressed, a GitHub release asset works; over that, Zenodo
   (which also gets you a DOI, which reviewers like).
2. Strip nothing, redact nothing — the transcripts are synthetic-task model
   output with no personal data. But **read a sample first** to confirm no local
   paths or keys leaked into the JSONL records.
3. Publish as a versioned archive with a manifest: `(condition, seed, config
   hash, model, instrument)` per episode, so a cited number can be traced to the
   exact rows that produced it.
4. Add `scripts/ops/fetch_corpus.py` that pulls the archive into `runs/`, and
   make `gate_report.py` fail with that instruction rather than an empty table.
5. Only then may the sentence stay in the paper.

**Cost:** half a day plus upload time. **Blocks:** 1.13, 1.15, and the
reproducibility statement in 1.11.

### 1.0.b A statistics module, because four Tier-1 items are the same module

Items 1.5, 1.6, 1.7 and the remainder of 1.8 are all *one* new file plus its
call sites. Build it once:

```
src/collabengine/analysis/inference.py
    holm(pvalues, labels)            -> adjusted p, in the prereg's hypothesis family
    bh_fdr(pvalues, labels)          -> reported alongside, not instead
    tost(a, b, delta)                -> two one-sided tests; returns (p_lower, p_upper, p_tost)
    smallest_equivalence_bound(a, b) -> the delta at which equivalence holds at 0.05
    mde(n, sd, alpha=0.05, power=0.8)-> minimum detectable effect for a planned arm
    n_for(delta, sd, alpha, power)   -> the inverse, for sizing runs before they run
```

Plain `scipy`/`statsmodels`, both already in the `analysis` extra. Every
function gets a test in `tests/test_inference.py` against a hand-computed case —
the repo's existing standard, and the reason its instrument survived seven
corrections.

**Cost:** one to two days including tests. **Blocks:** 1.5, 1.6, 1.7, 1.8.

### 1.0.c Pin the environment before any new corpus is generated

Every run from here forward should be attributable to an exact stack. Doing this
after the runs makes it a fiction.

- `uv lock` (or `pip-compile`) to a committed lockfile covering `analysis` and
  `dev`.
- A `Dockerfile` or `environment.yml` that also pins `torch`, `transformers`,
  the CUDA runtime, and the `llama.cpp` commit — the serving layer is where this
  project's results actually live, and it is currently pinned nowhere.
- Record the llama.cpp build hash and the GGUF file checksum in every run's
  metadata, next to the config hash.

**Cost:** one day. **Blocks:** 1.14, and the credibility of everything in §2.

---

## 2. Tier 1 — Scope

The four experimental items share one insight: **the core grid is already
written**. `configs/llamacpp/medium.yaml` + `scripts/experiments/h2-run.sh`
produce the four-arm fresh-seed comparison (`solo`, `solo_long`, 3-agent,
4-agent) that yields the headline 0.579 / 0.574 / 0.576. Each item below is that
same grid with one factor changed. Do not invent new analysis for any of them —
changing the analysis and the factor at once is §4.1c's mistake, and the log
says so.

### 2.1 Second task family ❌ — *the single biggest weakness*

**What done means.** A second, independently graded task in a different domain,
run through the identical harness, at an operating point calibrated the same
way, with the cap-asymmetry diagnostic printed beside the result. The finding
that matters: does the answer-turn truncation asymmetry appear there too?

**Choose one, and the recommendation is the middle one:**

| Candidate | For | Against |
|---|---|---|
| Multi-hop QA (HotpotQA / MuSiQue) | Public, cited, per-hop component scores map cleanly onto the agent × component design | Contamination risk at 8B; retrieval adds a confound the project does not want |
| **Program repair / code generation (HumanEval+, MBPP+)** | **Public, exactly gradable, per-test component scores are free, and verbose answers make the cap asymmetry maximally visible** | **8B pass rates are low — the failure floor PLAN §3 warns about** |
| Math (GSM8K / MATH subset) | Public, cheap, deterministic grading, short answers | Short answers *suppress* the cap artifact — the wrong test bed for this paper's mechanism |

**Recommendation: MBPP+ or HumanEval+.** The paper's contribution is that a
shared per-turn cap lands asymmetrically when arms spend it differently. Code
generation is where a single agent must emit an entire artifact in one turn —
the most adversarial possible venue for the claim, and therefore the most
convincing one. Guard the failure floor by calibrating on the easy split first;
PLAN §3's warning about SWE-bench-scale tasks does not extend to MBPP.

**Steps.**
1. `src/collabengine/tasks/code/` implementing the existing
   `generator`/`grader`/`render` interfaces — per-test-case component scores,
   not a scalar pass/fail. Components: *syntax/parse*, *base cases*, *edge
   cases*, *planted-bug detection* (mirrors the existing verification class).
2. Calibrate exactly as Phase 1 did: solo vs team across difficulty, with the
   `finish_reason` accounting printed beside it. **Print answer-turn truncation
   from the first calibration pass** — the whole point is that a calibration
   run without it measures the cap.
3. Run the four-arm fresh-seed grid, ~150 episodes per arm, seeds disjoint from
   anything used for calibration. The fresh-seed rule is not optional here.
4. Report: gate, cap-asymmetry (generation ratio and answer-turn cuts), TOST.

**Cost:** ~1 week of implementation, ~3–5 card-nights of generation.
**This is the item that converts "a bug in one harness" into "a class of
measurement error."**

### 2.2 Second model family ❌

**What done means.** The `medium` four-arm grid, same seeds, same instrument
(llama.cpp, 4-bit GGUF), run on **Qwen2.5-7B-Instruct** and **Mistral-7B-Instruct**
— two more families at comparable scale. Reported as one table with family as a
row.

The `configs/vllm/qwen3-8b.yaml` and `configs/hf-local/` files are *not* this.
They are a different backend at a different precision, and the bf16 corpora they
produced were deleted. Reusing them would confound family with instrument, which
is the exact error §4.11 refused to make when it moved the whole difficulty
curve onto one instrument together.

**Steps.**
1. Fetch both GGUFs at Q4_K_M (same quant as the Llama results).
2. `configs/llamacpp/medium-qwen.yaml`, `medium-mistral.yaml` — copies of
   `medium.yaml` with `backend.model` changed and **nothing else**.
3. `preflight.py` against each server before spending the night. Its
   weight-verification check (`verify_model: true`) is the guard that matters
   here: three families on one machine is exactly how a server ends up holding
   the wrong weights.
4. Same seeds 1000–1149. Same analysis path. Report all three families side by
   side with per-family TOST bounds.

**Cost:** ~4–8 card-nights. **Answers directly:** "is this an 8B-Llama quirk?"

### 2.3 Run the 14B ❌

**What done means.** The same grid at Qwen2.5-14B-Instruct-Q4_K_M, which
`scripts/ops/fetch_14b.py` already downloads and nothing has yet run.

This is the highest-upside item in Tier 1 and the plan says so in two places
(PLAN §6 risk register; PAPER-DRAFT §9). **If a team advantage appears at 14B
and not at 8B, the paper stops being negative-result-only and becomes a
capability-threshold finding with the artifact controls as its enabling
apparatus.** That is a materially better arc.

**Steps.**
1. Run the fetch script. Verify shard checksums; three-shard GGUFs are where
   silent truncation happens.
2. Memory arithmetic before the run, not after — `docs/LLAMACPP-SETUP.md` has
   the form. A 14B Q4_K_M is ~9 GB resident; at 7 slots the KV budget is the
   binding constraint, so expect to drop to 4–5 slots and take the throughput
   loss rather than page. **Windows pages instead of raising OOM** (§3.4) and
   the run will look healthy while it does.
3. Same four arms, same seeds, `medium` first. Add `hard` only if `medium`
   shows movement.
4. **Preregister the outcome before running it.** Write
   `docs/PREREG-14b.md` naming the threshold that would count as a capability
   effect and the equivalence bound that would count as a replication of the
   null. The project's credibility rests on this discipline and a reviewer will
   check whether it was applied to the most publication-favourable arm.

**Cost:** ~4–6 card-nights (slower per token than 8B). **Do this before 2.2** —
it is the one that can change the paper's thesis.

### 2.4 Precision / quantization control ❌

**What done means.** One tier, one family, one seed set, **precision as the only
varying factor**: Q4_K_M vs bf16 (or fp16). Then the limitation in
PAPER-DRAFT §9 — "if 4-bit weights degrade long-context instruction-following
more, the team arm absorbs more of the loss" — becomes measured rather than
argued.

**The hard part is that this control is not free of instrument.** bf16 at 8B
does not fit the in-process path at `xhard` (README: 1–1.5 MB of prefill logits
per prompt token on top of 15.3 GiB of weights). So:

- Run it at **`medium` only**, where the bf16 path demonstrably completes.
- Serve *both* precisions through llama.cpp — an F16 GGUF against a Q4_K_M GGUF
  — so the serving layer is held constant and only the weights differ. This is
  the version of the control that is actually clean, and it avoids repeating the
  bf16-`transformers` vs 4-bit-served confound entirely.
- Report the *interaction*: precision × arm. The claim under test is not "4-bit
  is worse", it is "4-bit is worse **for the team arm specifically**". A main
  effect of precision proves nothing here, for the same reason a main effect of
  ablation proved nothing in §0 of the plan.

**Cost:** ~2–3 card-nights. F16 8B is ~16 GB resident, so slots drop and the
night is longer than the token count suggests.

---

## 3. Tier 1 — Statistics

All four items are call sites of `analysis/inference.py` from §1.0.b.

### 3.1 Multiple-comparisons correction ❌

**What done means.** Every table that reports more than one *p* reports Holm-
adjusted values in an adjacent column, with the uncorrected value retained.

**The families, declared before adjusting** — this is the part reviewers check,
because a family drawn after seeing the results is not a correction:

| Family | Members | Source |
|---|---|---|
| Phase 1 gate | 3 tiers × 3 metrics | `PREREG-xhard.md` |
| Phase 3 | H1, H1e, H2, H3, H3b, H4, H5 | `PREREG-phase3.md` |
| Brief contrast | C4 vs C5 at each tier | exploratory — **label it so** |
| Phase 2 coding | differentiation, stability, propose-share | §4.8 |

Report Holm as primary (family-wise error, which is what "we ran H1–H5 at 0.05
each" invites) and BH-FDR alongside. Nothing in the paper's conclusions should
change — every headline is a null — but **that is exactly why it costs nothing
to do and looks bad to omit.**

**Cost:** one day once the module exists.

### 3.2 Equivalence testing (TOST) ❌ — *the real gap, not a nitpick*

The paper's core result is a set of null claims: team size does nothing, the
interaction is flat, fungibility is zero. Failing to reject H0 does not support
any of them. Right now the defence is a bootstrap CI that happens to be narrow —
which is the *right instinct* and the wrong instrument.

**What done means.** Every "no effect" sentence carries a TOST result with a
stated equivalence margin.

**Setting the margin, which is the only judgement call here.** Pick it from the
task, not from the data:

- The `fraction` metric's per-component granularity gives a natural smallest
  meaningful unit. Use it, and state it.
- Cross-check against a benchmark-relevance argument: a team-vs-solo difference
  below δ would not change any deployment decision.
- Preregister δ **before** running TOST on the existing corpus, in a short
  `docs/PREREG-equivalence.md`. Choosing δ after seeing the CI is the same
  failure as choosing a hypothesis after seeing the corpus, and this project has
  a rule about that.

**Then report, for each null:** the TOST *p*, and separately the *smallest δ at
which equivalence holds*. The second number is more honest and more useful — it
converts "we could not detect an effect" into "we can exclude effects larger
than X", which is the strongest form the negative result can take.

**Expected outcome, stated in advance.** The main grid at *n* ≈ 900 will support
tight equivalence bounds and the headline strengthens. The interaction test at
*n* = 48 will not, and the honest report is a wide bound — PLAN's own §1 note
already says the study "bounds the effect rather than excluding it." TOST is
what turns that sentence into a number.

**Cost:** two to three days including the prereg and the rewrite of every null
claim in the paper.

### 3.3 A priori power analysis ❌

**What done means.** Every planned arm has an MDE computed from a *prior* `sd`
estimate before the episodes run, recorded in the prereg, and compared against
the realised value afterwards.

**Steps.**
1. Retrospective-but-honest: tabulate the realised `sd` per arm from the corpus
   (`sd` ≈ 0.15 on the four-agent arm is the number that produced §4.22's
   failure). Publish that table — it is the prior every future sizing uses, and
   it is a genuine contribution to anyone else sizing a study like this.
2. For every *new* run in §2, compute `n_for(δ, sd, 0.05, 0.8)` before the run
   and write it into the prereg for that run.
3. In the paper, add one row to the setup table: planned *n*, MDE at 80% power,
   realised *n*. A reviewer who sees that row stops asking the question.
4. Retire "post-hoc power" language from the prereg postscripts. Post-hoc power
   is a deterministic function of the *p*-value and says nothing; the TOST bound
   from §3.2 is what those paragraphs were reaching for.

**Cost:** two days, most of it in step 1.

### 3.4 Effect-size CIs everywhere ✅ *(machinery)* / remaining work

`gate_report.py` already computes 10,000-draw percentile bootstrap intervals and
the headline contrasts carry them. What is missing is uniformity: several tables
in `PAPER-DRAFT.md` and in the docx report *p* alone.

**Steps.** Sweep every table in `build_paper.py`, and make the rule mechanical:
**no *p* appears in this paper without an effect size and a 95% interval beside
it.** Add an assertion to `build_paper.py` that refuses to emit a table row
matching `p = ` without a companion interval column — a lint, not a habit.

**Cost:** one day.

---

## 4. Tier 1 — Related work ❌

The submission carries five references. That is the fastest available route to a
desk reject, independent of the science.

**What done means.** A 1.5–2 page Related Work section covering five bodies of
literature, each connected to *this* paper's claim rather than summarized.

| Cluster | Minimum coverage | The connection to make |
|---|---|---|
| Multi-agent LLM frameworks | AutoGen, MetaGPT, CAMEL, ChatDev, AgentVerse | Every one ships role scaffolding in its prompts — the reason this project hand-writes orchestration (README's closing line). Also: **do their published solo baselines match generation budget?** |
| LLM debate / self-consistency | Du et al. multi-agent debate, self-consistency, reflection | The closest thing to a controlled solo-vs-multi comparison in the literature, and the most likely place for the cap artifact to be lurking |
| Mixture-of-agents | MoA and successors | Aggregation-based multi-agent gains; a direct competitor explanation for reported team advantages |
| Emergent roles / MARL lineage | ROMA, the two 2026 papers already cited | Already covered — extend rather than replace |
| **Measurement validity & reproducibility in ML** | Reproducibility-crisis work, benchmark-contamination work, leakage-in-evaluation work, "are we making progress" style reanalyses | **This is the literature the paper is actually a member of.** Currently uncited, and it is where the contribution is legible |

**Steps.** Read for a week, write for two days. Keep a `docs/RELATED.md` with a
one-line "what it claims / how it compares its baselines" note per paper — that
file is also the raw material for Tier 2.1, so the reading is not spent twice.

**Cost:** ~1 week. **Do it in parallel with the GPU runs in §2** — it needs no
card.

---

## 5. Tier 1 — Writing and structure

### 5.1 Prose register ✅

`build_paper.py` already emits standard academic register and the research log
is retained as the supplementary transparency artifact — which is the correct
split and worth keeping.

### 5.2 Section structure ❌

Sections 1–11, References, Appendix and the answered NeurIPS checklist all
exist. Two required items do not exist **as body sections**:

- **Broader Impact / Ethics.** Currently only checklist item 10. Write a short
  body section: the work is a measurement critique of multi-agent evaluation;
  the risk it addresses is over-claimed multi-agent gains driving wasted compute
  and misplaced architectural confidence; no human subjects, no personal data,
  synthetic tasks throughout.
- **Reproducibility statement.** Currently implied by the appendix. Write it
  explicitly: corpus archive DOI, lockfile, exact seeds, config hashes, the
  llama.cpp build, and the one command that regenerates the headline table.
  **This section cannot be written honestly until §1.0.a and §1.0.c land.**

Both are literal checklist items at NeurIPS and ICML. **Cost:** one day.

### 5.3 Page-limit compliance ❌

Nothing currently counts pages. After §3.3's venue decision, add a page-count
assertion to `build_paper.py` so an over-length build fails loudly rather than
at submission time. Note the structural risk: this paper's value is in its
tables and diagnostics, and the natural failure mode is a main body that
overflows while the appendix stays thin. Move the instrument-defect detail
(§7 of the draft) to the appendix first if something must give.

**Cost:** half a day, after the venue is chosen.

---

## 6. Tier 1 — Reproducibility package

### 6.1 Public repo ❌ — **verified 404**

`curl api.github.com/repos/guru-bharadwaj20/CollabEngine-LLM` returns **404**
while a control request to a known public repo returns 200. The remote is
private or does not exist. The paper must not claim a public repo until this is
resolved, and under double-blind (3.2) the *anonymized* version is what appears
in the submission — typically an anonymous mirror, with the real repo published
at camera-ready.

**Steps.** Make it public; verify by cloning it fresh into a scratch directory
and running `pip install -e ".[dev,analysis]" && pytest -q` from that clone.
"It works on my machine" and "it clones" are different claims and only the
second one is being made here.

### 6.2 Pinned environment ❌

Covered in §1.0.c. The one addition: the paper's setup table should name the
lockfile hash, so a reader can tell which environment produced which number.

### 6.3 README reproduces the headline end-to-end ❌

The command exists (`python scripts/analysis/gate_report.py`). The data it needs
is gitignored. Once §1.0.a lands:

```bash
git clone <repo> && cd CollabEngine-LLM
pip install -e ".[analysis]"
python scripts/ops/fetch_corpus.py          # pulls the archive into runs/
python scripts/analysis/gate_report.py      # reproduces the headline table
```

Add a CI job that runs exactly those four lines against a *subsampled* corpus
and diffs the output against a committed golden file. That converts "the README
reproduces the table" from a claim into a test. **Cost:** one day after 1.0.a.

---

## 7. Tier 2

### 7.1 Audit a public multi-agent system for the same artifact ❌ — *highest-leverage item on this page*

This is the item that changes the paper's ceiling. A careful negative result
about one's own harness is a solid workshop-to-conference paper. **Evidence that
the same artifact is present in someone else's published multi-agent-vs-solo
comparison is a main-conference finding**, because it makes the diagnostic
matter to people who have never heard of this codebase.

**What to look for, in order of how likely it is to be there:**
1. A per-turn or per-message token cap applied identically to both arms.
2. A solo baseline given the same *turn* count as the team rather than the same
   *generation* budget.
3. A solo baseline run through a prompt written for a group — the C5 defect.
4. No truncation or `finish_reason` accounting reported anywhere.

**Steps.** Pick two or three systems with released transcripts (AutoGen and
MetaGPT evaluation artifacts, and any debate paper with public logs). Point the
§7.2 tool at them. Report the verbosity ratio between arms and the truncation
rate per arm. **Report what you find, including "the artifact is not present
here" — a clean audit of three systems is still a contribution**, and this
project's whole credibility is that it reports the direction that goes against
it.

**Cost:** ~1 week, no GPU. Depends on §7.2 and on the reading from §4.

### 7.2 Release the diagnostic as a tool ❌

**What done means.** `collabengine audit <transcripts>` — a subcommand that
takes foreign agent transcripts in a documented minimal schema and reports:
per-arm generated characters and their ratio, per-arm truncation and answer-turn
cut counts, and a flag when the ratio exceeds a threshold.

Most of this exists inside `analysis/integrity.py`; the work is an adapter
layer, a schema, and a README section. **Adoption-oriented contributions score
well and this one is nearly free.** It is also the instrument §7.1 needs, so
build it first.

**Cost:** 2–3 days.

### 7.3 Replace or demote the κ = 0.29 judge ❌

Two acceptable resolutions and one unacceptable one.

- **Best:** a frontier judge on the full coding corpus plus a human-validated
  subsample, targeting the κ = 0.78 standard the cited prior work sets. Needs a
  paid API key — the free-tier ceiling is 20 requests/day/model (§4.4) and is
  not a workaround.
- **Acceptable and free:** the 14B from §2.3 is on disk anyway. Re-run the
  40-message hand-coded validation set through it first — that costs minutes and
  answers whether κ is a model-capacity limit or a wording one. §4.17 predicts
  capacity; measuring it settles the question either way.
- **Also acceptable:** demote Phase 2 to the appendix entirely and stop letting
  a κ = 0.29 instrument anchor the paper's credibility. The ablation-side nulls
  do not depend on it.
- **Not acceptable:** leaving κ = 0.29 in the body without one of the above.

**Cost:** minutes for the 14B check, ~2 days for the frontier path if a key
appears.

### 7.4 Dose-response cap sweep ❌ — *the best theory upgrade available*

Currently the artifact is characterized on/off at one cap. Sweep
`answer_max_tokens` across ~5 values (e.g. 512 / 1024 / 2048 / 3072 / 6144) at
one tier, both arms, and plot **apparent team advantage as a function of the
verbosity ratio between arms.**

This converts "we found a bug" into "we characterize when the bug appears and
how large it gets", which is a stronger and more citable contribution than the
before/after comparison. It also makes §7.1's audit *predictive*: given a
published system's cap and its arms' verbosity ratio, the curve estimates how
much of its reported advantage is artifact.

**Cost:** ~3–4 card-nights. **This is the item to keep if the schedule collapses**
— it is cheap, it is on the existing task and instrument, and it upgrades the
paper's central claim rather than broadening it.

---

## 8. Tier 3

### 8.1 Advisor / CCBD-CDSAML sign-off ❌
Institutional and review-trust value both. Start it now rather than at
submission, because it gates nothing but takes calendar time. Bring the
PAPER-DRAFT and this file to the conversation.

### 8.2 Anonymization ❌
`build_paper.py --anonymous` exists and has never been run. Anonymizing means
more than the byline: the repo URL, the `docs/` links, the affiliation, the
GPU-hardware description if it identifies the lab, and the acknowledgements.
Add a test that builds with `--anonymous` and greps the output for the author
name, the email, and the GitHub handle — the same lint-not-habit rule as §3.4.

### 8.3 Venue and deadline ❌ — *do this in the next 48 hours*
It determines everything else's schedule and is currently unfixed while the docx
defaults to a NeurIPS title. The decision inputs:

- **This is a measurement/negative-result paper about evaluation practice.** It
  fits the *Datasets & Benchmarks* track better than the main track, and it fits
  an evaluation-focused workshop extremely well as a fallback with a real
  publication.
- ACL/EMNLP cycles are separate from NeurIPS/ICML/ICLR and may be nearer.
- Check the actual current limits and dates — they move, and this file should
  not pretend to know them.

---

## 9. Critical path

The work splits cleanly into a **GPU lane** and a **desk lane** that do not
contend. Run both.

```
Week 0  ├─ 8.3 venue decision (48h)              ── unblocks 5.3, 8.2
        ├─ 1.0.a corpus release                  ── unblocks 6.1, 6.3, 5.2
        ├─ 1.0.c environment pin                 ── must precede all new runs
        └─ 1.0.b inference.py + tests            ── unblocks all of §3

Week 1  GPU:  2.3  14B grid (preregister first)  ── can change the thesis
        Desk: 4    related-work reading
              3.2  TOST + equivalence prereg

Week 2  GPU:  2.4  precision control
              7.4  dose-response sweep
        Desk: 4    related-work writing
              3.1/3.3/3.4  corrections, power, CIs

Week 3  GPU:  2.2  second model family (×2)
        Desk: 7.2  audit tool
              5.2  broader-impact + reproducibility sections

Week 4  GPU:  2.1  second task family
        Desk: 7.1  audit of public systems
              7.3  judge resolution (14B check is free)

Week 5  Full rewrite pass; 5.3 page limit; 8.2 anonymize; 6.3 CI reproduction
Week 6  Buffer. Something in weeks 1–4 will overrun; it is usually the task family.
```

**Two dependencies are load-bearing and easy to miss:**

1. **§2.3 (14B) before §2.2 (other families).** If 14B shows a team advantage,
   the paper's thesis changes and the other families should be run at 14B rather
   than 8B. Running 2.2 first risks spending eight card-nights at the wrong
   scale.
2. **§7.2 (tool) before §7.1 (audit).** The audit is the tool pointed at foreign
   data. Building the tool during the audit means changing the instrument and
   the subject together.

### GPU budget, and the constraint that governs it

| Item | Episodes (approx) | Card-nights |
|---|---|---|
| 2.3 — 14B grid | ~900 | 4–6 |
| 2.4 — precision control | ~600 | 2–3 |
| 7.4 — dose-response sweep | ~1,000 | 3–4 |
| 2.2 — two more families | ~1,800 | 4–8 |
| 2.1 — second task family | ~1,200 | 3–5 |
| **Total** | **~5,500** | **16–26** |

That is ~150–300 GPU-hours, which matches the independent estimate that prompted
this file.

**The card is shared, and that is a scheduling fact, not a footnote.** It is
shared with student accounts and a second project; several of those accounts are
administrators, so no lock is possible and the correct response to a busy card
is to **wait, never to kill**. `scripts/ops/resume-when-free.sh`,
`queue-judge.sh` and `overnight-watch.sh` already implement waiting — use them
rather than starting by hand. Plan on wall-clock of **1.5–2× the card-night
count above**, i.e. **5–7 weeks of nights for the GPU lane**, which is why the
desk lane must run concurrently rather than after.

---

## 10. If the schedule collapses, keep these four

In priority order, the smallest set that still materially improves the odds:

1. **§3.2 TOST** — a null-result paper without equivalence testing has a hole in
   its central claim. Two days.
2. **§4 Related work** — the cheapest desk-reject to avoid. One week, no GPU.
3. **§7.4 dose-response sweep** — upgrades the contribution rather than
   broadening it, on the existing task and instrument. Three nights.
4. **§1.0.a + §6.x reproducibility** — the paper currently makes a release claim
   it does not meet. This is the only item on the list that is also a
   correctness issue.

**§2.1 (second task) and §2.2 (second family) are the highest-value items and
the first to fall out of a compressed schedule.** If they do, the paper must
scope its title and abstract to one task family and one model family
explicitly — which PAPER-DRAFT §9 already does, and which should not be quietly
relaxed.

---

## 11. What is already strong, and should not be traded away for breadth

Worth stating, because a sweep like this reads as a list of deficiencies and the
foundation underneath it is the hard part:

- **Preregistration written before the episodes existed**, amended in dated
  public steps, with a fresh-seed rule that overturned the project's own
  headline. Very few submissions at any venue can show this.
- **An instrument that has caught seven silent corruption modes**, the largest
  of which would have published *d* = 1.09 at *p* < 0.0001.
- **Every correction moved the result in the same direction — against the
  author's interest.** Say this in the paper; it is the strongest available
  argument that the null is real rather than a failure to look hard enough.
- **353 tests, seconds, no GPU**, plus a `selftest` that validates the analysis
  against three worlds with known answers.
- **A negative result that survived three instruments, five operating points and
  two disjoint seed sets.**

Every item in this file is breadth or packaging. **None of it is redoing the
science**, and nothing in §2–§8 should be allowed to compromise the disciplines
in this section — in particular, a new task family or a new model does not get
to skip its preregistration or its fresh seeds because the deadline is close.
