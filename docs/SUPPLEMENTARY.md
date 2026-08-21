# What is the paper, and what is supplementary

This project has an unusual amount of writing in it, and two documents could
each be mistaken for the submission. This file says which is which, because a
reviewer who opens `RESEARCH-LOG.md` expecting a paper will conclude the paper
is 3,200 lines of diary, and that is nobody's fault but ours.

---

## The paper

**`CollabEngine-NeurIPS2026.docx`**, built by `scripts/analysis/build_paper.py`.

Standard academic register, standard structure: Abstract, Introduction, Related
Work, Setup, the artifact, the baseline that did not reproduce, Results,
instrument defects, Recommendations, Limitations, Future Work, Conclusion,
Broader Impact, Reproducibility Statement, References, appendices, and the
answered conference checklist.

**It is assembled from the corpus, not written by hand.** Every number in every
table is read from the episodes at build time. This is deliberate and it has a
cost worth stating: the build *fails* when the corpus is absent rather than
emitting the last known figures. A paper that still compiles when its data is
missing is a paper whose numbers can drift from its data silently.

Two build-time assertions guard it, both because the defect they catch is
invisible in a proofread:

- **`check_citations`** — every `[n]` resolves and every reference is cited.
  Two references carried invented titles for weeks before this existed.
- **the p-value lint in `table()`** — no significance column renders without an
  effect size or interval beside it. It fired on the paper's own central table.

## Supplementary material

| File | What it is | Why it is *not* the paper |
|---|---|---|
| `RESEARCH-LOG.md` | The full record — every failure, pivot, retraction and measurement, dated, in the order it happened | Research-log register throughout. It is a transparency artifact and it is genuinely valuable as one; it is not prose anyone should have to read to evaluate the claim |
| `PLAN.md` | The original design and its phases, annotated in place as each phase contradicted it | Same reason. Its value is precisely that the annotations were not tidied away |
| `PREREG-*.md` | Predictions registered before the episodes they govern, with dated amendments and postscripts | These are evidence, not argument. Five of them: `xhard`, `phase3`, `equivalence`, `14b`, plus amendments |
| `RELATED.md` | 41 verified citations with, for each framework, what its solo baseline was matched on | Working material for §2 and for the Tier-2 audit. Longer and more granular than a Related Work section can be |
| `Final Sweep.md` | What remains between this corpus and a submission, with status per item | A project-management document |
| `ENVIRONMENT.md`, `LLAMACPP-SETUP.md` | The pinned stack, and the serving arithmetic | Reproducibility appendix material |

**The log is supplementary on purpose, not by omission.** It is the strongest
evidence the project has that its corrections were made *before* the results
were known rather than after — seven of them, every one moving the headline
against the author's interest. A reviewer who wants to check that claim should
be pointed at the log; a reviewer evaluating the contribution should not have to
read it.

## The register rule, and why it is linted

The paper says "we"; the log says "I". The paper cites `[1]`; the log cites
`§4.19`. The paper has no dates in its prose; the log is organised by them.

These are not stylistic preferences — they are what separates the two documents,
and register creep in either direction destroys the split. `tests/test_paper_lints.py`
asserts that the paper's prose contains no research-log references, no first
person singular, and no calendar dates, so the boundary is maintained by the
build rather than by whoever edits it last.

The one thing that crosses freely is the *numbers*, and they cross in one
direction: the log records how a number was arrived at, the paper reports it,
and both read it from the same corpus.
