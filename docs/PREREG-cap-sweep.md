# Preregistration — the dose-response cap sweep

**Registered 2026-08-21, before a single episode existed at any of the five
caps.** The configs `configs/llamacpp/cap-*.yaml` were written the same day and
no inference has been run against any of them. The corpus that supplies the
priors below is the fresh-seed `medium` rebuild on seeds 1000–1149; this sweep
runs on seeds 2000–2099, which are disjoint from it and from everything else
this project has spent.

This is the item Final Sweep §10 says to keep if the schedule collapses, and it
is the one whose result the paper's central claim depends on most directly.
That is the reason the predicted shape, the confirming threshold and the
stopping rule are fixed here rather than read off the curve.

---

## 1. Why this arm exists

The paper's mechanism claim is currently supported by two points. At
`answer_max_tokens` = 1024 the team appeared to beat a single agent by
*d* = 1.09 at `hard`, *p* < 0.0001; at 3072 the same contrast collapsed
(RESEARCH-LOG §4.9–4.11). Two points establish that a cap change moved the
result. They do not establish **when** the artifact appears, **how large** it
gets, or **what predicts it in a system we did not build** — and that last one
is what Final Sweep §7.1's audit needs in order to be predictive rather than
anecdotal.

Five caps at one tier, both arms, converts "we found a bug" into "we
characterise the bug", which is a stronger and more citable claim than the
before/after comparison, and it is measured on the existing task and the
existing instrument.

## 2. Instrument

| | |
|---|---|
| Model | Meta-Llama-3.1-8B-Instruct, Q4_K_M, `mradermacher` conversion |
| Serving | `llama.cpp`, **3 slots × 21,504 tokens** (`CTX=64512 PARALLEL=3`) |
| Configs | `configs/llamacpp/cap-512.yaml` … `cap-6144.yaml` |
| Tier | `medium` only |
| Seeds | **2000–2099**, the same 100 instances at every rung |
| Arms | `baseline` (4 agents × 3 rounds), `solo` (1 agent × 3 rounds) |
| Rungs | `answer_max_tokens` ∈ {512, 1024, 2048, 3072, 6144} |
| *n* | 100 per arm per rung — 1,000 episodes, ~3–4 card-nights |

**`max_model_len` is held at 15,360 at every rung and the slot grows with the
cap.** The `-ans` instrument spends a fixed 18,432-token slot as 15360 + 3072.
Holding the slot fixed here would mean the prompt window shrank as the cap grew,
so each rung would differ from its neighbour in two ways and the curve would
separate neither. The prompt window is therefore the constant and the server is
started once, at the largest rung's slot, for the whole sweep. Three slots
rather than four is the shared-card rule (§3.4): ~12.5 GiB, less than the
default preset's ~13.6.

**The same 100 seeds at every rung, deliberately.** The curve is read *across*
caps, so the instance draw must not be free to move with the cap. This is the
`medium-h3b` discipline — hold everything that is not the manipulated quantity —
applied to a five-point ladder instead of a two-arm contrast.

**`solo`, not `solo_long`.** The contrast being characterised is the one that
produced the artifact: team against a single agent at equal *rounds*, which is
the preregistered Phase 1 gate. `solo_long` matches the turn budget instead and
is a different question with its own registration (`PREREG-equivalence` E2).
Running both would double the sweep and answer the second question badly.

## 3. Power, computed before the run

Realised spread on `fraction` from the fresh-seed `medium` corpus
(`scripts/analysis/power_report.py --run-dir runs/llama31-8b-q4-medium-h3b`,
2026-08-21): `baseline` sd = 0.098 (*n* = 150), `solo` sd = 0.252 (*n* = 149).

| prior used | sd | MDE at *n* = 100/arm, 80% power, α = 0.05 |
|---|---|---|
| team arm alone | 0.098 | 0.039 |
| **pooled across the two arms (registered)** | **0.191** | **0.076** |
| solo arm alone, the pessimistic bound | 0.252 | 0.100 |

**The registered figure is MDE = 0.076 per cap point**, from the pooled prior,
computed with `collabengine.analysis.inference.mde`. Three things follow and all
three are registered rather than discovered:

1. **Each rung can resolve about 1.5 constraints and no less.** The equivalence
   margin is 0.05 (`PREREG-equivalence`); a single rung cannot resolve one
   constraint. It is sized for the artifact, which the two-point evidence puts
   at 0.1–0.25, not for the residual.
2. **The solo arm's spread will grow as the cap shortens**, because truncation
   adds variance to exactly one arm. So 0.076 is the optimistic end and 0.100
   the honest worst case. Rungs are read against **0.100**, and where a rung's
   gap falls between the two figures it is reported as unresolved, not as small.
3. **A null rung is a bound, not an absence.** Every rung whose interval
   contains zero is reported through `smallest_equivalence_bound`, which
   `scripts/analysis/dose_response.py` prints for all five.

*n* = 100 rather than 150 is the sizing already registered in
`power_report.py`'s `PLANNED` table and it is a budget decision: 150 per arm
would be 1,500 episodes and ~6 card-nights against Final Sweep's ~1,000-episode
allocation for this item. Raising it later **resumes** — `EPISODES=150` adds
seeds 2100–2149 and regenerates nothing — and doing so is an amendment recorded
below, not a silent extension.

## 4. Hypotheses

Priors for the mechanism, measured on the fresh-seed corpus at cap 3072
(answer-turn characters, the quantity a per-turn cap can actually cut):

| arm | mean | p50 | p90 | ≈ tokens at mean | ≈ tokens at p90 |
|---|---|---|---|---|---|
| team `baseline` final turn | 382 | 355 | 503 | 147 | 193 |
| `solo` final turn | 2,011 | 1,423 | 3,661 | 774 | 1,408 |

**The ratio of answer-turn lengths is 5.3×.** That is the number the mechanism
is about, and it is *not* RESEARCH-LOG §4.18's 1.87× — that figure is
whole-episode generation for the matched-budget arm, and for a 3-round `solo`
the whole-episode ratio runs the *other* way (0.58×) because three turns are
fewer than twelve. `dose_response.py` prints both columns for this reason.

**H-C1 (primary, confirmatory) — the artifact is a threshold, not a slope.**
Apparent team advantage is **largest at the shortest cap, decreasing and
saturating near zero as the cap passes the solo arm's answer length.** The team
answer needs ~147 tokens and its p90 is ~193, so no rung in this sweep truncates
the team; the solo answer needs ~774 and its p90 is ~1,408, so 512 and 1024
truncate it routinely, 2048 truncates its tail, and 3072/6144 should not.

- **Registered threshold for confirming the mechanism:** the largest advantage
  at cap ≤ 1024 minus the largest at cap ≥ 3072 is **≥ +0.10**, *and* the
  short-cap interval excludes zero, *and* the truncation counts are asymmetric
  (solo ≥ 20% of episodes, team ≤ 2%), *and* dropping every answer-turn-truncated
  episode removes most of the short-cap gap.
- **All four clauses are required.** Each one alone has a benign explanation:
  a drop with symmetric truncation is not this mechanism, and a gap that
  survives the truncation-dropped sensitivity was not produced by the cap.

**H-C2 (primary) — the long-cap rungs are equivalent to zero.** At 3072 and
6144 the team–solo gap is within δ = 0.05 by TOST (`PREREG-equivalence`, margin
unchanged). If it is not, the achieved bound is reported instead and the paper
says so.

**H-C3 (secondary) — the curve is monotone in the cap.** Advantage is
non-increasing from 512 to 6144. A non-monotone curve with an interior maximum
would falsify the threshold account and is the outcome that would most change
the paper.

**H-C4 (secondary) — the ratio is what transfers.** The rung at which the
advantage falls below 0.05 is the rung whose cap first exceeds the solo arm's
answer-length p90. This is the clause an outside auditor would actually use:
it predicts the artifact's size from two quantities a published system reports
about itself.

## 5. What would falsify what

| outcome | reading |
|---|---|
| H-C1 fires, all four clauses | **The artifact is characterised, not merely found.** The paper's central claim upgrades from a correction to a dose-response result, and §7.1's audit becomes predictive |
| Advantage flat and near zero at every rung | The 1024-vs-3072 difference was not the cap alone. The paper's §4.9–4.11 account is incomplete and must be re-opened — this is the outcome that costs the most and is registered because of it |
| Advantage large at every rung, including 6144 | A real team advantage at `medium` that the earlier corpora missed. The negative result weakens and the abstract changes |
| Curve non-monotone with an interior maximum | H-C3 falsified; the threshold account is wrong and the shape is reported as unexplained rather than smoothed |
| Short-cap gap survives the truncation-dropped row | Whatever the shape, the cause is not truncation. Reported as such, and H-C1 does not fire even if the shape matches |

## 6. Rules fixed in advance

1. **The fresh-seed rule applies.** Seeds 2000–2099, disjoint from the pilot
   (0–47) and the confirmatory corpus (1000–1149). Nothing in this corpus may
   test a hypothesis this corpus generated.
2. **All five rungs, or the shape is not reported.** A curve through three
   points chosen after seeing three others is not a curve. If the sweep dies
   partway, the completed rungs are reported as individual contrasts and the
   dose-response claim waits.
3. **No rung is dropped for being uninteresting**, and no sixth cap is added
   after seeing the five. A cap added to fill in a suggestive gap is an
   amendment, dated, with its reason, below this line.
4. **The integrity block is printed above every rung.** `cap-sweep.sh` runs
   `gate_report.py` per rung so a mean cannot be read without its truncation
   counts — those counts are the mechanism, not a footnote.
5. **Multiplicity.** H-C1 and H-C2 join the Phase 1 gate family for Holm
   adjustment. The five per-rung contrasts are *one* declared family; reading
   five rungs at α = 0.05 each and reporting the largest is what §3.1 exists to
   forbid.
6. **The curve is fit by nothing.** Five points are plotted and described. No
   sigmoid is fitted to five points and no threshold is estimated by
   interpolation; the reported threshold is "between rung X and rung Y".

## 7. The prediction, on the record

**I expect H-C1 to fire, and I expect the residual at 6144 to be small but not
zero.** The measured answer lengths make the threshold account almost
arithmetic: the team's answer turn has never needed more than ~200 tokens and
the solo arm's p90 needs ~1,400, so a 512-token cap can only bite one arm. What
I am less sure of is the 3072 rung, where the fresh-seed corpus still shows
7/149 solo answer turns truncating and a team–solo gap of +0.081 that the
present sweep will re-measure on fresh instances. If that gap holds at 6144 with
zero truncation in either arm, then part of what this project has been calling
artifact is a real deficit of the 3-round solo arm, and **the paper has to say
so** — the honest version of this result is a curve that decays to a non-zero
floor, not one that decays to nothing.

Recorded because it is the inconvenient half of the prediction, and because the
floor is the number a reader should trust least if it arrives unregistered.

---

*Amendments below this line, dated, with the reason. Nothing above it is edited
after registration.*
