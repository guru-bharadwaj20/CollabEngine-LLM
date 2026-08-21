# Audit targets

Reconnaissance for Final Sweep §7.1: which published multi-agent systems release
artifacts that `collabengine audit` can actually read. This is a survey. No
corpus was downloaded, and the tool was not run against anything.

The question is narrower than "did they release code". `docs/AUDIT-SCHEMA.md`
needs, per episode, an **arm label** and a list of **turns each carrying raw
generated text**. Everything else — `finish_reason`, `tokens`,
`answer_parsed` — is optional and buys exactly one column. So a system can be
fully open-source and still be unauditable, because releasing a *framework* is
not releasing *generations*, and almost none of these projects release the
single-agent arm's generations at all. That turns out to be the dominant
finding: the artifact is hard to look for not because the systems hide it but
because the baseline arm's text is usually never written to disk.

Every URL below was fetched on 2026-08-21. Claims that could not be confirmed by
fetching are marked **not confirmed** with the check that was made. Nothing here
is recalled.

---

## What the four artifacts need in order to be visible

Stating this first, because it determines the verdicts.

| Final Sweep §7.1 artifact | Minimum evidence in a released corpus |
|---|---|
| 1. Per-turn cap applied identically to both arms | The configured cap for each arm, from code or config — plus `finish_reason` to see where it binds |
| 2. Solo baseline matched on turns, not generation | Both arms' turn counts and generated text; the character ratio alone is enough to show it |
| 3. Solo baseline run through a group prompt | The prompt template used for each arm |
| 4. No truncation or `finish_reason` accounting anywhere | The absence of the field, which is itself readable from the artifact |

Artifacts 1 and 3 are answerable from **source code and config** without any
transcript. Artifacts 2 and 4 need the corpus. This matters for ranking: a
target with no released transcripts can still settle 1 and 3, and several below
do.

---

## Per-system findings

### Mixture-of-Agents, Self-MoA, and the AlpacaEval results directory

The strongest target found, and it was not on the §6 list in `docs/RELATED.md`.

MoA's own repository ([`github.com/togethercomputer/MoA`](https://github.com/togethercomputer/MoA))
releases code plus an `outputs/` directory containing exactly two files —
`Qwen-72B-round-1.json` (1,787,875 bytes) and `Qwen-72B-round-2.json`
(1,835,867 bytes). Intermediate layer outputs for one reference model; not an
arm comparison.

The comparison arms are in the AlpacaEval leaderboard repository instead.
[`github.com/tatsu-lab/alpaca_eval`](https://github.com/tatsu-lab/alpaca_eval)
carries a `results/` directory of 228 model entries, each holding a
`model_outputs.json`. Confirmed present, by name:

- Aggregation arms: `Together-MoA`, `Together-MoA-Lite`, `SelfMoA_gemma-2-9b-it-SimPO`,
  `SelfMoA_gemma-2-9b-it-WPO-HB`, `Shopee-SlimMoA-v1`, `openpipe-moa-gpt-4-turbo-v1`
- Matching single-model arms: `gemma-2-9b-it-SimPO`, `gemma-2-9b-it-WPO-HB`,
  `gemma-2-9b-it-DPO`, and the rest of the 228

`Together-MoA/model_outputs.json` is 1,722,072 bytes;
`SelfMoA_gemma-2-9b-it-SimPO/model_outputs.json` is 1,814,278 bytes. Both parse
to 805 records — the AlpacaEval 2.0 instruction set — with fields
`instruction`, `dataset`, `output`, `generator` and nothing else. The `output`
field is the raw final generation, not a parsed or trimmed one.

Every arm answers the **same 805 prompts**, so this is a matched task set across
a multi-generation arm and a single-generation arm, which no other candidate
here supplies.

Two things were read out of source rather than guessed:

- MoA's caller keeps only `res.json()["choices"][0]["message"]["content"]`
  (`utils.py`). The response's `finish_reason` and `usage` are discarded at the
  call site, so no downstream release could contain them. **Artifact 4 is
  confirmed present in MoA by source inspection**, independent of any corpus.
- The per-call cap is `max_tokens=2048` throughout `utils.py`, applied to every
  reference-model call and to the aggregator call. The MoA arm therefore spends
  *n_reference × n_layers + 1* generations of 2048 against a baseline's one.

The caps across arms are **not** identical, and the direction is worth stating
precisely rather than assimilating to artifact 1. From
`src/alpaca_eval/models_configs/`:

| entry | cap | temperature |
|---|---|---|
| `Together-MoA` | `max_tokens: 2048` | 0.7 |
| `SelfMoA_gemma-2-9b-it-SimPO` | `max_new_tokens: 2048` | 0.7 |
| `gemma-2-9b-it-SimPO` | `max_new_tokens: 4096` | 0.5 |

The single-model baseline gets **twice the per-call cap** of the Self-MoA arm
built on the same model. That is the opposite of artifact 1 as specified, and it
is the honest thing to report: the cap here is asymmetric in specification, in
the baseline's favour, while the *number* of capped calls is asymmetric in the
aggregation arm's favour. Whether the baseline's answers pile up near either
limit is exactly what `chars/ep` would show, and it is measurable from the
released text.

**Schema recovery:** `arm` ← `generator` (present on every record). `turns` ←
one turn per episode, `text` ← `output`. `episode_id` ← `instruction` or index.
No `role` — every turn is a generation, which is correct here. No
`finish_reason`, no `tokens`, no `answer_parsed`.

**Computable:** episode count, turn count, `chars/ep` per arm, the verbosity
ratio between any two arms, and the flag. **Not computable:** `tok/ep`, `trunc`,
`cut@end`, `malformed` — all report `n/a` with caveats, and the §4.24 check
cannot be run.

**Verdict: partially auditable — and the best partial available.** The one
column it does produce is the tool's primary signal, on 805 matched prompts,
across arms whose generation counts differ by a documented factor.

**Caveat that must be checked before any number is reported:** an AlpacaEval
`model_outputs.json` is submitted by whoever ran the model, so provenance varies
per entry and the entries were not produced by one harness. Treat a ratio here
as a comparison between two submissions, not between two arms of one experiment.

### Du et al., multi-agent debate

Code at
[`github.com/composable-models/llm_multiagent_debate`](https://github.com/composable-models/llm_multiagent_debate)
— directories `math/`, `gsm/`, `biography/`, `mmlu/`, each with generation and
evaluation scripts. The README links additional debate logs to a Dropbox folder:
`https://www.dropbox.com/sh/6kq5ixfnf4zqk09/AABezsYsBhgg1IQAZ12yQ43_a?dl=0`.

That link is **live as of 2026-08-21**: requesting it with `?dl=1` returns
HTTP 200, `Content-Disposition: attachment; filename="debate_data.zip"`,
`Content-Length: 3945954` — 3.9 MB.

`gsm/gen_gsm.py` was read directly. It stores only
`completion["choices"][0]["message"]["content"]` into each agent's context and
writes the accumulated contexts to `gsm_{agents}_{rounds}.json`. So:

- Full raw generated text per round is preserved, per agent. Good.
- `finish_reason` and `usage` are discarded at the call site. **Artifact 4 is
  confirmed present here too, by source inspection.**
- The arm label is recoverable from the *filename*, since `agents` and `rounds`
  are interpolated into it. A converter reads the arm off the path.
- `agents = 1` is a reachable configuration, so a single-agent arm can exist in
  principle.

**Not confirmed:** whether `debate_data.zip` actually contains a
one-agent file. The archive index could not be enumerated — a fetch of the zip
was attempted and declined by this environment's tooling, and Dropbox renders
its folder listing client-side so the web view returns nothing enumerable. What
was checked: the folder's HTTP headers, the README, and `gen_gsm.py`. If the
zip holds only `*_3_2.json`-style multi-agent runs, this target has one arm and
the verbosity ratio is not computable — the tool would say so rather than invent
a baseline.

**Schema recovery, if a one-agent file is present:** `arm` ← filename,
`turns` ← the assistant entries of each agent context (the `role` field in the
stored context distinguishes assistant from user, so prompt scaffolding is
excludable, which matters because the debate arm's scaffold is much larger).
`text` ← `content`. No `finish_reason`, `tokens`, or parse status.

**Verdict: partially auditable, conditional on one unverified fact.** Resolving
it costs a 3.9 MB download and a `zipfile.namelist()`. Rank it second only
because that fact is unresolved, not because the target is weaker — the fixed
per-round prompts make this the cleanest place to test for the artifact's
*absence*, which §7.1 counts as a result.

### CAMEL

Framework at [`github.com/camel-ai/camel`](https://github.com/camel-ai/camel);
released conversation corpus at
[`huggingface.co/datasets/camel-ai/ai_society`](https://huggingface.co/datasets/camel-ai/ai_society).

The dataset card's schema was read directly and is unusually good for this
purpose: `role_1`, `role_2`, `id`, `original_task`, `specified_task`,
`message_1` … `message_12` (each a struct of `role_type`, `role_name`, `role`,
`content`), `num_messages`, and — the field that matters —
**`termination_reason`**, a string.

`termination_reason` is a per-episode stop reason, the closest thing to
`finish_reason` in any candidate here. Two qualifications, both real:

- It is per **episode**, not per turn, so it does not satisfy the schema's
  per-turn `finish_reason` and would not populate `trunc`. It could populate
  `cut@end` only under an assumption the tool deliberately refuses to make.
- Its value vocabulary is not in the tool's alias list. It would need an
  explicit mapping in a converter, per the schema doc's instruction to map
  rather than hope.

The blocking problem is elsewhere. This corpus is the **role-playing arm only**.
The paper's comparison is a 40-message CAMEL conversation, summarised, judged
against one `gpt-3.5-turbo` single-shot generation — and no single-shot
generations are released. `docs/RELATED.md` §6 rates CAMEL "highest prior of
finding the artifact" because of that 40:1 design, and that judgement stands
about the *design*; it does not survive contact with what is *released*. One arm
means no verbosity ratio.

Also noted: the HF dataset viewer currently errors on this repo — *"All the data
files must have the same columns, but at some point there are 2 new columns
({'message_6', 'message_5'})"* — because episode length varies. That affects the
viewer, not a local read, but a converter must handle ragged `message_N` fields.

**Not confirmed:** exact file names, download size, and row count. The card
reports neither and the viewer error suppressed the row statistics. The 25,000-
conversation figure circulating for this dataset was **not** confirmed on the
card and is not asserted here.

**Verdict: not auditable as released.** Single arm. Would become auditable only
by generating the single-shot baseline oneself, at which point it is a re-run,
not an audit.

### ChatDev

[`github.com/OpenBMB/ChatDev`](https://github.com/OpenBMB/ChatDev). The `main`
branch is ChatDev 2.0 and **has no `WareHouse/` directory** — the API call
against `main` returns 404. The generated-project corpus exists on the v1 tags;
at `v1.1.6` the directory holds 89 run directories (`Gomoku_THUNLP_20230625201030`,
`FlappyBird_THUNLP_20230726121145`, and so on).

One run was inspected end to end.
`WareHouse/Gomoku_THUNLP_20230625201030/` at `v1.1.6` contains a 53,041-byte
`20230625201030.log`, source files, and a `meta.txt` naming the roster
(Chief Executive Officer, Programmer, Code Reviewer, …). The log carries
timestamped entries with the speaking agent and full verbatim assistant text.

Searching that whole log for `token|finish|max_tokens|usage` returns exactly one
line, `<INFO> Finished.` — a phase marker, not a stop reason. **No token
accounting and no finish reason anywhere in the run log**, in a system whose
paper reports per-run token totals (7,182 for GPT-Engineer against 22,949 for
ChatDev). The totals were computed somewhere and not written into the released
artifact. That is artifact 4, confirmed by direct inspection of a released file.

The second problem is again the arm. `WareHouse/` holds ChatDev runs only. No
GPT-Engineer baseline output is released, so the 3.2× generation gap the paper
states in prose cannot be re-derived from the artifact, and the tool has one arm.

**Schema recovery:** `arm` would be a constant. `turns` require parsing an
ANSI-coloured plaintext log into speaker/text pairs — feasible, since the
speaker prefix is regular, but it is a real converter rather than a field
rename. `text` recoverable. `agent` recoverable from the role prefix. Nothing
else.

**Verdict: not auditable.** Single arm, no stop-reason field. Its value is as a
*cited example* of artifact 4 in a major system, which the inspection above
establishes without running the tool.

### MAST-Data (Cemri et al.)

[`huggingface.co/datasets/mcemri/MAD`](https://huggingface.co/datasets/mcemri/MAD),
code at
[`github.com/multi-agent-systems-failure-taxonomy/MAST`](https://github.com/multi-agent-systems-failure-taxonomy/MAST).
`docs/RELATED.md` §6 calls this the best candidate. Reading the card changes
that.

Confirmed schema: `mas_name`, `llm_name`, `benchmark_name`, `trace_id`,
`trace` (a dict of `key`, `index`, `trajectory`), `mast_annotation` (14 codes,
each `1`/`0`/`null`). Files `MAD_full_dataset.json` (1,642 traces) and
`MAD_human_labelled_dataset.json` (19).

Three findings, in descending order of how much they hurt:

1. **There is no solo arm and no arm field.** `mas_name` distinguishes AG2 from
   MetaGPT from ChatDev — every value is a multi-agent framework. The corpus is
   a set of *failure traces of multi-agent systems*, not a set of paired
   comparisons. It could be loaded with `arm ← mas_name`, which would produce a
   cross-framework verbosity comparison; that is a different and much weaker
   question than the one §7.1 asks, and reporting it as if it answered §7.1
   would be a misuse of the tool.
2. **The card does not mention `finish_reason` or token counts** anywhere.
3. The trajectory samples visible on the card are formatted log text with
   embedded timestamps (`[2025-31-03 19:09:41 INFO]`), the same shape as the
   ChatDev log above — i.e. probably prose logs rather than structured turns, so
   `text` extraction needs a per-framework parser. **Not confirmed:** whether
   every framework's trajectory preserves raw message text; only truncated
   samples were visible, and the full file was not downloaded.

The §6 open questions are therefore answered in the unhelpful direction, and one
that §6 did not ask — is there a second arm — is the fatal one.

**Verdict: not auditable for §7.1.** Still valuable to the paper for the reason
`docs/RELATED.md` already gives: MAST's 14-mode taxonomy contains no category
for *the harness truncated the answer*, and 1,642 traces annotated without such
a category is evidence about what the field looks for.

The human-labelled subset discrepancy flagged under **Dropped** in
`docs/RELATED.md` (19 traces on the card against 150 in the paper) was not
re-investigated here and remains open.

### AgentVerse

[`github.com/OpenBMB/AgentVerse`](https://github.com/OpenBMB/AgentVerse).
Top-level listing confirmed: `agentverse/`, `agentverse_command/`, `data/`,
`dataloader/`, `documentation/`, `scripts/`, `ui/`, plus the usual files. The
`data/` directory contains `commongen`, `humaneval`, `logic_grid`, `mgsm`,
`responsegen` — **benchmark inputs, not run outputs**. No results, logs, or
outputs directory exists at the top level.

**Verdict: not auditable from public artifacts.** No generations released.
`docs/RELATED.md` §6 already reaches this conclusion and calls AgentVerse the
best *re-run* candidate rather than the best log candidate; nothing found here
contradicts that. Its `Solo` condition remains the closest published analogue of
the C5 arm, and artifact 3 — a solo baseline running a group prompt — is
answerable for AgentVerse from its prompt configs alone, without any transcript.
That check was not performed here and is the cheapest unclaimed win on this
page.

### AutoGen / AutoGenBench / Magentic-One

AutoGenBench is documented as recording "all logs and trace information" per
run, with repetition and isolation controls. That is instrumentation the user
runs locally. **No released corpus of AutoGenBench run logs was found.** What
was checked: a web search for released AutoGenBench run logs and evaluation
traces, which returned the AG2 and AutoGen 0.2 blog posts describing the tool
and no artifact link.

**Verdict: not auditable without re-running.** Consistent with the §6 entry.

### MetaGPT

**No released HumanEval or SoftwareDev run logs were found.** What was checked:
a web search for MetaGPT released logs and artifacts, which surfaced three
separate GitHub issues on the MetaGPT repository asking where the SoftwareDev
dataset is (`#482`, `#1660`) and one on HumanEval execution (`#1062`) — i.e.
users requesting access, not documentation of a download. The dataset's public
availability is **not confirmed**; the issues indicate it is at least not
straightforwardly available.

**Verdict: not auditable from public artifacts.**

### Leads found and not pursued

Named so the search is reproducible, all **unverified beyond the noted check**:

- **"Debate or Vote", NeurIPS 2025 Spotlight**,
  [`github.com/deeplearning-wisc/debate-or-vote`](https://github.com/deeplearning-wisc/debate-or-vote).
  README fetched: it instructs the user to `mkdir out` and describes sparse,
  centralised and heterogeneous MAD variants. No released result files, no
  single-agent arm, and no `finish_reason` or token accounting are mentioned in
  the README. A code-level check of whether it logs a single-agent arm was not
  done.
- **"12 Angry AI Agents"** (arXiv:2605.01986), project page
  [`ahmetbersoz.github.io/12-angry-ai-agents/`](https://ahmetbersoz.github.io/12-angry-ai-agents/),
  code at `github.com/ahmetbersoz/12-angry-ai-agents`. The page advertises
  raw transcripts behind an `experiments.html` link. Transcript format, download
  size, presence of a single-agent arm, and any token or cap accounting are all
  **not confirmed** — only the landing page was fetched.
- **Tran & Kiela** (arXiv:2604.02460), the near-scoop. `docs/RELATED.md`
  records that no public code, data or transcript release could be verified as
  of 2026-08-20. Not re-checked here. It is the one paper whose corpus would
  answer §7.1 directly, and an email is still the cheapest route.

---

## Ranked shortlist

**1. AlpacaEval `results/` — Together-MoA and Self-MoA against their single-model baselines.**
Access: individual files over HTTPS,
`https://raw.githubusercontent.com/tatsu-lab/alpaca_eval/main/results/<entry>/model_outputs.json`.
Roughly 1.7–1.8 MB per arm; four arms is ~7 MB. No clone needed and no auth. A
directory of four such files is a container the tool already accepts. Produces
`chars/ep` and the verbosity ratio on 805 matched prompts. Everything else
reports `n/a`, correctly.

**2. Du et al. `debate_data.zip`.**
Access: `curl -L "https://www.dropbox.com/sh/6kq5ixfnf4zqk09/AABezsYsBhgg1IQAZ12yQ43_a?dl=1"`,
3,945,954 bytes, confirmed live. First action is `zipfile.namelist()` to find out
whether a one-agent file exists. If it does, this is the better target of the
two, because both arms come from one harness and one prompt family. If it does
not, the target is dead and the check cost four megabytes.

**3. CAMEL `ai_society`, as a `termination_reason` case study rather than an audit.**
Access: the Hugging Face dataset repo, `huggingface.co/datasets/camel-ai/ai_society`.
Size not confirmed. It cannot answer §7.1 — one arm — but it is the only
released multi-agent corpus found that carries a stop reason at all, which makes
it the reference point for the claim that the field does not record this.

ChatDev v1.1.6 `WareHouse/` is deliberately not on the shortlist despite being
the most thoroughly verified item on this page. It has one arm. Its use is as an
inspected example of artifact 4, and that inspection is already complete above.

---

## Likely outcome, stated in advance

Two predictions, so the write-up cannot be tuned after the fact.

**The truncation columns will be empty on every target.** Four systems were
inspected at source level — MoA, Du et al., ChatDev, CAMEL — and three of the
four discard `finish_reason` at the API call site or never write it to the
released artifact. That is artifact 4, present in the strongest form §7.1
describes: not "unreported in the paper" but *structurally unrecoverable from
the release*. This is a finding on its own, and it is the one the survey is most
confident of. It is also the finding that most limits the rest: `trunc` and
`cut@end`, the two columns that would show *where* a cap binds, cannot be
produced for any public target found.

**What is left is the character ratio, and it may well come back under 1.5.**
On the AlpacaEval targets the caps run the wrong way for artifact 1 — the
single-model baseline gets 4096 where the Self-MoA arm gets 2048 — so the
mechanism this project measured is not the mechanism present there. A ratio
under threshold is the honest expectation.

Per §7.1 that does not disqualify anything. "The artifact is not present here"
is a publishable result, and the schema doc already fixes the wording: a ratio
under the threshold is an absence of evidence at this tool's sensitivity, not a
clean bill of health. The contribution of a clean audit is that the diagnostic
ran on someone else's corpus, unchanged, and said so — which is the claim §7.2
needs anyway.

The risk to name plainly is the opposite one: with `trunc` and `cut@end`
unavailable everywhere, an audit that reports only a character ratio is a thin
result compared to what §7.1 imagines. The strongest available version of the
item is therefore two findings, not one — the ratio where it is computable, and
a source-level census of *which systems throw the stop reason away and where in
the code they do it*. The second half is already half-written above, needs no
downloads, and is not weakened by any target coming back clean.

---

## Not verified — read this before citing anything above

- Contents of `debate_data.zip`. Headers and size confirmed; the archive index
  was not enumerated. Whether a one-agent arm is inside is unknown.
- CAMEL `ai_society` file names, download size, and row count. The card does not
  state them and the HF viewer errored. The widely-quoted 25,000-conversation
  figure is not asserted here.
- Whether MAST-Data trajectories preserve raw message text for every framework.
  Only truncated card samples were seen.
- MAST-Data's 19-vs-150 human-labelled discrepancy. Carried over from
  `docs/RELATED.md`, not re-checked.
- Whether `WareHouse/` exists on ChatDev tags other than `v1.1.6`, and whether
  the 89 entries there are consistent across v1 tags. One tag and one run
  directory were inspected.
- ChatDev: only one log file was searched for token accounting. The claim "no
  token accounting in the released logs" rests on that one file plus its
  `meta.txt`, and should be checked against two or three more before it appears
  in the paper.
- Whether AutoGenBench or MetaGPT logs exist somewhere not surfaced by the
  searches described. Absence of a search hit is weaker evidence than a 404.
- Everything in **Leads found and not pursued**, at the level of detail stated
  there.
- Provenance of individual AlpacaEval `model_outputs.json` submissions. Field
  structure and record counts were confirmed by parsing two of them; who
  generated each, and under what harness, was not.
