# The audit transcript schema

`collabengine audit <transcripts>` runs this project's truncation and verbosity
accounting on transcripts from any multi-agent system. This file is the contract
it reads against: the smallest set of fields a foreign system must supply, what
each one means, and — the part that matters more — what the tool stops being
able to compute when a field is absent.

The schema is deliberately small. Every field beyond the three required ones
buys exactly one named quantity, and the report says which quantities it could
not produce and why, so a partial export is useful rather than rejected.

## Shape

One JSON object per **episode** — a single run of the system on a single task
instance under a single arm. Three accepted containers:

| container | shape |
|---|---|
| `corpus.jsonl` | one episode object per line |
| `corpus.json` | a list of episode objects, or `{"episodes": [...]}` (`records` and `data` also accepted) |
| a directory | every `.json` and `.jsonl` file inside it, read in name order |

## Episode fields

| field | required | meaning |
|---|---|---|
| `arm` | **yes** | The comparison cell this episode belongs to: `"solo"`, `"team"`, `"debate"`, whatever the paper calls it. Aliases: `condition`, `system`, `group`. |
| `turns` | **yes** | List of turn objects, in the order they occurred. Alias: `messages`. |
| `episode_id` | no | Identifier, used only in messages about records that failed to parse. Aliases: `id`, `task_id`. Defaults to the file and line the record came from. |
| `answer_parsed` | no | `true` if the harness successfully extracted an answer from this episode, `false` if it scored the episode zero because it could not. |
| `malformed` | no | The same fact stated the other way round; wins over `answer_parsed` if both are present. |

`arm` is required because every quantity the tool reports is a comparison
*between* arms. A corpus with one arm is readable, but the report will say the
verbosity ratio is not computable rather than inventing a baseline.

## Turn fields

A turn is one generation by the system under audit.

| field | required | meaning |
|---|---|---|
| `text` | **yes** | The generated text, exactly as the model produced it. Aliases: `content`, `message`. |
| `role` | no | `assistant`, `agent`, `model`, `ai` — model output, counted. Anything else (`user`, `system`, `tool`) is prompt scaffolding and is excluded from every count. Alias: `speaker`. **A turn with no `role` is counted as generated.** |
| `finish_reason` | no | Why generation stopped. Alias: `stop_reason`. |
| `tokens` | no | Completion tokens for this turn. Aliases: `completion_tokens`, `output_tokens`. |
| `agent` | no | Which agent produced the turn. Aliases: `author`, `name`. Not used in any reported quantity; carried so that a flagged corpus can be re-examined per agent. |

`text` must be the raw generation. Post-processed text — a parsed answer block,
a summary, a trimmed field — makes the character counts measure the
post-processing rather than the model, and the character counts are the tool's
primary signal.

### `finish_reason` values

Normalised, case-insensitively:

- **truncated:** `length`, `max_tokens`, `max_output_tokens`,
  `max_tokens_reached`, `truncated`
- **errored:** `error`, `exception`, `failed`, `content_filter`
- anything else passes through unchanged and counts as neither

An unrecognised spelling therefore under-reports truncation visibly instead of
being guessed at. If your system spells it something else, map it in your
converter rather than hoping.

## What is lost when a field is absent

This is the operative half of the document. The tool never infers a missing
field, and never reports an absent field as a zero.

| absent | what stops being computable | what the report says |
|---|---|---|
| `finish_reason` on **any** generated turn in an arm | truncated-turn count, answer-turn cut count, and errored-turn count **for that whole arm** | `n/a` in those columns, plus a caveat line naming the arm |
| `tokens` on **any** generated turn in an arm | tokens per episode for that arm | `n/a` in `tok/ep`; the character ratio is unaffected |
| `answer_parsed` / `malformed` on **any** episode in an arm | unparseable-answer count for that arm, and the whole §4.24 check | `n/a` in `malformed`, plus a caveat saying the fourth artifact cannot be checked for here |
| fewer than two arms with episodes | the verbosity ratio, and therefore the flag | a line saying the tool compares arms and says nothing about a single one |

The rule in the first three rows is **all-or-nothing per arm**: one turn missing
`finish_reason` disqualifies the arm, not just that turn. A corpus that carries
the field on some rows and not others is not missing rows at random — the usual
reason a row lacks it is that a wrapper layer dropped it, and wrapper layers
correlate with arm. A tally over the instrumented subset would report the
least-instrumented arm as the cleanest, which is the exact failure mode the
tool exists to catch.

Absence of `finish_reason` is itself a finding. Final Sweep §7.1 lists "no
truncation or `finish_reason` accounting reported anywhere" as one of the four
things to look for in a published system, and a corpus that cannot answer the
question is evidence about the system that released it.

## Minimal example

Two arms, everything supplied:

```json
{"episode_id": "t0", "arm": "solo", "answer_parsed": false,
 "turns": [{"role": "assistant", "text": "...", "finish_reason": "length", "tokens": 2048}]}
{"episode_id": "t0", "arm": "team", "answer_parsed": true,
 "turns": [{"role": "assistant", "agent": "A1", "text": "...", "finish_reason": "stop", "tokens": 402},
           {"role": "assistant", "agent": "A2", "text": "...", "finish_reason": "stop", "tokens": 388}]}
```

The absolute minimum the tool will accept, which gets the verbosity ratio and
nothing else:

```json
{"arm": "solo", "turns": [{"text": "..."}]}
{"arm": "team", "turns": [{"text": "..."}, {"text": "..."}]}
```

## Reading the output

```
arm                   eps  turns   chars/ep   tok/ep   trunc  cut@end  malformed
```

- **chars/ep** — generated characters per episode. Per *episode*, not per turn,
  because the asymmetry survives a matched turn budget: this project's own
  matched-budget arm generated 25,901 characters per episode against the team's
  13,873 on identical turn counts and an identical per-turn cap (RESEARCH-LOG
  4.23). A per-turn ratio divides that finding away.
- **trunc** — generated turns that stopped at the cap.
- **cut@end** — episodes whose *final* turn stopped at the cap. This is the
  truncation that costs a score, and it does not fall equally on arms: a short
  solo episode carries its whole answer in the last turn, while a long team
  episode commits an answer the transcript already contains.
- **malformed** — episodes whose answer did not parse. Each scores zero and is
  counted, so an asymmetry here moves an arm's mean directly.

The flag fires when the verbosity ratio exceeds the threshold, `1.5` by default.
That number sits below the 1.87 at which the artifact was measured to distort a
headline here; it is not a standard, and `--threshold` changes it. A ratio under
it is an absence of evidence at this tool's sensitivity, not a clean bill of
health, and the report words it that way.

`audit` exits 0 whether or not it flags. Only a corpus it could not read exits
2.
