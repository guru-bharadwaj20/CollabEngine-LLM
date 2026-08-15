"""Agreement between a human reading and the local behavioural judge.

RESEARCH-LOG 4.8 defends Phase 2's differentiation null with kappa = 0.68 against
a second model, and flags that both raters were Google models so their agreement
is an upper bound. This is the check that flag was asking for: a human instead of
a second model. It returns **kappa = 0.07** (RESEARCH-LOG 4.13).

The human labels in `docs/data/handcode-sample.json` were assigned by reading the 40
messages before any judge label for this corpus existed. This script supplies the
other half and the agreement between them.

Coding goes through the pipeline's own `_code_one` on purpose -- same prompt,
same identity stripping, same per-turn seed. An earlier attempt used a fixed seed
and hand-rolled parsing, and one reply came back as raw JSON that a naive split()
turned into a label; `parse_action` already falls back to OTHER for that. A kappa
measured against a judge nobody runs is not the judge's kappa.

    python scripts/analysis/handcode_kappa.py            # re-code with the judge, then report
    python scripts/analysis/handcode_kappa.py --report   # report from the saved labels
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict

SAMPLE = "docs/data/handcode-sample.json"


async def recode() -> None:
    from collabengine.analysis.coding import CodingStats, _code_one
    from collabengine.config import ExperimentConfig
    from collabengine.protocol import Message, Speaker

    config = ExperimentConfig.load("configs/llamacpp/hard.yaml")
    backend = config.backend.build()
    rows = json.load(open(SAMPLE))
    stats = CodingStats()
    # Two slots, not four: the experiment owns the card and this is 40 requests.
    sem = asyncio.Semaphore(2)

    async def one(row):
        async with sem:
            message = Message(
                turn=row["turn"], speaker=Speaker.AGENT,
                author="A1", content=row["text"], meta={},
            )
            row["local8b"] = (await _code_one(backend, message, 4000, stats)).value
            return row

    rows = await asyncio.gather(*(one(r) for r in rows))
    json.dump(rows, open(SAMPLE, "w"), indent=1)
    await backend.aclose()
    print(f"coded {len(rows)}; unparseable {stats.unparseable}, errors {stats.errors}")


def report() -> None:
    from collabengine.analysis import cohens_kappa, kappa_interval

    rows = json.load(open(SAMPLE))
    human = [r["human"] for r in rows]
    judge = [r["local8b"] for r in rows]
    lo, hi = kappa_interval(human, judge)
    agree = sum(a == b for a, b in zip(human, judge))
    print(
        f"kappa = {cohens_kappa(human, judge):.3f}  95% CI [{lo:.2f}, {hi:.2f}]  "
        f"raw agreement {agree}/{len(human)} = {agree / len(human):.0%}"
    )

    labels = sorted(set(human) | set(judge))
    matrix: dict[str, Counter] = defaultdict(Counter)
    for a, b in zip(human, judge):
        matrix[a][b] += 1
    width = max(len(x) for x in labels) + 1
    print("\nrows = human, cols = local 8B")
    print(" " * width + "".join(f"{x[:6]:>7}" for x in labels))
    for a in labels:
        print(f"{a:<{width}}" + "".join(f"{matrix[a][b] or '':>7}" for b in labels))

    # The single largest disagreement, isolated. `organize` is defined as
    # "divides up the work or sets procedure, without doing the task" -- and this
    # task *is* dividing work among workers, so the judge applies it to the
    # content rather than to the discourse act. Worth reporting separately
    # because it is a taxonomy defect, not only a judge one.
    kept = [(a, b) for a, b in zip(human, judge) if b != "organize"]
    if kept and len(kept) < len(human):
        k = cohens_kappa([a for a, _ in kept], [b for _, b in kept])
        print(f"\njudge said 'organize' {len(human) - len(kept)}/{len(human)} times; "
              f"kappa on the rest = {k:.3f} (n={len(kept)})")
    print("\nhuman:", Counter(human).most_common())
    print("judge:", Counter(judge).most_common())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report", action="store_true",
        help="report from the saved labels without calling the judge",
    )
    if not parser.parse_args().report:
        asyncio.run(recode())
    report()
