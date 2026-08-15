"""Can the judge be fixed? Four codebooks against the same 40 human labels.

Phase 2 is blocked at kappa = 0.241 (RESEARCH-LOG 4.13b), and 468 already-coded
messages are sitting on disk waiting on the answer. This is the cheapest
experiment in the project: 40 messages x 4 codebooks = 160 classifications at
8 tokens each.

Everything is held except the system prompt. Same 40 messages, same human
column, same `_code_one` path, same identity stripping, same per-turn seed,
temperature 0. What varies is the codebook, so what the kappas differ by is the
codebook -- see `analysis/codebooks.py` for the four and why each exists.

**The v3 row is not comparable to the others** and the table says so: three
categories is an easier question than eight, and kappa does not correct for
that. It corrects for chance agreement given the marginals, not for the
granularity of the question being asked. Read it as "a coarser Phase 2 is
available at this reliability", never as "the judge improved".

    python scripts/judge_sweep.py                 # code, then report
    python scripts/judge_sweep.py --report        # report saved labels
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

SAMPLE = Path("docs/data/handcode-sample.json")
OUT = Path("docs/data/judge-sweep.json")
#: Two slots. The card may be busy with a real run; this is 160 short requests
#: and finishing four minutes sooner is not worth perturbing an experiment.
CONCURRENCY = 2


async def sweep(config_path: str) -> None:
    from collabengine.analysis.codebooks import CODEBOOKS, parse_label
    from collabengine.analysis.coding import CodingStats, _code_one, _strip_identity
    from collabengine.config import ExperimentConfig
    from collabengine.protocol import Message, Speaker

    config = ExperimentConfig.load(config_path)
    backend = config.backend.build()
    rows = json.loads(SAMPLE.read_text(encoding="utf-8"))
    results: dict[str, list[str]] = {}

    for book in CODEBOOKS:
        stats = CodingStats()
        sem = asyncio.Semaphore(CONCURRENCY)

        async def one(row: dict, book=book, stats=stats, sem=sem) -> str:
            async with sem:
                message = Message(
                    turn=row["turn"], speaker=Speaker.AGENT,
                    author="A1", content=row["text"], meta={},
                )
                # The eight-label books can reuse the pipeline's own parser via
                # _code_one. The coarse book emits words that are not ActionType
                # members, so it needs its own parse -- but it must still go
                # through the same request shape, or the comparison includes the
                # request shape.
                if book.collapse is None:
                    action = await _code_one(
                        backend, message, 4000, stats, system=book.system
                    )
                    return action.value
                from collabengine.backends.base import ChatMessage, GenRequest

                response = await backend.generate(GenRequest(
                    messages=[
                        ChatMessage(role="system", content=book.system),
                        ChatMessage(
                            role="user",
                            content=_strip_identity(row["text"])[:4000],
                        ),
                    ],
                    max_tokens=8, temperature=0.0, top_p=1.0, seed=row["turn"],
                ))
                if not response.ok:
                    stats.record_error(response.error or "unknown")
                    return "meta"
                stats.record_success()
                label = parse_label(response.text, book)
                if label is None:
                    stats.unparseable += 1
                    return "meta"
                stats.coded += 1
                return label

        labels = await asyncio.gather(*(one(r) for r in rows))
        results[book.name] = list(labels)
        print(f"  {book.name}: coded {stats.coded}, "
              f"unparseable {stats.unparseable}, errors {stats.errors}")

    await backend.aclose()
    OUT.write_text(json.dumps(
        {"human": [r["human"] for r in rows], "judge": results}, indent=1,
    ), encoding="utf-8")
    print(f"wrote {OUT}")


def report() -> None:
    from collabengine.analysis import cohens_kappa, kappa_interval
    from collabengine.analysis.codebooks import CODEBOOKS

    data = json.loads(OUT.read_text(encoding="utf-8"))
    human_raw: list[str] = data["human"]

    print(f"\n{'codebook':<24}{'labels':>7}{'kappa':>8}{'95% CI':>16}{'raw':>9}")
    print("-" * 64)
    best: tuple[float, str] | None = None
    for book in CODEBOOKS:
        judge = data["judge"].get(book.name)
        if judge is None:
            continue
        human = [book.project(h) for h in human_raw]
        k = cohens_kappa(human, judge)
        lo, hi = kappa_interval(human, judge)
        agree = sum(a == b for a, b in zip(human, judge))
        n_labels = len(set(human) | set(judge))
        print(f"{book.name:<24}{n_labels:>7}{k:>8.3f}"
              f"{f'[{lo:+.2f}, {hi:+.2f}]':>16}"
              f"{f'{agree}/{len(human)}':>9}")
        # v3 is excluded from "best" on purpose: a coarser question cannot win a
        # comparison against a finer one on kappa alone.
        if book.collapse is None and (best is None or k > best[0]):
            best = (k, book.name)

    print("\nv3 is a different question, not a better answer to the same one:")
    print("three categories against eight. Compare it only with itself.")
    if best:
        print(f"\nbest eight-label codebook: {best[1]} at kappa = {best[0]:.3f}")
        print("Phase 2 needs ~0.6 to be worth re-running; 4.8's convergent")
        print("validity claim would need ~0.78.")

    for book in CODEBOOKS:
        judge = data["judge"].get(book.name)
        if judge is None:
            continue
        human = [book.project(h) for h in human_raw]
        labels = sorted(set(human) | set(judge))
        matrix: dict[str, Counter] = defaultdict(Counter)
        for a, b in zip(human, judge):
            matrix[a][b] += 1
        width = max(len(x) for x in labels) + 1
        print(f"\n{book.name} -- rows human, cols judge")
        print(" " * width + "".join(f"{x[:6]:>7}" for x in labels))
        for a in labels:
            print(f"{a:<{width}}" + "".join(f"{matrix[a][b] or '':>7}" for b in labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--config", default="configs/llamacpp/hard-ans.yaml")
    args = parser.parse_args()
    if not args.report:
        asyncio.run(sweep(args.config))
    report()
