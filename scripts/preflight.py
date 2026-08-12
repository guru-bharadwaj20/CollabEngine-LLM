#!/usr/bin/env python
"""Check a served backend before spending a night on it.

Three corpora in this project have been lost to a condition that was true
before the first episode ran and detectable in seconds: a context ceiling that
the largest prompts could not fit under (RESEARCH-LOG 3.10), and twice before
that a truncation that silently ate the head of the brief. Every one of them
produced a run that reported healthy episode counts and a corpus that could not
be analysed.

This script asks the three questions that would have caught all of them, in
increasing order of cost:

1. Is the server up, and is it serving the model this config names? A mismatch
   here breaks the identical-weights control silently, because every arm is
   equally wrong and the scores stay plausible.
2. Is a single slot big enough for the worst-case request this tier will make?
   Under llama.cpp the answer depends on `--parallel` as well as `-c`, since
   the context is divided across slots -- which is exactly the kind of
   arithmetic nobody redoes after changing one flag.
3. Does a real request of that size actually come back? Asked last because it
   is the only one that costs GPU time, and answered with one request.

    python scripts/preflight.py --config configs/llamacpp-xhard.yaml

Exit status is 0 when the run may proceed, 1 when it must not, 2 on a usage or
config error. Chain it: `python scripts/preflight.py --config C && collabengine
pipeline --config C --phases baseline,solo`.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from collabengine.backends.base import ChatMessage, GenRequest  # noqa: E402
from collabengine.backends.openai_compat import (  # noqa: E402
    OpenAICompatBackend,
    is_context_overflow,
)
from collabengine.config import ExperimentConfig  # noqa: E402
from collabengine.orchestrator.episode import build_system_prompt  # noqa: E402
from collabengine.orchestrator.team import build_team  # noqa: E402
from collabengine.tasks.generator import generate  # noqa: E402

CHARS_PER_TOKEN = 2.6
"""Fallback ratio when the server has no `/tokenize`.

Measured on this task's own briefs under Qwen3's tokenizer: 4,639 characters to
1,704 tokens at `hard`, 6,694 to 2,520 at `xhard`. The text is dense with
identifiers and punctuation, so it tokenizes far worse than prose. Only ever
used to produce an estimate that is labelled as one."""

BANNER_OVERHEAD_TOKENS = 400
"""Round banners, the moderator's final-answer call, and chat-template markup.

Rounded up. It is a margin on a check whose whole job is to fail early."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="experiment YAML")
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="skip the live worst-case request (checks 1 and 2 only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="instance seed to size the brief with (default: config seed_start)",
    )
    args = parser.parse_args(argv)

    try:
        config = ExperimentConfig.load(args.config)
    except (OSError, ValueError) as exc:
        print(f"cannot load {args.config}: {exc}", file=sys.stderr)
        return 2

    if config.backend.kind != "openai":
        print(
            f"backend.kind is {config.backend.kind!r}; this script only checks "
            "served backends (kind: openai)",
            file=sys.stderr,
        )
        return 2

    return asyncio.run(_run(config, seed=args.seed, probe=not args.skip_probe))


async def _run(config: ExperimentConfig, *, seed: int | None, probe: bool) -> int:
    backend = config.backend.build()
    assert isinstance(backend, OpenAICompatBackend)

    # The answer turn is the largest single request the run makes, so it is the
    # one the slot has to hold. Checking against `max_tokens` when
    # `answer_max_tokens` is set would under-state the requirement by exactly
    # the amount the answer-budget fix adds -- a guard that passes because it is
    # measuring the wrong turn (cf. 3.12).
    answer_cap = config.team.answer_max_tokens or config.team.max_tokens
    required_ctx = config.backend.max_model_len + answer_cap
    failed = False

    try:
        report = await backend.preflight(required_ctx=required_ctx)
        print(f"-- server ({config.backend.base_url})")
        for line in report.lines():
            print(f"   {line}")
        if not report.ok:
            failed = True
        if report.reachable and report.ctx_per_slot is None:
            print(
                "   note: no /props; slot size unverified (vLLM, or an older "
                "llama.cpp). The live probe below is the only check on it."
            )

        print(f"-- worst case at difficulty={config.team.difficulty}")
        brief = _worst_case_brief(
            config, seed if seed is not None else config.seed_start
        )
        brief_tokens, exact = await _count_tokens(backend, brief)
        turns = config.team.n_agents * config.team.rounds - 1
        worst = brief_tokens + turns * config.team.max_tokens + BANNER_OVERHEAD_TOKENS
        label = "measured" if exact else f"estimated at {CHARS_PER_TOKEN} chars/token"

        print(f"   brief           {brief_tokens} tokens ({label})")
        print(
            f"   + {turns} prior turns at the {config.team.max_tokens}-token cap "
            f"+ {BANNER_OVERHEAD_TOKENS} overhead"
        )
        print(f"   worst prompt    {worst} tokens")
        print(f"   max_model_len   {config.backend.max_model_len}")
        print(f"   required slot   {required_ctx} tokens")

        if worst > config.backend.max_model_len:
            print(
                f"   PROBLEM         the worst case exceeds max_model_len by "
                f"{worst - config.backend.max_model_len} tokens. Raise it (and "
                "the server's -c to match), or the final round of a team "
                "episode will be rejected -- which is precisely the turn that "
                "submits the answer."
            )
            failed = True

        if probe and report.reachable:
            failed |= not await _probe(backend, config, brief, brief_tokens)
        elif probe:
            print("-- live probe skipped: server unreachable")
    finally:
        await backend.aclose()

    print()
    print("PREFLIGHT FAILED" if failed else "preflight ok")
    return 1 if failed else 0


def _worst_case_brief(config: ExperimentConfig, seed: int) -> str:
    """The largest system prompt this tier produces, built the way the run does.

    Built through `build_system_prompt` rather than `render_instance` so the
    team brief, the symmetry-breaking scratch line, and the answer-format block
    are all counted -- every one of which the real prompt carries and a
    hand-rolled estimate forgets.
    """
    instance = generate(seed, config.team.difficulty)
    agents = build_team(config.team, episode_seed=seed)
    return max(
        (
            build_system_prompt(agent, instance, config.team.n_agents)
            for agent in agents
        ),
        key=len,
    )


async def _count_tokens(backend: OpenAICompatBackend, text: str) -> tuple[int, bool]:
    """Exact count from llama.cpp's `/tokenize`, or a labelled estimate.

    Returns (tokens, exact). The endpoint lives at the server root rather than
    under /v1, because it is not part of the OpenAI surface.
    """
    root = backend.base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")]
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{root}/tokenize", json={"content": text})
        if response.status_code == 200:
            tokens = response.json().get("tokens")
            if isinstance(tokens, list):
                return len(tokens), True
    except (httpx.TimeoutException, httpx.TransportError, ValueError):
        pass
    return int(len(text) / CHARS_PER_TOKEN), False


async def _probe(
    backend: OpenAICompatBackend,
    config: ExperimentConfig,
    brief: str,
    brief_tokens: int,
) -> bool:
    """One real request at the worst-case size.

    Padded to `max_model_len` with filler rather than with a real transcript,
    because what is being tested is whether the slot accepts a prompt of that
    length and returns `team.max_tokens` on top of it. The content is
    irrelevant to that question and generating a realistic transcript would
    cost the night this script exists to protect.

    The timing it prints is worth as much as the pass/fail: it is the per-turn
    cost of the largest turn in the tier, and multiplying it out is the only
    honest estimate of how long the corpus takes.
    """
    print("-- live probe")
    # The padding is measured, not estimated from characters. Getting it wrong
    # in the generous direction would push the probe past the slot and report a
    # failure that is the probe's own -- a false alarm on the one check whose
    # authority rests on being trustworthy enough to stop a night's work.
    unit = "The above brief is under review. " * 8 + "\n"
    unit_tokens, _ = await _count_tokens(backend, unit)
    deficit = max(0, config.backend.max_model_len - brief_tokens - 64)
    filler = unit * max(1, deficit // max(1, unit_tokens))

    request = GenRequest(
        messages=[
            ChatMessage("system", brief),
            ChatMessage("user", filler),
            ChatMessage("user", "Reply with the single word ACK."),
        ],
        max_tokens=config.team.answer_max_tokens or config.team.max_tokens,
        temperature=config.team.temperature,
        top_p=config.team.top_p,
        seed=0,
    )

    started = time.monotonic()
    response = await backend.generate(request)
    elapsed = time.monotonic() - started

    if response.ok:
        print(f"   accepted        {response.prompt_tokens} prompt tokens")
        print(
            f"   generated       {response.completion_tokens} tokens in "
            f"{elapsed:.1f}s (finish_reason={response.finish_reason})"
        )
        episodes = config.n_episodes
        per_episode = config.team.n_agents * config.team.rounds
        hours = elapsed * per_episode * episodes / max(1, config.max_concurrency) / 3600
        print(
            f"   implies roughly {hours:.1f}h for {episodes} team episodes at "
            f"max_concurrency={config.max_concurrency}, if every turn were this "
            "large (they are not; early rounds are much shorter)"
        )
        return True

    error = response.error or "unknown"
    print(f"   REJECTED        {error}")
    if is_context_overflow(error):
        print(
            "   The slot is too small for this tier. Under llama.cpp, -c is "
            f"divided across --parallel slots: for {config.max_concurrency} "
            f"slots this run needs -c "
            f"{(config.backend.max_model_len + (config.team.answer_max_tokens or config.team.max_tokens)) * config.max_concurrency}. "
            "Regenerating episodes will not help -- the failure is "
            "deterministic in the prompt length."
        )
    return False


if __name__ == "__main__":
    raise SystemExit(main())
