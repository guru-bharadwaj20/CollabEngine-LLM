"""Instance generation for the code family.

Ground-truth-first, exactly as the allocation generator is. A pipeline of
transformation stages is drawn, the reference module is what *defines* the
answer, and every hidden test's expected value is computed from it. Instances
are therefore satisfiable by construction and the ceiling is 1.0 -- which is
what makes a difficulty curve on this family readable next to the one on the
allocation family rather than merely alongside it.

Generated rather than scraped, for the reason PLAN 3 gives: an 8B model that has
seen MBPP during pretraining is not being measured on the task, it is being
measured on recall, and the difference is invisible in the score.

**The planted bug is the second thing generation fixes, and the ordering
matters.** One stage of the starter code is mutated, and the test suite is then
sorted *around* that mutation: base and edge inputs are kept only where the
correct and mutated pipelines agree, verify inputs only where they disagree. A
submission that copies the starter code verbatim therefore loses `verify` and
keeps `base` and `edge` -- the code analogue of producing a feasible schedule
while failing the allocation task's audit. Without that sort the four component
scores would move together and the agent x component interaction would have
nothing to interact with.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from collabengine.tasks.code.schema import (
    CodeInstance,
    HiddenTest,
    RuleSpec,
    TestKind,
)

# --------------------------------------------------------------------------
# The rule library.
#
# Each kind carries five things that have to stay in step: a spec sentence, a
# reference implementation, the source that implementation is *emitted* as, one
# mutation of it, and a way to build an input that exposes the mutation.
#
# The spec sentence is written to contradict the mutation flatly rather than
# subtly. That mirrors the allocation task's rule that planted errors are always
# skill violations: the bug must be decidable from what the agents were given,
# never a matter of guessing our intent.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuleKind:
    kind: str
    bug: str
    """One-word name for the mutation. Recorded on the instance for error analysis."""
    draw: Callable[[random.Random, int], dict[str, Any]]
    describe: Callable[[dict[str, Any]], str]
    apply: Callable[[list[int], dict[str, Any]], list[int]]
    apply_bug: Callable[[list[int], dict[str, Any]], list[int]]
    source: Callable[[str, dict[str, Any]], str]
    source_bug: Callable[[str, dict[str, Any]], str]
    inject: Callable[[list[int], dict[str, Any], random.Random], list[int]]
    """Return a copy of `values` altered to make the mutation observable."""


def _signed(delta: int) -> str:
    """`- 5` rather than `+ -5`. The starter code is read by the model, and
    source that looks machine-written invites the model to rewrite rather than
    check it."""
    return f"+ {delta}" if delta >= 0 else f"- {-delta}"


def _keep_at_least() -> RuleKind:
    return RuleKind(
        kind="keep_at_least",
        bug="drops_the_boundary",
        draw=lambda rng, span: {"threshold": rng.randint(-span // 2, span // 2)},
        describe=lambda p: (
            f"keep only the values that are greater than or equal to "
            f"{p['threshold']}, and discard the rest"
        ),
        apply=lambda xs, p: [v for v in xs if v >= p["threshold"]],
        apply_bug=lambda xs, p: [v for v in xs if v > p["threshold"]],
        source=lambda n, p: (
            f"def {n}(values):\n"
            f"    return [v for v in values if v >= {p['threshold']}]\n"
        ),
        source_bug=lambda n, p: (
            f"def {n}(values):\n"
            f"    return [v for v in values if v > {p['threshold']}]\n"
        ),
        inject=lambda xs, p, rng: xs + [p["threshold"]],
    )


def _drop_above() -> RuleKind:
    return RuleKind(
        kind="drop_above",
        bug="drops_the_boundary",
        draw=lambda rng, span: {"threshold": rng.randint(0, span)},
        describe=lambda p: (
            f"discard every value that is strictly greater than {p['threshold']}, "
            f"keeping the rest"
        ),
        apply=lambda xs, p: [v for v in xs if v <= p["threshold"]],
        apply_bug=lambda xs, p: [v for v in xs if v < p["threshold"]],
        source=lambda n, p: (
            f"def {n}(values):\n"
            f"    return [v for v in values if v <= {p['threshold']}]\n"
        ),
        source_bug=lambda n, p: (
            f"def {n}(values):\n"
            f"    return [v for v in values if v < {p['threshold']}]\n"
        ),
        inject=lambda xs, p, rng: xs + [p["threshold"]],
    )


def _drop_multiples() -> RuleKind:
    return RuleKind(
        kind="drop_multiples",
        bug="ignores_zero_and_negatives",
        draw=lambda rng, span: {"divisor": rng.choice([2, 3, 4, 5])},
        describe=lambda p: (
            f"discard every value that is an exact multiple of {p['divisor']}; "
            f"zero and negative numbers count as multiples like any other"
        ),
        apply=lambda xs, p: [v for v in xs if v % p["divisor"] != 0],
        apply_bug=lambda xs, p: [
            v for v in xs if not (v > 0 and v % p["divisor"] == 0)
        ],
        source=lambda n, p: (
            f"def {n}(values):\n"
            f"    return [v for v in values if v % {p['divisor']} != 0]\n"
        ),
        source_bug=lambda n, p: (
            f"def {n}(values):\n"
            f"    return [v for v in values if not (v > 0 and v % {p['divisor']} == 0)]\n"
        ),
        inject=lambda xs, p, rng: xs
        + [rng.choice([0, -p["divisor"], -2 * p["divisor"]])],
    )


def _shift() -> RuleKind:
    return RuleKind(
        kind="shift",
        bug="skips_non_positive",
        draw=lambda rng, span: {"delta": rng.choice([-5, -3, -2, 2, 3, 5])},
        describe=lambda p: (
            f"add {p['delta']} to every value, whatever its sign"
        ),
        apply=lambda xs, p: [v + p["delta"] for v in xs],
        apply_bug=lambda xs, p: [v + p["delta"] if v > 0 else v for v in xs],
        source=lambda n, p: (
            f"def {n}(values):\n"
            f"    return [v {_signed(p['delta'])} for v in values]\n"
        ),
        source_bug=lambda n, p: (
            f"def {n}(values):\n"
            f"    return [v {_signed(p['delta'])} if v > 0 else v for v in values]\n"
        ),
        inject=lambda xs, p, rng: xs + [rng.choice([0, -1, -2])],
    )


def _clamp_low() -> RuleKind:
    return RuleKind(
        kind="clamp_low",
        bug="only_clamps_negatives",
        # `floor` starts at 2 so `inject` always has a value in [0, floor) to
        # place -- a floor of 1 would leave only 0 and make the probe degenerate.
        draw=lambda rng, span: {"floor": rng.randint(2, max(3, span // 3))},
        describe=lambda p: (
            f"raise every value below {p['floor']} up to exactly {p['floor']}, "
            f"and leave the rest alone"
        ),
        apply=lambda xs, p: [max(v, p["floor"]) for v in xs],
        apply_bug=lambda xs, p: [max(v, p["floor"]) if v < 0 else v for v in xs],
        source=lambda n, p: (
            f"def {n}(values):\n"
            f"    return [max(v, {p['floor']}) for v in values]\n"
        ),
        source_bug=lambda n, p: (
            f"def {n}(values):\n"
            f"    return [max(v, {p['floor']}) if v < 0 else v for v in values]\n"
        ),
        inject=lambda xs, p, rng: xs + [rng.randrange(0, p["floor"])],
    )


def _dedupe() -> RuleKind:
    return RuleKind(
        kind="dedupe",
        bug="drops_every_copy",
        draw=lambda rng, span: {},
        describe=lambda p: (
            "keep only the first occurrence of each value and discard later "
            "repeats of it"
        ),
        apply=lambda xs, p: _dedupe_apply(xs),
        apply_bug=lambda xs, p: [v for v in xs if xs.count(v) == 1],
        source=lambda n, p: (
            f"def {n}(values):\n"
            f"    seen = set()\n"
            f"    out = []\n"
            f"    for v in values:\n"
            f"        if v not in seen:\n"
            f"            seen.add(v)\n"
            f"            out.append(v)\n"
            f"    return out\n"
        ),
        source_bug=lambda n, p: (
            f"def {n}(values):\n"
            f"    return [v for v in values if values.count(v) == 1]\n"
        ),
        inject=lambda xs, p, rng: (
            xs + [rng.choice(xs)] if xs else [3, 3]
        ),
    )


def _dedupe_apply(xs: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for v in xs:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


RULE_KINDS: dict[str, RuleKind] = {
    k.kind: k
    for k in (
        _keep_at_least(),
        _drop_above(),
        _drop_multiples(),
        _shift(),
        _clamp_low(),
        _dedupe(),
    )
}

#: Sorted so `rng.sample` over it is stable across interpreter runs.
RULE_NAMES: tuple[str, ...] = tuple(sorted(RULE_KINDS))


# --------------------------------------------------------------------------
# Aggregation. Never mutated -- the planted bug always lives in a rule, so
# `verify` stays a statement about the starter code rather than about the tail
# of the module.
# --------------------------------------------------------------------------

AGGREGATES: tuple[str, ...] = ("total", "count", "largest", "spread")


def _aggregate(kind: str, xs: list[int]) -> int:
    if kind == "total":
        return sum(xs)
    if kind == "count":
        return len(xs)
    if kind == "largest":
        return max(xs)
    if kind == "spread":
        return max(xs) - min(xs)
    raise ValueError(f"unknown aggregate {kind!r}")


def describe_aggregate(kind: str) -> str:
    if kind == "total":
        return "the sum of the values that remain"
    if kind == "count":
        return "how many values remain"
    if kind == "largest":
        return "the largest value that remains"
    if kind == "spread":
        return "the largest remaining value minus the smallest remaining value"
    raise ValueError(f"unknown aggregate {kind!r}")


def _aggregate_source(kind: str) -> str:
    if kind == "total":
        return "    return sum(remaining)\n"
    if kind == "count":
        return "    return len(remaining)\n"
    if kind == "largest":
        return "    return max(remaining)\n"
    if kind == "spread":
        return "    return max(remaining) - min(remaining)\n"
    raise ValueError(f"unknown aggregate {kind!r}")


# --------------------------------------------------------------------------
# Difficulty.
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DifficultySpec:
    """Knobs controlling instance hardness.

    `name` is recorded on the instance so runs can be grouped without recomputing.
    """

    name: str
    n_rules: int
    """Pipeline stages. The dominant knob: every stage is another sentence the
    submission has to get exactly right, and another place the bug can hide."""
    n_base: int
    n_edge: int
    n_verify: int
    list_len: int
    value_span: int
    require_helper: bool
    """Whether `apply_rules` is demanded alongside `solve`.

    The only thing this changes is the `syntax` component's denominator. Base,
    edge and verify tests call `solve` and nothing else, deliberately: a hidden
    test that called a function only some tiers require would fold `syntax` into
    the other three and cost the design its independent columns."""


#: Ordered easy -> hard. `cli.calibrate` sweeps in this order, so it is part of
#: the family's interface rather than a presentation detail.
DIFFICULTY_ORDER: tuple[str, ...] = ("tiny", "easy", "medium", "hard", "xhard")

PRESETS: dict[str, DifficultySpec] = {
    "tiny": DifficultySpec(
        name="tiny",
        n_rules=1,
        n_base=2,
        n_edge=2,
        n_verify=1,
        list_len=4,
        value_span=8,
        require_helper=False,
    ),
    "easy": DifficultySpec(
        name="easy",
        n_rules=2,
        n_base=3,
        n_edge=3,
        n_verify=2,
        list_len=6,
        value_span=12,
        require_helper=False,
    ),
    # The tier the calibration pass starts from, per Final Sweep 2.1: guard the
    # failure floor by pitching the second family low and moving up, rather than
    # opening at a tier where an 8B floors out for reasons that have nothing to
    # do with whether a team helps.
    "medium": DifficultySpec(
        name="medium",
        n_rules=3,
        n_base=4,
        n_edge=4,
        n_verify=2,
        list_len=8,
        value_span=20,
        require_helper=True,
    ),
    "hard": DifficultySpec(
        name="hard",
        n_rules=4,
        n_base=5,
        n_edge=5,
        n_verify=3,
        list_len=12,
        value_span=30,
        require_helper=True,
    ),
    # Scaled along the same axes as the allocation family's `xhard`, and for the
    # same reason: total work grows while the per-agent budget does not, which is
    # the first condition under which dividing the work could pay.
    "xhard": DifficultySpec(
        name="xhard",
        n_rules=5,
        n_base=6,
        n_edge=6,
        n_verify=4,
        list_len=16,
        value_span=40,
        require_helper=True,
    ),
}


# --------------------------------------------------------------------------
# Generation.
# --------------------------------------------------------------------------

#: How many random draws to spend looking for inputs of each class before
#: declaring the instance unlucky and reseeding. Generous: a failed draw costs
#: microseconds, a reseed costs a whole instance.
_DRAW_BUDGET = 400


def generate(seed: int, difficulty: str = "medium") -> CodeInstance:
    """Build one satisfiable problem.

    Deterministic in `(seed, difficulty)`: the same pair always yields an
    identical instance, so runs are reproducible from config alone and a
    transcript need not embed the instance to be rescorable offline.
    """
    if difficulty not in PRESETS:
        raise ValueError(
            f"unknown difficulty {difficulty!r}; expected one of {sorted(PRESETS)}"
        )
    spec = PRESETS[difficulty]
    rng = random.Random(seed)

    rules = _draw_rules(rng, spec)
    aggregate = rng.choice(AGGREGATES)
    empty_default = rng.choice((-1, -100, 7))
    bug_index = rng.randrange(len(rules))

    ref = _pipeline(rules, bug_index=None)
    bug = _pipeline(rules, bug_index=bug_index)

    def solve_ref(xs: list[int]) -> int:
        out = ref(xs)
        return empty_default if not out else _aggregate(aggregate, out)

    def solve_bug(xs: list[int]) -> int:
        out = bug(xs)
        return empty_default if not out else _aggregate(aggregate, out)

    tests = _sort_tests_around_the_bug(
        rng, spec, rules, bug_index, ref, bug, solve_ref, solve_bug
    )
    if tests is None:
        # The draw produced a pipeline whose mutation is unobservable behind the
        # other stages -- a `shift` bug behind a filter that removes every
        # non-positive value, say. Reseeding deterministically is cheaper than
        # backtracking and keeps generation total, exactly as the allocation
        # generator does when a skill draw leaves a job unplaceable.
        return generate(seed * 2654435761 % (2**31 - 1) + 1, difficulty)

    required = ("apply_rules", "solve") if spec.require_helper else ("solve",)

    return CodeInstance(
        instance_id=f"code-{difficulty}-{seed}",
        seed=seed,
        difficulty=difficulty,
        rules=rules,
        aggregate=aggregate,
        empty_default=empty_default,
        required_functions=required,
        starter_code=render_starter(rules, bug_index),
        tests=tests,
        planted_bug_rule=rules[bug_index].rid,
        planted_bug_kind=RULE_KINDS[rules[bug_index].kind].bug,
        reference_code=render_module(rules, aggregate, empty_default),
    )


def _draw_rules(rng: random.Random, spec: DifficultySpec) -> tuple[RuleSpec, ...]:
    """One stage per kind, so no instance repeats a sentence.

    Repeating a kind would make two stages collapse into one in the reader's head
    without collapsing in the grader, which is difficulty from confusion rather
    than from work.
    """
    kinds = rng.sample(RULE_NAMES, min(spec.n_rules, len(RULE_NAMES)))
    return tuple(
        RuleSpec(
            rid=f"R{i + 1}",
            kind=kind,
            params=RULE_KINDS[kind].draw(rng, spec.value_span),
        )
        for i, kind in enumerate(kinds)
    )


def _pipeline(
    rules: tuple[RuleSpec, ...], *, bug_index: int | None
) -> Callable[[list[int]], list[int]]:
    """Compose the stages, optionally running one of them mutated."""

    def run(values: list[int]) -> list[int]:
        out = list(values)
        for i, r in enumerate(rules):
            kind = RULE_KINDS[r.kind]
            out = (
                kind.apply_bug(out, r.params)
                if i == bug_index
                else kind.apply(out, r.params)
            )
        return out

    return run


def _sort_tests_around_the_bug(
    rng: random.Random,
    spec: DifficultySpec,
    rules: tuple[RuleSpec, ...],
    bug_index: int,
    ref: Callable[[list[int]], list[int]],
    bug: Callable[[list[int]], list[int]],
    solve_ref: Callable[[list[int]], int],
    solve_bug: Callable[[list[int]], int],
) -> tuple[HiddenTest, ...] | None:
    """Partition drawn inputs into the three graded classes, or fail.

    The three predicates below are the whole independence argument:

      base    correct and mutated agree, and something survives the rules
      edge    correct and mutated agree, on an input from the awkward pool
      verify  correct and mutated disagree, and both leave something behind

    "Both leave something behind" on base and verify is what keeps a submission
    that forgets the empty case from losing those two components as well as
    `edge` -- the empty path is reachable only from the edge pool, by design.
    """
    base = _draw_base(rng, spec, ref, bug, solve_ref, solve_bug)
    if base is None:
        return None
    edge = _draw_edge(spec, ref, bug, solve_ref, solve_bug)
    if edge is None:
        return None
    verify = _draw_verify(rng, spec, rules, bug_index, ref, bug, solve_ref, solve_bug)
    if verify is None:
        return None

    tests: list[HiddenTest] = []
    for kind, inputs in (
        (TestKind.BASE, base),
        (TestKind.EDGE, edge),
        (TestKind.VERIFY, verify),
    ):
        for n, xs in enumerate(inputs):
            tests.append(
                HiddenTest(
                    tid=f"{kind.value}-{n + 1}",
                    kind=kind,
                    function="solve",
                    args=tuple(xs),
                    expected=solve_ref(list(xs)),
                )
            )
    return tuple(tests)


def _draw_base(
    rng: random.Random,
    spec: DifficultySpec,
    ref: Callable[[list[int]], list[int]],
    bug: Callable[[list[int]], list[int]],
    solve_ref: Callable[[list[int]], int],
    solve_bug: Callable[[list[int]], int],
) -> list[list[int]] | None:
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(_DRAW_BUDGET):
        if len(out) >= spec.n_base:
            return out
        xs = [
            rng.randint(-spec.value_span, spec.value_span)
            for _ in range(spec.list_len)
        ]
        key = tuple(xs)
        if key in seen or not ref(xs) or not bug(xs):
            continue
        if solve_ref(xs) != solve_bug(xs):
            continue
        seen.add(key)
        out.append(xs)
    return out if len(out) >= spec.n_base else None


def _draw_edge(
    spec: DifficultySpec,
    ref: Callable[[list[int]], list[int]],
    bug: Callable[[list[int]], list[int]],
    solve_ref: Callable[[list[int]], int],
    solve_bug: Callable[[list[int]], int],
) -> list[list[int]] | None:
    """The awkward pool, in a fixed order so the empty case is always test 1.

    Fixed rather than sampled because the empty list is the single most
    informative edge input on this task -- it is the only one that reaches the
    `empty_default` branch on every instance -- and leaving its presence to a
    random draw would make the `edge` component mean different things on
    different seeds.
    """
    span = spec.value_span
    pool: list[list[int]] = [
        [],
        [0],
        [span],
        [-span],
        [0, 0, 0],
        [span * 10, -span * 10],
        [1, 1, 1, 1],
        [-1],
        [span, span],
        [0, 1, -1],
        [span * 10],
        [-span * 10],
        [2, -2],
        [span - 1, 1 - span],
    ]
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for xs in pool:
        if len(out) >= spec.n_edge:
            break
        key = tuple(xs)
        if key in seen or solve_ref(list(xs)) != solve_bug(list(xs)):
            continue
        seen.add(key)
        out.append(list(xs))
    return out if len(out) >= spec.n_edge else None


def _draw_verify(
    rng: random.Random,
    spec: DifficultySpec,
    rules: tuple[RuleSpec, ...],
    bug_index: int,
    ref: Callable[[list[int]], list[int]],
    bug: Callable[[list[int]], list[int]],
    solve_ref: Callable[[list[int]], int],
    solve_bug: Callable[[list[int]], int],
) -> list[list[int]] | None:
    """Inputs the mutation is visible on.

    Built by asking the mutated rule for a value that exposes it and then
    *checking* the whole pipeline rather than trusting the injection: the
    stage runs somewhere in the middle, and an earlier filter may well remove
    the probe before it reaches the mutation. Checking is what makes the failure
    mode a reseed instead of a silently unfalsifiable component.
    """
    kind = RULE_KINDS[rules[bug_index].kind]
    params = rules[bug_index].params
    out: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for _ in range(_DRAW_BUDGET):
        if len(out) >= spec.n_verify:
            return out
        carrier = [
            rng.randint(-spec.value_span, spec.value_span)
            for _ in range(max(1, spec.list_len - 1))
        ]
        xs = kind.inject(carrier, params, rng)
        rng.shuffle(xs)
        key = tuple(xs)
        if key in seen or not ref(xs) or not bug(xs):
            continue
        if solve_ref(xs) == solve_bug(xs):
            continue
        seen.add(key)
        out.append(xs)
    return out if len(out) >= spec.n_verify else None


# --------------------------------------------------------------------------
# Source emission.
#
# One renderer serves the reference module, the starter code, and the degraded
# variants the mock backend and the tests need. Emitting them from one template
# is what stops the reference and the starter drifting apart -- a drift that
# would move the ceiling below 1.0 without anything failing loudly.
# --------------------------------------------------------------------------


def _stage_name(index: int) -> str:
    return f"_stage_{index + 1}"


def render_starter(rules: tuple[RuleSpec, ...], bug_index: int) -> str:
    """The agent-visible helpers, one of them wrong.

    Only the stages: `apply_rules` and `solve` are what the submission is for,
    and handing over a working `solve` would leave nothing to write.
    """
    blocks = []
    for i, r in enumerate(rules):
        kind = RULE_KINDS[r.kind]
        emit = kind.source_bug if i == bug_index else kind.source
        blocks.append(emit(_stage_name(i), r.params))
    return "\n".join(blocks).rstrip() + "\n"


def render_module(
    rules: tuple[RuleSpec, ...],
    aggregate: str,
    empty_default: int,
    *,
    fix_bug: bool = True,
    bug_index: int | None = None,
    empty_guard: bool = True,
    define_helper: bool = True,
    correct_rules: bool = True,
) -> str:
    """A complete module, optionally degraded along exactly one axis.

    The four keyword knobs map one-to-one onto the four components, which is what
    lets the mock backend model a genuinely specialized agent on this family and
    lets the tests damage one column without touching the others:

      define_helper  -> syntax   (drop `apply_rules`, which the spec demanded)
      correct_rules  -> base     (drop a stage, so ordinary inputs come out wrong)
      empty_guard    -> edge     (drop the empty case, which only edge inputs reach)
      fix_bug        -> verify   (keep the starter's mutation)
    """
    if not fix_bug and bug_index is None:
        raise ValueError("fix_bug=False needs the bug_index to reproduce")

    stages = list(range(len(rules)))
    if not correct_rules and len(stages) > 1:
        # Drop a stage that is not the planted one, so `base` fails for a reason
        # the `verify` column is not already reporting.
        droppable = [i for i in stages if i != bug_index] or stages
        stages.remove(droppable[-1])
    elif not correct_rules:
        stages = []

    blocks: list[str] = []
    for i, r in enumerate(rules):
        kind = RULE_KINDS[r.kind]
        emit = kind.source_bug if (not fix_bug and i == bug_index) else kind.source
        blocks.append(emit(_stage_name(i), r.params))

    calls = "".join(f"    values = {_stage_name(i)}(values)\n" for i in stages)

    if define_helper:
        blocks.append(f"def apply_rules(values):\n    values = list(values)\n{calls}    return values\n")
        body = "    remaining = apply_rules(values)\n"
    else:
        body = "    remaining = list(values)\n" + calls.replace("values", "remaining")

    guard = (
        f"    if not remaining:\n        return {empty_default}\n"
        if empty_guard
        else ""
    )
    blocks.append(f"def solve(values):\n{body}{guard}{_aggregate_source(aggregate)}")
    return "\n".join(blocks)


def reference_source(instance: CodeInstance) -> str:
    """The module that scores 1.0. Withheld from agents; used by the audit."""
    return instance.reference_code


def variant_source(
    instance: CodeInstance,
    *,
    fix_bug: bool = True,
    empty_guard: bool = True,
    define_helper: bool = True,
    correct_rules: bool = True,
) -> str:
    """A degraded reference, for the mock backend and the independence tests."""
    bug_index = next(
        i for i, r in enumerate(instance.rules) if r.rid == instance.planted_bug_rule
    )
    return render_module(
        instance.rules,
        instance.aggregate,
        instance.empty_default,
        fix_bug=fix_bug,
        bug_index=bug_index,
        empty_guard=empty_guard,
        define_helper=define_helper,
        correct_rules=correct_rules,
    )


def infer_variant(instance: CodeInstance, code: str) -> dict[str, bool]:
    """Which of the four knobs a submission appears to have got right.

    Structural, not behavioural: it reads the text for the markers the renderer
    emits. That is enough for the mock backend, whose only need is to recover its
    own prior state *from the transcript* -- the property that makes frozen
    ablation work at all, and the reason the mock holds nothing on the backend.
    It is not a grader and must never be used as one.
    """
    bug_index = next(
        i for i, r in enumerate(instance.rules) if r.rid == instance.planted_bug_rule
    )
    kind = RULE_KINDS[instance.rules[bug_index].kind]
    params = instance.rules[bug_index].params
    buggy_body = kind.source_bug(_stage_name(bug_index), params).split("\n", 1)[1]

    return {
        "define_helper": "def apply_rules(" in code,
        # `= _stage_n(` appears only where a stage is *called*; the definition
        # reads `def _stage_n(`, so this does not count a helper that was
        # written and then never wired into the pipeline.
        "correct_rules": all(
            f"= {_stage_name(i)}(" in code for i in range(len(instance.rules))
        ),
        "empty_guard": "if not remaining:" in code,
        "fix_bug": buggy_body not in code,
    }
