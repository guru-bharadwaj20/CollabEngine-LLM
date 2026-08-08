"""Does the scoring metric, rather than the difficulty, lack dynamic range?

Solo scored 0.879 at medium and 0.842 at hard -- a 50% increase in instance size
moved the score by 0.037. The suspicion is that fraction-of-constraints-satisfied
cannot discriminate, because adding constraints adds to numerator and
denominator together.

Instances are deterministic in (seed, difficulty), so this re-scores solutions
already on disk under alternative metrics. No generation, no GPU. What matters
is not which metric gives a lower number -- any metric can do that -- but which
separates a four-agent team from one agent by more standard deviations.
"""

import json
import statistics as st
import sys
from collections import defaultdict

from collabengine.tasks.generator import generate
from collabengine.tasks.grader import grade
from collabengine.tasks.schema import ALL_COMPONENTS, Component, Solution


def metrics(instance, solution):
    """Current score, plus two candidates with more headroom."""
    g = grade(instance, solution)
    if solution.malformed:
        return None
    sat = g.satisfied

    strict = {}
    for comp in ALL_COMPONENTS:
        if comp is Component.VERIFICATION:
            strict[comp] = g.per_component[comp]
            continue
        cs = instance.constraints_for(comp)
        # All-or-nothing: a schedule that violates one capacity limit is not
        # 90% feasible, it is infeasible.
        strict[comp] = float(all(sat[c.cid] for c in cs)) if cs else 1.0

    return {
        "fraction": g.overall,
        "strict": sum(strict.values()) / len(strict),
        "feasible": float(all(sat.values())),   # whole instance, all-or-nothing
        "per_component_fraction": dict(g.per_component),
        "per_component_strict": strict,
    }


def load(path):
    out = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        sol = Solution.from_dict(r["solution"])
        inst = generate(r["instance_seed"], r["difficulty"])
        m = metrics(inst, sol)
        if m:
            out[r["condition"]].append(m)
    return out


def report(by_condition, title):
    print(f"\n=== {title} ===")
    conds = [c for c in ("solo", "baseline") if c in by_condition]
    if len(conds) < 2:
        print("  need both solo and baseline; have:", list(by_condition))
        conds = list(by_condition)

    for key in ("fraction", "strict", "feasible"):
        line = f"{key:>10}: "
        vals = {}
        for c in conds:
            v = [m[key] for m in by_condition[c]]
            vals[c] = v
            line += f"{c} {st.mean(v):.3f}  "
        if len(conds) == 2:
            a, b = vals[conds[0]], vals[conds[1]]
            gap = st.mean(b) - st.mean(a)
            pooled = ((st.pstdev(a) ** 2 + st.pstdev(b) ** 2) / 2) ** 0.5
            d = gap / pooled if pooled else float("nan")
            line += f"| gap {gap:+.3f}  Cohen's d {d:+.2f}"
        print(line)

    print("  per-component (fraction -> strict):")
    for comp in ALL_COMPONENTS:
        row = f"    {comp.value:<14}"
        for c in conds:
            f = st.mean([m["per_component_fraction"][comp] for m in by_condition[c]])
            s = st.mean([m["per_component_strict"][comp] for m in by_condition[c]])
            row += f"{c}: {f:.2f}->{s:.2f}   "
        print(row)


for path, title in [(p, p.split("/")[-1]) for p in sys.argv[1:]]:
    report(load(path), title)
