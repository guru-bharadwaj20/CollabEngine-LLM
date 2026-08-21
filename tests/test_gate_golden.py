"""The headline numbers, recomputed from a committed slice of the corpus.

`README.md` and the paper both claim that the gate is reproducible from the
released episodes. That claim was untestable: the full corpus is a release
asset, so nothing in CI could check that the analysis path still produces the
numbers the documents quote. A reproduction claim that only holds when a 4 GB
download succeeds is a claim nobody verifies, including us.

So a deterministic slice of the real corpus is committed as a fixture, and this
test recomputes the gate from it and compares against a golden file. It does not
prove the *published* numbers -- the slice is far too small for that, and saying
otherwise would be the sample-size error this project spent 599 episodes
learning. It proves the thing that actually breaks silently: that scoring,
integrity filtering, the metric definitions and the equivalence machinery still
turn these episodes into those numbers.

Regenerate the fixture and its golden file with:

    python scripts/analysis/make_gate_fixture.py

and commit both. A change in the golden file is a change in the analysis, and
should be justified in the commit that makes it.
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "gate_slice.jsonl"
GOLDEN = Path(__file__).parent / "fixtures" / "gate_slice.golden.json"

pytest.importorskip("scipy")

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists() or not GOLDEN.exists(),
    reason="gate fixture not generated; see scripts/analysis/make_gate_fixture.py",
)


def _arms():
    from collabengine.analysis.integrity import is_instrument_failure
    from collabengine.analysis.scoring import METRICS, rescore
    from collabengine.transcripts.store import TranscriptReader

    arms: dict[str, dict[str, list[float]]] = {}
    for rec in TranscriptReader(str(FIXTURE)):
        if is_instrument_failure(rec):
            continue
        scored = rescore(rec)
        bucket = arms.setdefault(rec.condition, {m: [] for m in METRICS})
        for metric in METRICS:
            bucket[metric].append(scored.overall[metric])
    return arms


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_the_fixture_holds_the_arms_the_golden_file_describes(golden):
    arms = _arms()
    assert sorted(arms) == sorted(golden["arms"])
    for name, expected in golden["arms"].items():
        assert len(arms[name]["fraction"]) == expected["n"], name


def test_every_arm_mean_matches_the_golden_file(golden):
    """Scoring is deterministic, so this is an exact comparison, not a tolerance.

    A tolerance here would hide exactly the drift it is meant to catch.
    """
    arms = _arms()
    for name, expected in golden["arms"].items():
        for metric, want in expected["mean"].items():
            got = st.mean(arms[name][metric])
            assert got == pytest.approx(want, abs=1e-9), f"{name}.{metric}"


def test_the_gate_gap_matches_the_golden_file(golden):
    arms = _arms()
    for metric, want in golden["gap"].items():
        got = st.mean(arms["baseline"][metric]) - st.mean(arms["solo"][metric])
        assert got == pytest.approx(want, abs=1e-9), metric


def test_the_equivalence_bound_matches_the_golden_file(golden):
    """The number the paper now quotes for every null, held to the same standard."""
    from collabengine.analysis.inference import smallest_equivalence_bound

    arms = _arms()
    got = smallest_equivalence_bound(arms["solo"]["fraction"],
                                     arms["baseline"]["fraction"])
    assert got == pytest.approx(golden["equivalence_bound_fraction"], abs=1e-9)


def test_the_slice_is_read_as_the_family_it_was_generated_from(golden):
    """Guards the defect that a second task family introduced.

    `rescore` once regenerated every instance with the allocation generator no
    matter what the record said.
    """
    from collabengine.transcripts.store import TranscriptReader

    families = {
        (rec.config or {}).get("task", "scheduling")
        for rec in TranscriptReader(str(FIXTURE))
    }
    assert families == {golden["task_family"]}
