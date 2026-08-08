"""Telling a team's failure apart from the token cap's.

The filter these tests cover decides which episodes reach every mean in the
analysis, so the cost of getting it wrong is asymmetric: excluding too much
throws away real failures, and excluding too little lets `max_tokens` write
zeros into the ablation grid. Both directions are tested here.
"""

from __future__ import annotations

import pytest

from collabengine.analysis.integrity import (
    audit,
    finish_reasons,
    is_instrument_failure,
)
from collabengine.protocol import Message, Speaker
from collabengine.tasks.grader import GradeResult
from collabengine.tasks.schema import ALL_COMPONENTS, Solution
from collabengine.transcripts.store import EpisodeRecord


def _msg(turn: int, *, finish: str, speaker: Speaker = Speaker.AGENT) -> Message:
    return Message(
        turn=turn,
        speaker=speaker,
        author="A1" if speaker is Speaker.AGENT else "moderator",
        content="...",
        meta={"finish_reason": finish},
    )


def _record(
    *, finishes: list[str], malformed: bool, condition: str = "baseline"
) -> EpisodeRecord:
    solution = (
        Solution(malformed=True)
        if malformed
        else Solution(assignment={"J1": "W1"}, flagged_errors=[])
    )
    return EpisodeRecord(
        episode_id="e1",
        condition=condition,
        instance_seed=0,
        difficulty="tiny",
        agents=["A1"],
        messages=[_msg(i, finish=f) for i, f in enumerate(finishes)],
        solution=solution,
        grade=GradeResult(
            per_component={c: 0.0 for c in ALL_COMPONENTS},
            overall=0.0,
            satisfied={},
            detail={"malformed": malformed},
        ),
    )


def test_every_turn_truncated_and_no_answer_is_the_instrument() -> None:
    record = _record(finishes=["length", "length", "length"], malformed=True)
    assert is_instrument_failure(record)


def test_a_malformed_answer_after_a_turn_that_finished_is_a_real_failure() -> None:
    """The model had its chance to answer and did not take it.

    Excluding this would quietly drop genuine failures to commit, which is the
    team's behaviour and belongs in the result.
    """
    record = _record(finishes=["length", "stop", "length"], malformed=True)
    assert not is_instrument_failure(record)


def test_truncation_without_a_malformed_answer_is_kept() -> None:
    """Most turns hit the cap. That alone costs nothing if an answer survived."""
    record = _record(finishes=["length", "length"], malformed=False)
    assert not is_instrument_failure(record)


def test_an_episode_with_no_agent_turns_is_not_charged_to_the_cap() -> None:
    record = _record(finishes=[], malformed=True)
    assert not is_instrument_failure(record)


def test_moderator_turns_do_not_count_toward_truncation() -> None:
    """Only agent turns are generated, so only they can hit `max_tokens`.

    A moderator message carries no `finish_reason` from a backend; counting it
    as un-truncated would rescue an episode in which every generated turn was
    in fact cut off.
    """
    record = _record(finishes=["length", "length"], malformed=True)
    record.messages.append(_msg(9, finish="stop", speaker=Speaker.MODERATOR))
    assert is_instrument_failure(record)


def test_audit_separates_malformed_from_instrument_failure() -> None:
    records = [
        _record(finishes=["length", "length"], malformed=True),   # the cap
        _record(finishes=["stop", "stop"], malformed=True),       # a real failure
        _record(finishes=["length", "stop"], malformed=False),    # fine
    ]
    report = audit(records)
    cell = report.by_condition["baseline"]

    assert cell.episodes == 3
    assert cell.malformed == 2
    assert cell.instrument_failures == 1
    assert cell.usable == 2
    assert cell.agent_turns == 6
    assert cell.truncated_turns == 3
    assert report.truncation_rate == pytest.approx(0.5)


def test_audit_keeps_conditions_apart() -> None:
    """A drop is read per cell, so the damage has to be attributable per cell."""
    report = audit(
        [
            _record(finishes=["length"], malformed=True, condition="live:A1"),
            _record(finishes=["stop"], malformed=False, condition="baseline"),
        ]
    )
    assert report.by_condition["live:A1"].instrument_failures == 1
    assert report.by_condition["baseline"].instrument_failures == 0
    assert report.instrument_failures == 1


def test_finish_reasons_tallies_unset_metadata() -> None:
    record = _record(finishes=["length"], malformed=False)
    record.messages.append(Message(5, Speaker.AGENT, "A1", "...", meta={}))
    assert finish_reasons([record]) == {"length": 1, "unset": 1}


def test_report_lines_put_the_worst_cell_first() -> None:
    report = audit(
        [
            _record(finishes=["stop"], malformed=False, condition="baseline"),
            _record(finishes=["length"], malformed=True, condition="live:A2"),
        ]
    )
    lines = report.lines()
    assert "live:A2" in lines[2]
