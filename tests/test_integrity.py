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
    final_turn_truncated,
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


def test_the_answer_bearing_turn_is_the_one_truncation_costs() -> None:
    """Only the last agent turn can cut off the answer being submitted."""
    assert final_turn_truncated(_record(finishes=["stop", "stop", "length"],
                                        malformed=True))
    assert not final_turn_truncated(_record(finishes=["length", "length", "stop"],
                                            malformed=True))


def test_final_turn_truncation_is_flagged_even_when_the_answer_parsed() -> None:
    """It is a diagnostic on the instrument, not a verdict on the episode.

    A cut-off final turn whose answer still parsed is exactly the case that
    makes the flag worth reporting separately: nothing else in the audit would
    show that the cap reached the answer at all.
    """
    record = _record(finishes=["stop", "length"], malformed=False)
    assert final_turn_truncated(record)
    assert not is_instrument_failure(record)


def test_a_moderator_turn_after_the_answer_does_not_hide_truncation() -> None:
    """The last *agent* turn is the answer-bearing one, not the last message."""
    record = _record(finishes=["stop", "length"], malformed=True)
    record.messages.append(_msg(9, finish="stop", speaker=Speaker.MODERATOR))
    assert final_turn_truncated(record)


def test_audit_counts_final_truncation_per_condition() -> None:
    """The column exists because it reads differently across arms: on the
    served `medium` corpus it was 7 of 24 solo episodes and 0 of 24 team."""
    report = audit(
        [
            _record(finishes=["stop", "length"], malformed=True, condition="solo"),
            _record(finishes=["stop", "stop"], malformed=False, condition="solo"),
            _record(finishes=["length", "stop"], malformed=False),
        ]
    )

    assert report.by_condition["solo"].final_truncated == 1
    assert report.by_condition["baseline"].final_truncated == 0
    assert report.final_truncated == 1


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


def test_an_errored_turn_condemns_the_episode_on_its_own() -> None:
    """A batch that OOMed says nothing about the agent whose turn it was.

    Unlike truncation, an errored turn carries no model output, so there is no
    reading under which its zero is the team's. One is enough -- and it must
    count even when the surviving turns still produced a parseable answer,
    because the episode then reflects a team missing a member it should have
    had.
    """
    record = _record(finishes=["error", "stop", "stop"], malformed=False)
    assert is_instrument_failure(record)


def test_an_oom_cascade_is_counted_per_turn_not_just_per_episode() -> None:
    """OOMs arrive in bursts, so the turn count is what shows the scale."""
    records = [_record(finishes=["error", "error", "error"], malformed=True)] * 2
    report = audit(records)

    assert report.instrument_failures == 2
    assert report.errored_turns == 6
    assert report.by_condition["baseline"].usable == 0


# ---------------------------------------------- answer-turn budget (C, 4.10) --


def test_answer_turn_cap_defaults_to_the_shared_cap() -> None:
    """`answer_max_tokens=None` must reproduce every pre-2026-08-12 corpus."""
    from collabengine.orchestrator.team import TeamConfig

    assert TeamConfig(max_tokens=1024).answer_max_tokens is None
    assert TeamConfig.from_dict(TeamConfig(max_tokens=1024).to_dict()).answer_max_tokens is None


def test_answer_turn_cap_survives_a_round_trip() -> None:
    from collabengine.orchestrator.team import TeamConfig

    cfg = TeamConfig(max_tokens=1024, answer_max_tokens=3072)
    assert TeamConfig.from_dict(cfg.to_dict()).answer_max_tokens == 3072
