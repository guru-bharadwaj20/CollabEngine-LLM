"""The diagnostic, run on transcripts it did not produce.

Two properties are load-bearing and both are tested here in both directions.

The tool must **find** the artifact when it is there: a corpus built to look
like RESEARCH-LOG 4.23 and 4.24 -- one arm writing roughly twice as much, taking
the truncation, and failing to parse -- has to come back flagged, with the cut
and malformed counts attached to the right arm.

The tool must **clear** a corpus when the artifact is not there. Final Sweep 7.1
turns on a clean audit being a reportable result, so "artifact not present" is a
tested output rather than a silence. A diagnostic that only speaks when it finds
something cannot be used to clear anything.

The third thing tested is the degradation rules. A missing field must produce
`n/a` and a named reason, never a zero: reporting an uninstrumented arm as an
untruncated one would make the tool reward the systems it is meant to catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from collabengine.analysis.audit import (
    DEFAULT_RATIO_THRESHOLD,
    SchemaError,
    audit_path,
    audit_records,
    normalize_finish_reason,
    read_records,
    record_from_dict,
)
from collabengine.cli import main


# ---------------------------------------------------------------------------
# fixtures: one foreign corpus with the artifact, one without
# ---------------------------------------------------------------------------


def _turn(chars: int, *, finish: str = "stop", agent: str = "A1") -> dict:
    return {
        "role": "assistant",
        "agent": agent,
        "text": "x" * chars,
        "finish_reason": finish,
        "tokens": chars // 4,
    }


def _episode(arm: str, idx: int, turns: list[dict], *, parsed: bool = True) -> dict:
    return {
        "episode_id": f"{arm}:{idx}",
        "arm": arm,
        "answer_parsed": parsed,
        "turns": turns,
    }


@pytest.fixture
def artifact_corpus() -> list[dict]:
    """A foreign system exhibiting the artifact, shaped like LOG 4.23/4.24.

    Both arms get twelve turns and the same nominal per-turn cap. The solo arm
    restates the whole working solution each round and so generates ~2x the
    characters; that is what makes the shared cap bind on it, truncate its
    answer-bearing turn, and leave a quarter of its answers unparseable. Every
    number here is a property of the fixture, not of any real corpus.
    """
    corpus = []
    for i in range(12):
        cut = i % 4 == 0
        corpus.append(
            _episode(
                "solo",
                i,
                [_turn(2000, finish="stop") for _ in range(11)]
                + [_turn(2000, finish="length" if cut else "stop")],
                parsed=not cut,
            )
        )
    for i in range(12):
        corpus.append(
            _episode(
                "team",
                i,
                [
                    _turn(1000, agent=f"A{1 + (t % 4)}")
                    for t in range(12)
                ],
            )
        )
    return corpus


@pytest.fixture
def clean_corpus() -> list[dict]:
    """The same system after the budget is matched on generation, not turns.

    Nothing else changes: same turn counts, same instrumentation, same fields.
    The arms now generate within a few percent of each other, nothing is
    truncated, and every answer parses.
    """
    corpus = []
    for i in range(12):
        corpus.append(_episode("solo", i, [_turn(1020) for _ in range(12)]))
    for i in range(12):
        corpus.append(
            _episode(
                "team",
                i,
                [_turn(1000, agent=f"A{1 + (t % 4)}") for t in range(12)],
            )
        )
    return corpus


def _audit(corpus: list[dict], **kw):
    return audit_records(
        ((r, f"fixture[{i}]") for i, r in enumerate(corpus)), **kw
    )


# ---------------------------------------------------------------------------
# the artifact is found
# ---------------------------------------------------------------------------


def test_the_verbose_arm_is_identified_and_flagged(artifact_corpus) -> None:
    report = _audit(artifact_corpus)
    assert report.records_read == 24
    assert report.verbose_arm == "solo"
    assert report.spare_arm == "team"
    assert report.verbosity_ratio == pytest.approx(2.0)
    assert report.flagged


def test_the_flag_respects_the_threshold_it_is_given(artifact_corpus) -> None:
    # A 2.0x ratio is above the default and below an explicit 2.5x. The
    # threshold is an argument precisely so that a reader can see the ratio and
    # decide, rather than inheriting this project's 1.5 as a standard.
    assert _audit(artifact_corpus, threshold=DEFAULT_RATIO_THRESHOLD).flagged
    assert not _audit(artifact_corpus, threshold=2.5).flagged


def test_truncation_and_answer_turn_cuts_land_on_the_verbose_arm(
    artifact_corpus,
) -> None:
    report = _audit(artifact_corpus)
    solo = report.integrity.by_condition["solo"]
    team = report.integrity.by_condition["team"]
    assert solo.truncated_turns == 3
    assert solo.final_truncated == 3
    assert team.truncated_turns == 0
    assert team.final_truncated == 0


def test_unparseable_answers_are_counted_per_arm(artifact_corpus) -> None:
    """The fourth artifact (RESEARCH-LOG 4.24), on a foreign corpus.

    25 of 149 against 0 of 150 was enough to move a mean by more than the whole
    measured gap, so the count has to survive the adapter with its arm attached.
    """
    report = _audit(artifact_corpus)
    assert report.integrity.by_condition["solo"].malformed == 3
    assert report.integrity.by_condition["team"].malformed == 0
    assert report.malformed_asymmetry() == ("solo", "team")


def test_the_report_names_the_arm_and_the_mechanism(artifact_corpus) -> None:
    text = "\n".join(_audit(artifact_corpus).lines())
    assert "FLAG" in text
    assert "2.00x" in text
    assert "solo" in text
    assert "n/a" not in text


# ---------------------------------------------------------------------------
# the artifact is absent, and the tool says so
# ---------------------------------------------------------------------------


def test_a_matched_corpus_is_not_flagged(clean_corpus) -> None:
    report = _audit(clean_corpus)
    assert report.verbosity_ratio == pytest.approx(1.02)
    assert not report.flagged


def test_a_clean_audit_states_the_absence_rather_than_going_quiet(
    clean_corpus,
) -> None:
    """Final Sweep 7.1: "the artifact is not present here" is a contribution."""
    text = "\n".join(_audit(clean_corpus).lines())
    assert "artifact not present" in text
    assert "FLAG" not in text
    assert "answer-turn cuts: none in any arm." in text
    assert "the same rate in every arm" in text


def test_a_clean_audit_is_still_qualified_by_its_sensitivity(clean_corpus) -> None:
    # The claim is bounded by the threshold, not absolute. A tool that reported
    # "no artifact" full stop would overstate what a ratio under 1.5 licenses.
    text = "\n".join(_audit(clean_corpus).lines())
    assert "at this sensitivity" in text
    assert f"{DEFAULT_RATIO_THRESHOLD:.2f}x" in text


# ---------------------------------------------------------------------------
# degradation: an absent field is never read as a zero
# ---------------------------------------------------------------------------


def test_no_finish_reason_means_truncation_is_not_computable(clean_corpus) -> None:
    for record in clean_corpus:
        for turn in record["turns"]:
            turn.pop("finish_reason")
    report = _audit(clean_corpus)
    assert not report.truncation_computable
    assert report.to_dict()["arms"]["solo"]["truncated_turns"] is None
    text = "\n".join(report.lines())
    assert "not computable -- truncation" in text
    assert "not evidence of no truncation" in text
    # The ratio does not depend on the field and must survive its absence.
    assert report.verbosity_ratio == pytest.approx(1.02)


def test_one_turn_missing_the_field_disqualifies_the_whole_arm(
    clean_corpus,
) -> None:
    """All-or-nothing per arm, because the gaps are not random.

    A wrapper layer that drops `finish_reason` drops it for whatever it wraps,
    and what it wraps correlates with arm. Tallying over the instrumented subset
    would report the least instrumented arm as the cleanest one.
    """
    clean_corpus[0]["turns"][0].pop("finish_reason")
    report = _audit(clean_corpus)
    assert not report.by_arm["solo"].truncation_known
    assert report.by_arm["team"].truncation_known


def test_no_token_counts_leave_the_character_ratio_intact(clean_corpus) -> None:
    for record in clean_corpus:
        for turn in record["turns"]:
            turn.pop("tokens")
    report = _audit(clean_corpus)
    assert report.by_arm["solo"].tokens_per_episode is None
    assert report.verbosity_ratio == pytest.approx(1.02)
    assert "not computable -- tokens per episode" in "\n".join(report.lines())


def test_no_parse_status_means_the_fourth_artifact_cannot_be_checked(
    artifact_corpus,
) -> None:
    for record in artifact_corpus:
        record.pop("answer_parsed")
    report = _audit(artifact_corpus)
    assert not report.parse_status_computable
    assert report.malformed_asymmetry() is None
    assert report.integrity.by_condition["solo"].malformed == 0
    text = "\n".join(report.lines())
    assert "RESEARCH-LOG 4.24" in text
    assert "not computable -- unparseable-answer counts" in text


def test_a_single_arm_yields_no_ratio_and_says_why(clean_corpus) -> None:
    solo = [r for r in clean_corpus if r["arm"] == "solo"]
    report = _audit(solo)
    assert report.verbosity_ratio is None
    assert not report.flagged
    assert "not computable" in "\n".join(report.lines())


# ---------------------------------------------------------------------------
# the adapter
# ---------------------------------------------------------------------------


def test_prompt_rows_are_excluded_from_generation_counts() -> None:
    """Counting the prompt would bias toward the arm with the longer scaffold.

    On a team-vs-solo comparison that is systematically the team, i.e. the
    opposite direction to the artifact -- so it would quietly clear exactly the
    corpora worth flagging.
    """
    record = record_from_dict(
        {
            "arm": "solo",
            "turns": [
                {"role": "system", "text": "y" * 5000},
                {"role": "user", "text": "y" * 5000},
                {"role": "assistant", "text": "x" * 10},
            ],
        }
    )
    assert len(record.messages) == 1
    assert record.messages[0].content == "x" * 10


def test_a_turn_with_no_role_counts_as_generated() -> None:
    record = record_from_dict({"arm": "solo", "turns": [{"text": "abc"}]})
    assert len(record.messages) == 1


def test_field_aliases_are_accepted() -> None:
    record = record_from_dict(
        {
            "condition": "team",
            "messages": [
                {"content": "abc", "stop_reason": "MAX_TOKENS", "output_tokens": 7}
            ],
            "malformed": True,
        }
    )
    assert record.condition == "team"
    assert record.solution.malformed is True
    assert record.messages[0].meta == {"finish_reason": "length", "tokens": 7}


def test_finish_reason_spellings_normalize_or_pass_through() -> None:
    assert normalize_finish_reason("max_output_tokens") == "length"
    assert normalize_finish_reason("Exception") == "error"
    assert normalize_finish_reason("end_turn") == "end_turn"
    assert normalize_finish_reason(None) is None
    assert normalize_finish_reason("") is None


def test_malformed_wins_over_answer_parsed_when_both_are_present() -> None:
    record = record_from_dict(
        {"arm": "solo", "turns": [{"text": "a"}], "malformed": False,
         "answer_parsed": False}
    )
    assert record.solution.malformed is False


def test_records_missing_a_required_field_are_named_not_guessed() -> None:
    with pytest.raises(SchemaError, match="`arm`"):
        record_from_dict({"turns": []}, where="corpus.jsonl:4")
    with pytest.raises(SchemaError, match="`turns`"):
        record_from_dict({"arm": "solo"}, where="corpus.jsonl:5")
    with pytest.raises(SchemaError, match="corpus.jsonl:6 turn 0"):
        record_from_dict(
            {"arm": "solo", "turns": [{"role": "assistant"}]}, where="corpus.jsonl:6"
        )


def test_bad_records_are_skipped_and_counted_unless_strict() -> None:
    raw = [({"arm": "solo", "turns": [{"text": "a"}]}, "f:1"), ({"turns": []}, "f:2")]
    report = audit_records(iter(raw))
    assert report.records_read == 1
    assert report.records_skipped == 1
    assert "`arm`" in report.skip_reasons[0]
    assert "did not meet the schema" in "\n".join(report.lines())
    with pytest.raises(SchemaError):
        audit_records(iter(raw), strict=True)


# ---------------------------------------------------------------------------
# reading, and the CLI
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, corpus: list[dict]) -> Path:
    path.write_text(
        "\n".join(json.dumps(r) for r in corpus) + "\n", encoding="utf-8"
    )
    return path


def test_jsonl_json_and_directory_all_read_the_same_corpus(
    tmp_path: Path, clean_corpus
) -> None:
    (tmp_path / "flat").mkdir()
    _write_jsonl(tmp_path / "flat" / "corpus.jsonl", clean_corpus)
    (tmp_path / "corpus.json").write_text(
        json.dumps({"episodes": clean_corpus}), encoding="utf-8"
    )
    from_jsonl = audit_path(tmp_path / "flat" / "corpus.jsonl")
    from_json = audit_path(tmp_path / "corpus.json")
    from_dir = audit_path(tmp_path / "flat")
    assert from_jsonl.records_read == from_json.records_read == 24
    assert from_dir.records_read == 24
    assert from_jsonl.verbosity_ratio == pytest.approx(from_json.verbosity_ratio)
    assert from_dir.verbosity_ratio == pytest.approx(from_jsonl.verbosity_ratio)


def test_a_missing_path_is_an_error_not_an_empty_report(tmp_path: Path) -> None:
    with pytest.raises(SchemaError, match="no such file"):
        audit_path(tmp_path / "nope.jsonl")
    (tmp_path / "empty").mkdir()
    with pytest.raises(SchemaError, match="no .json or .jsonl"):
        audit_path(tmp_path / "empty")


def test_cli_prints_the_table_and_exits_zero_on_a_flagged_corpus(
    tmp_path: Path, artifact_corpus, capsys
) -> None:
    """A flagged corpus exits 0.

    The exit code reports whether the audit ran, not what it found. Making a
    finding an error would give anyone auditing their own system a reason not to
    run it in CI, which is the opposite of what the tool is for.
    """
    path = _write_jsonl(tmp_path / "corpus.jsonl", artifact_corpus)
    assert main(["audit", str(path)]) == 0
    out = capsys.readouterr().out
    assert "FLAG" in out and "chars/ep" in out


def test_cli_exits_zero_on_a_clean_corpus_too(
    tmp_path: Path, clean_corpus, capsys
) -> None:
    path = _write_jsonl(tmp_path / "corpus.jsonl", clean_corpus)
    assert main(["audit", str(path)]) == 0
    assert "artifact not present" in capsys.readouterr().out


def test_cli_json_output_carries_every_reported_quantity(
    tmp_path: Path, artifact_corpus, capsys
) -> None:
    path = _write_jsonl(tmp_path / "corpus.jsonl", artifact_corpus)
    assert main(["audit", str(path), "--json"]) == 0
    got = json.loads(capsys.readouterr().out)
    assert got["flagged"] is True
    assert got["verbose_arm"] == "solo"
    assert got["verbosity_ratio"] == pytest.approx(2.0)
    assert got["arms"]["solo"]["answer_turn_cut"] == 3
    assert got["arms"]["team"]["malformed"] == 0


def test_cli_threshold_flag_changes_the_verdict(
    tmp_path: Path, artifact_corpus, capsys
) -> None:
    path = _write_jsonl(tmp_path / "corpus.jsonl", artifact_corpus)
    assert main(["audit", str(path), "--threshold", "2.5", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["flagged"] is False


def test_cli_reports_an_unreadable_corpus_as_an_error(
    tmp_path: Path, capsys
) -> None:
    assert main(["audit", str(tmp_path / "nope.jsonl")]) == 2
    assert "AUDIT-SCHEMA.md" in capsys.readouterr().err


def test_cli_reports_a_corpus_with_no_usable_records_as_an_error(
    tmp_path: Path, capsys
) -> None:
    path = _write_jsonl(tmp_path / "corpus.jsonl", [{"turns": []}, {"turns": []}])
    assert main(["audit", str(path)]) == 2
    assert "no records read" in capsys.readouterr().err


def test_cli_strict_refuses_a_corpus_it_would_otherwise_partly_skip(
    tmp_path: Path, clean_corpus, capsys
) -> None:
    """Without --strict the bad record is a footnote; with it, the run fails.

    Both are right for their case. An auditor reading someone else's release
    wants the 24 good records; someone writing the converter wants to be stopped
    at the first record it got wrong, before a silent skip rate eats an arm.
    """
    path = _write_jsonl(tmp_path / "corpus.jsonl", clean_corpus + [{"turns": []}])
    assert main(["audit", str(path)]) == 0
    assert "did not meet the schema" in capsys.readouterr().out
    assert main(["audit", str(path), "--strict"]) == 2
    assert "corpus.jsonl:25" in capsys.readouterr().err


def test_read_records_reports_where_each_record_came_from(
    tmp_path: Path, clean_corpus
) -> None:
    # The location is the whole value of the skip message: a converter that got
    # one field name wrong is fixed by looking at the record it failed on.
    path = _write_jsonl(tmp_path / "corpus.jsonl", clean_corpus[:2])
    where = [w for _, w in read_records(path)]
    assert where == ["corpus.jsonl:1", "corpus.jsonl:2"]
