"""Prompt rendering and answer parsing.

The leak tests are the important ones. If `render_instance` ever reads
`ground_truth` or `planted_errors`, the verification component becomes free and
every score in the study is inflated -- so we assert independence structurally
rather than by grepping for suspicious substrings.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from collabengine.tasks.generator import PRESETS, generate
from collabengine.tasks.render import (
    ANSWER_CLOSE,
    ANSWER_OPEN,
    parse_solution,
    render_instance,
)

DIFFICULTIES = sorted(PRESETS)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_render_ignores_ground_truth(difficulty: str) -> None:
    """Scrambling the hidden solution must not change one byte of the prompt."""
    inst = generate(13, difficulty)
    scrambled = dataclasses.replace(
        inst,
        ground_truth={k: "W_SCRAMBLED" for k in inst.ground_truth},
    )
    assert render_instance(inst) == render_instance(scrambled)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_render_ignores_planted_errors(difficulty: str) -> None:
    """Which draft entries are wrong must be inferable only from the constraints."""
    inst = generate(17, difficulty)
    hidden = dataclasses.replace(inst, planted_errors=())
    assert render_instance(inst) == render_instance(hidden)


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_render_mentions_every_job_and_worker(difficulty: str) -> None:
    inst = generate(21, difficulty)
    text = render_instance(inst)
    for j in inst.jobs:
        assert j.jid in text
    for w in inst.workers:
        assert w.wid in text


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_render_uses_no_role_language(difficulty: str) -> None:
    """No hint that division of labor is expected -- that is the thing under study."""
    text = render_instance(generate(23, difficulty)).lower()
    for banned in (
        "role",
        "specialist",
        "divide",
        "split the work",
        "each of you",
        "coordinator",
        "leader",
    ):
        assert banned not in text, f"prompt plants structure via {banned!r}"


def test_parse_tagged_answer() -> None:
    text = f'blah blah {ANSWER_OPEN}{{"assignment": {{"J1": "W2"}}, "flagged_errors": ["J3"]}}{ANSWER_CLOSE} trailing'
    sol = parse_solution(text)
    assert not sol.malformed
    assert sol.assignment == {"J1": "W2"}
    assert sol.flagged_errors == ["J3"]


def test_parse_fenced_json() -> None:
    text = 'Here is my answer:\n```json\n{"assignment": {"J1": "W1"}, "flagged_errors": []}\n```\n'
    sol = parse_solution(text)
    assert sol.assignment == {"J1": "W1"}
    assert sol.flagged_errors == []


def test_parse_bare_json_in_prose() -> None:
    text = 'I think {"assignment": {"J5": "W3"}, "flagged_errors": ["J1"]} is right.'
    assert parse_solution(text).assignment == {"J5": "W3"}


def test_tagged_answer_beats_earlier_draft() -> None:
    """A later revision supersedes an earlier one in the same turn."""
    text = (
        'First draft: {"assignment": {"J1": "W1"}, "flagged_errors": []}\n'
        "On reflection that breaks the capacity limit.\n"
        f'{ANSWER_OPEN}{{"assignment": {{"J1": "W2"}}, "flagged_errors": ["J1"]}}{ANSWER_CLOSE}'
    )
    sol = parse_solution(text)
    assert sol.assignment == {"J1": "W2"}
    assert sol.flagged_errors == ["J1"]


def test_last_bare_object_wins() -> None:
    text = (
        '{"assignment": {"J1": "W1"}, "flagged_errors": []} '
        'then actually {"assignment": {"J1": "W9"}, "flagged_errors": ["J2"]}'
    )
    assert parse_solution(text).assignment == {"J1": "W9"}


def test_braces_inside_strings_do_not_break_parsing() -> None:
    text = '{"assignment": {"J1": "W1"}, "flagged_errors": [], "note": "use {curly} braces"}'
    assert parse_solution(text).assignment == {"J1": "W1"}


def test_prose_only_is_malformed() -> None:
    assert parse_solution("I am not sure how to solve this.").malformed


def test_json_without_assignment_key_is_malformed() -> None:
    assert parse_solution('{"flagged_errors": ["J1"]}').malformed


def test_non_string_values_are_dropped_not_fatal() -> None:
    text = '{"assignment": {"J1": "W1", "J2": 7, "J3": null}, "flagged_errors": ["J1", 2]}'
    sol = parse_solution(text)
    assert sol.assignment == {"J1": "W1"}
    assert sol.flagged_errors == ["J1", "2"]


def test_flagged_errors_accepts_comma_string() -> None:
    """Small models frequently emit a string where a list was specified."""
    text = '{"assignment": {"J1": "W1"}, "flagged_errors": "J2, J3"}'
    assert parse_solution(text).flagged_errors == ["J2", "J3"]


def test_round_trip_through_json_dumps() -> None:
    payload = {"assignment": {"J1": "W2", "J2": "W1"}, "flagged_errors": ["J2"]}
    sol = parse_solution(f"{ANSWER_OPEN}{json.dumps(payload)}{ANSWER_CLOSE}")
    assert sol.assignment == payload["assignment"]
    assert sol.flagged_errors == payload["flagged_errors"]
