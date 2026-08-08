"""The Phase 2-4 commands, end to end on the mock backend.

`pipeline` exists to keep a GPU busy, so the property worth testing is not that
it produces a particular number but that it produces the *whole grid* in one
invocation, resumes without redoing work, and hands each stage what the next one
needs. Those are the failures that cost hours of card time rather than seconds.
"""

from __future__ import annotations

import json

import pytest
import yaml

from collabengine.analysis.coding import ActionType, MessageCode
from collabengine.cli import _ALL_MODES, main
from collabengine.transcripts.store import TranscriptReader


@pytest.fixture()
def run(tmp_path):
    """A tiny mock experiment; returns (config_path, run_dir)."""
    config_path = tmp_path / "c.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "grid",
                "n_episodes": 2,
                "out_dir": tmp_path.as_posix(),
                "max_concurrency": 4,
                "backend": {"kind": "mock", "mock_competence": 0.5},
                "team": {"n_agents": 3, "rounds": 2, "difficulty": "tiny"},
            }
        ),
        encoding="utf-8",
    )
    return config_path, tmp_path / "grid"


def _conditions(path):
    return {r.condition for r in TranscriptReader(path)}


def test_pipeline_produces_every_phase_in_one_invocation(run, capsys) -> None:
    config_path, run_dir = run
    assert main(["pipeline", "--config", str(config_path)]) == 0

    baseline = _conditions(run_dir / "baseline.jsonl")
    assert "baseline" in baseline
    assert "solo" in baseline  # the Phase 1 gate, run at the operating point
    assert "fixed_order" in baseline  # the C2 control
    assert any(c.startswith("symmetry:") for c in baseline)  # the C3 sweep

    ablation = _conditions(run_dir / "ablation.jsonl")
    for mode in ("live:", "frozen_replay:", "frozen_excise:", "random_message:"):
        assert any(c.startswith(mode) for c in ablation), mode
    assert any(c.startswith("capacity:") for c in ablation)


def test_ablation_ignores_the_sweep_conditions(run) -> None:
    """The grid is defined over baseline episodes.

    Symmetry and fixed-order episodes share the transcript file, and ablating
    them too would quietly multiply the grid and mix conditions into one matrix.
    """
    config_path, run_dir = run
    main(["pipeline", "--config", str(config_path)])

    baselines = [
        r for r in TranscriptReader(run_dir / "baseline.jsonl")
        if r.condition == "baseline"
    ]
    live = [
        r for r in TranscriptReader(run_dir / "ablation.jsonl")
        if r.condition.startswith("live:")
    ]
    # one live ablation per agent per baseline episode, and no more
    assert len(live) == len(baselines) * 3


def test_the_ablation_reference_uses_only_baseline_episodes(run) -> None:
    """Every drop is measured against this reference.

    Stage 1 writes baseline, solo, symmetry and fixed-order episodes to one
    transcript. Averaging across all of them would fold a one-agent team's
    score into the number four-agent ablations are compared against, shrinking
    every drop toward zero -- and, where solo scores low enough, past it.
    """
    from collabengine.cli import _component_means

    config_path, run_dir = run
    main(["pipeline", "--config", str(config_path), "--phases", "baseline,solo"])

    path = run_dir / "baseline.jsonl"
    filtered = _component_means(TranscriptReader(path))
    unfiltered = _component_means(TranscriptReader(path), condition=None)
    baseline_only = _component_means(TranscriptReader(path), condition="baseline")

    assert filtered == baseline_only
    # Solo teams score differently, so mixing them in moves the reference.
    assert filtered != unfiltered


def test_pipeline_resumes_without_redoing_work(run, capsys) -> None:
    """A second run must cost nothing -- this is what makes a crash survivable."""
    config_path, _ = run
    main(["pipeline", "--config", str(config_path)])
    capsys.readouterr()

    assert main(["pipeline", "--config", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "0 done" in out


def test_every_plan_id_matches_the_record_it_produces(run) -> None:
    """Resume compares plan ids to recorded ids, so a mismatch costs the run.

    A plan whose id never appears in the transcript is never skipped, so a
    restart silently re-runs work already on disk -- hours of card time for the
    ablation grid, with no error to notice. Checking the two agree for every
    mode is the only thing that keeps this fixed.
    """
    config_path, run_dir = run
    main(["pipeline", "--config", str(config_path)])

    from collabengine.cli import (
        _ablation_plans,
        _baseline_plans,
        _fixed_order_plans,
        _solo_plans,
        _symmetry_plans,
    )
    from collabengine.config import ExperimentConfig

    config = ExperimentConfig.load(config_path)
    backend = config.backend.build()
    baselines = [
        r for r in TranscriptReader(run_dir / "baseline.jsonl")
        if r.condition == "baseline"
    ]

    planned = {p.episode_id for p in _baseline_plans(config, backend)}
    planned |= {p.episode_id for p in _solo_plans(config, backend)}
    planned |= {p.episode_id for p in _symmetry_plans(config, backend)}
    planned |= {p.episode_id for p in _fixed_order_plans(config, backend)}
    planned |= {
        p.episode_id
        for p in _ablation_plans(config, backend, baselines, set(_ALL_MODES))
    }

    recorded = {r.episode_id for r in TranscriptReader(run_dir / "baseline.jsonl")}
    recorded |= {r.episode_id for r in TranscriptReader(run_dir / "ablation.jsonl")}

    assert planned == recorded


def test_pipeline_phase_subset_runs_only_what_was_asked(run) -> None:
    config_path, run_dir = run
    assert main(["pipeline", "--config", str(config_path), "--phases", "baseline"]) == 0

    assert _conditions(run_dir / "baseline.jsonl") == {"baseline"}
    assert not (run_dir / "ablation.jsonl").exists()


def test_ablate_without_baseline_in_the_phase_list_exits_cleanly(run, capsys) -> None:
    config_path, _ = run
    assert main(["pipeline", "--config", str(config_path), "--phases", "ablate"]) == 2
    assert "no baseline" in capsys.readouterr().err


def test_code_writes_labels_and_reports_against_the_null(run, capsys) -> None:
    config_path, run_dir = run
    main(["pipeline", "--config", str(config_path), "--phases", "baseline"])

    assert main(["code", "--config", str(config_path), "--judge", "mock"]) == 0

    out_path = run_dir / "codes.mock.jsonl"
    codes = [
        MessageCode.from_dict(json.loads(line))
        for line in out_path.read_text(encoding="utf-8").splitlines()
    ]
    assert codes
    assert all(isinstance(c.action, ActionType) for c in codes)

    out = capsys.readouterr().out
    assert "differentiation" in out


def test_code_refuses_the_frontier_judge_without_a_key(run, capsys, monkeypatch) -> None:
    """Fail before spending anything, and say what the alternative is."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config_path, _ = run
    main(["pipeline", "--config", str(config_path), "--phases", "baseline"])

    assert main(["code", "--config", str(config_path), "--judge", "anthropic"]) == 2
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err
    assert "--judge self" in err


def test_kappa_compares_two_judges_on_shared_messages(tmp_path, capsys) -> None:
    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"

    def write(path, actions):
        path.write_text(
            "\n".join(
                json.dumps(
                    MessageCode("e1", turn, f"A{turn}", action, "j").to_dict()
                )
                for turn, action in enumerate(actions)
            ),
            encoding="utf-8",
        )

    labels = [ActionType.COMPUTE, ActionType.VERIFY, ActionType.SEARCH] * 4
    write(first, labels)
    write(second, labels)

    assert main(["kappa", str(first), str(second)]) == 0
    out = capsys.readouterr().out
    assert "Cohen's kappa:     1.000" in out


def test_kappa_reports_when_the_files_share_nothing(tmp_path, capsys) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    a.write_text(
        json.dumps(MessageCode("e1", 1, "A1", ActionType.COMPUTE).to_dict()),
        encoding="utf-8",
    )
    b.write_text(
        json.dumps(MessageCode("e9", 9, "A1", ActionType.COMPUTE).to_dict()),
        encoding="utf-8",
    )

    assert main(["kappa", str(a), str(b)]) == 2
    assert "share no coded messages" in capsys.readouterr().err


def test_converge_reports_a_correlation_and_a_verdict(run, capsys) -> None:
    config_path, run_dir = run
    main(["pipeline", "--config", str(config_path)])
    main(["code", "--config", str(config_path), "--judge", "mock"])

    codes = run_dir / "codes.mock.jsonl"
    assert main(["converge", "--config", str(config_path), "--codes", str(codes),
                 "--permutations", "50"]) == 0

    out = capsys.readouterr().out
    assert "r = " in out
    assert "transcript labels" in out
