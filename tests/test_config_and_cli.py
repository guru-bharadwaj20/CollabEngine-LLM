"""Config loading and CLI wiring.

Config round-tripping matters more than it looks: every episode must be
reconstructible from config alone, because Phase 3 replays what Phase 2 recorded
and Phase 4 re-codes it again. A field that silently fails to round-trip is a
reproducibility hole that only shows up months later.
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from collabengine.backends.mock import MockBackend
from collabengine.backends.openai_compat import OpenAICompatBackend
from collabengine.cli import main
from collabengine.config import ExperimentConfig
from collabengine.orchestrator.team import SymmetryBreaking


def test_defaults_build_a_usable_mock_backend() -> None:
    config = ExperimentConfig()
    assert isinstance(config.backend.build(), MockBackend)
    assert list(config.seeds) == list(range(100))


def test_yaml_round_trips_every_field(tmp_path) -> None:
    original = ExperimentConfig.from_dict(
        {
            "name": "exp",
            "seed_start": 5,
            "n_episodes": 7,
            "max_concurrency": 3,
            "backend": {"kind": "openai", "model": "m", "max_concurrency": 12},
            "team": {
                "n_agents": 5,
                "rounds": 4,
                "difficulty": "easy",
                "symmetry": "name_seed_scratch",
                "randomize_turn_order": False,
            },
        }
    )
    path = tmp_path / "c.yaml"
    original.save(path)
    assert ExperimentConfig.load(path).to_dict() == original.to_dict()


def test_openai_backend_is_constructed_from_config() -> None:
    config = ExperimentConfig.from_dict(
        {"backend": {"kind": "openai", "base_url": "http://x/v1", "model": "q"}}
    )
    backend = config.backend.build()
    assert isinstance(backend, OpenAICompatBackend)
    assert backend.model == "q"


def test_unknown_backend_kind_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown backend kind"):
        ExperimentConfig.from_dict({"backend": {"kind": "telepathy"}}).backend.build()


def test_bad_symmetry_value_fails_at_load_not_at_run() -> None:
    """Typos must surface before a multi-hour run starts, not during it."""
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"team": {"symmetry": "nmae_seed"}})


def test_shipped_configs_are_valid(tmp_path) -> None:
    """The example configs must actually load -- they are the documented path in."""
    from pathlib import Path

    for name in ("mock.yaml", "vllm-8b.yaml"):
        path = Path(__file__).parent.parent / "configs" / name
        config = ExperimentConfig.load(path)
        assert config.team.n_agents >= 2
        assert config.n_episodes > 0


def test_randomize_turn_order_defaults_on() -> None:
    """The C2 control should be opt-out, never opt-in."""
    assert ExperimentConfig().team.randomize_turn_order is True


def test_symmetry_default_is_name_seed() -> None:
    assert ExperimentConfig().team.symmetry is SymmetryBreaking.NAME_SEED


def test_cli_selftest_runs_and_reports(capsys) -> None:
    assert main(["selftest", "--episodes", "4"]) == 0
    out = capsys.readouterr().out
    assert "specialized" in out and "positional" in out and "null" in out


def test_cli_ablate_without_baseline_exits_cleanly(tmp_path, capsys) -> None:
    config_path = tmp_path / "c.yaml"
    config_path.write_text(
        textwrap.dedent(
            f"""
            name: nothing
            out_dir: {tmp_path.as_posix()}
            backend:
              kind: mock
            """
        ),
        encoding="utf-8",
    )
    assert main(["ablate", "--config", str(config_path)]) == 2
    assert "run `baseline` first" in capsys.readouterr().err


def test_cli_baseline_then_analyze(tmp_path, capsys) -> None:
    """The documented happy path, end to end, on the mock backend."""
    config_path = tmp_path / "c.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "name": "e2e",
                "n_episodes": 3,
                "out_dir": tmp_path.as_posix(),
                "max_concurrency": 2,
                "backend": {"kind": "mock", "mock_competence": 0.5},
                "team": {"n_agents": 3, "rounds": 2, "difficulty": "easy"},
            }
        ),
        encoding="utf-8",
    )

    assert main(["baseline", "--config", str(config_path)]) == 0
    assert main(["ablate", "--config", str(config_path), "--modes", "live"]) == 0
    assert main(["analyze", "--config", str(config_path)]) == 0

    out = capsys.readouterr().out
    assert "interaction strength" in out
    assert (tmp_path / "e2e" / "config.resolved.yaml").exists()
