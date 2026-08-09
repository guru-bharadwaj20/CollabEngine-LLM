"""Experiment configuration.

A run is fully specified by a YAML file plus a seed range. That is deliberate:
every episode must be reconstructible from config alone, because Phase 3 replays
episodes recorded in Phase 2 and Phase 4 re-codes them again. Anything that
influences generation and is not captured here is a reproducibility hole.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from collabengine.backends.base import LLMBackend
from collabengine.backends.mock import MockBackend, MockMode
from collabengine.backends.openai_compat import OpenAICompatBackend
from collabengine.orchestrator.team import SymmetryBreaking, TeamConfig


@dataclass(slots=True)
class BackendConfig:
    kind: str = "mock"
    """"mock", "hf" (in-process CUDA), "openai" (vLLM / llama.cpp server), or
    "anthropic" (frontier judge -- coding only, never an agent)."""
    model: str = "Qwen/Qwen3-8B"
    base_url: str = "http://localhost:8000/v1"
    api_key: str | None = None
    max_concurrency: int = 64
    timeout_s: float = 600.0
    max_retries: int = 4
    mock_mode: str = "specialized"
    mock_competence: float = 0.5
    mock_off_focus: float = 0.0

    # -- hf backend. Recorded even when unused, because batch size changes what
    # -- the sampler produces and so belongs in the reproducibility record.
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_batch_size: int = 16
    max_batch_tokens: int = 32768
    batch_window_s: float = 0.05
    max_model_len: int = 8192
    enable_thinking: bool = False
    trust_remote_code: bool = False
    memory_fraction: float = 0.85
    """Fraction of the card the process may allocate. See `hf_local`: above
    this Windows pages rather than raising OOM, and a paging run looks
    healthy everywhere except PCIe traffic."""
    oom_retries: int = 4
    """Waits before a single sequence that OOMs is recorded as an error.

    Set above zero because this study's card is shared: every OOM recorded here
    has been transient foreign memory pressure rather than a real ceiling, and
    without retries that writes empty turns into the corpus."""
    oom_retry_s: float = 8.0

    def build(self) -> LLMBackend:
        if self.kind == "mock":
            return MockBackend(
                mode=MockMode(self.mock_mode),
                competence=self.mock_competence,
                off_focus_competence=self.mock_off_focus,
            )
        if self.kind == "hf":
            # Imported here so that parsing a mock config never pulls in torch.
            from collabengine.backends.hf_local import HFLocalBackend

            return HFLocalBackend(
                model_id=self.model,
                device=self.device,
                dtype=self.dtype,
                max_batch_size=self.max_batch_size,
                max_batch_tokens=self.max_batch_tokens,
                batch_window_s=self.batch_window_s,
                max_model_len=self.max_model_len,
                enable_thinking=self.enable_thinking,
                trust_remote_code=self.trust_remote_code,
                memory_fraction=self.memory_fraction,
                oom_retries=self.oom_retries,
                oom_retry_s=self.oom_retry_s,
            )
        if self.kind == "gemini":
            from collabengine.backends.gemini_judge import GeminiJudgeBackend

            return GeminiJudgeBackend(
                model=self.model,
                api_key=self.api_key,
                timeout_s=self.timeout_s,
                max_retries=self.max_retries,
            )
        if self.kind == "anthropic":
            from collabengine.backends.anthropic_judge import AnthropicJudgeBackend

            return AnthropicJudgeBackend(
                model=self.model,
                api_key=self.api_key,
                max_concurrency=self.max_concurrency,
                max_retries=self.max_retries,
                timeout_s=self.timeout_s,
            )
        if self.kind == "openai":
            return OpenAICompatBackend(
                base_url=self.base_url,
                model=self.model,
                api_key=self.api_key,
                max_concurrency=self.max_concurrency,
                timeout_s=self.timeout_s,
                max_retries=self.max_retries,
            )
        raise ValueError(
            f"unknown backend kind {self.kind!r}; "
            "expected mock|hf|openai|anthropic|gemini"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "model": self.model,
            "base_url": self.base_url,
            "max_concurrency": self.max_concurrency,
            "timeout_s": self.timeout_s,
            "max_retries": self.max_retries,
            "mock_mode": self.mock_mode,
            "mock_competence": self.mock_competence,
            "mock_off_focus": self.mock_off_focus,
            "device": self.device,
            "dtype": self.dtype,
            "max_batch_size": self.max_batch_size,
            "max_batch_tokens": self.max_batch_tokens,
            "batch_window_s": self.batch_window_s,
            "max_model_len": self.max_model_len,
            "enable_thinking": self.enable_thinking,
            "memory_fraction": self.memory_fraction,
            "oom_retries": self.oom_retries,
            "oom_retry_s": self.oom_retry_s,
            "trust_remote_code": self.trust_remote_code,
        }


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "default"
    seed_start: int = 0
    n_episodes: int = 100
    out_dir: str = "runs"
    max_concurrency: int = 32
    backend: BackendConfig = field(default_factory=BackendConfig)
    team: TeamConfig = field(default_factory=TeamConfig)

    @property
    def seeds(self) -> range:
        return range(self.seed_start, self.seed_start + self.n_episodes)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        backend = BackendConfig(**(d.get("backend") or {}))

        team_raw = dict(d.get("team") or {})
        if "symmetry" in team_raw:
            SymmetryBreaking(team_raw["symmetry"])  # fail fast on a typo
        team = TeamConfig.from_dict(team_raw)

        return cls(
            name=d.get("name", "default"),
            seed_start=int(d.get("seed_start", 0)),
            n_episodes=int(d.get("n_episodes", 100)),
            out_dir=d.get("out_dir", "runs"),
            max_concurrency=int(d.get("max_concurrency", 32)),
            backend=backend,
            team=team,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed_start": self.seed_start,
            "n_episodes": self.n_episodes,
            "out_dir": self.out_dir,
            "max_concurrency": self.max_concurrency,
            "backend": self.backend.to_dict(),
            "team": self.team.to_dict(),
        }

    def save(self, path: str | Path) -> None:
        """Write the resolved config beside the transcript.

        Saving the *resolved* config rather than trusting the input file means a
        transcript can be interpreted months later even if the source YAML has
        drifted.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )
