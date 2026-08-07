"""Parallel experiment execution.

The experiment is throughput-bound, not latency-bound: roughly 14M output tokens
across the full condition grid. The runner's job is to keep the serving GPU
saturated and never block on any single slow episode.
"""

from collabengine.runner.parallel import (
    RunPlan,
    RunStats,
    completed_episode_ids,
    run_plan,
)

__all__ = ["RunPlan", "RunStats", "completed_episode_ids", "run_plan"]
