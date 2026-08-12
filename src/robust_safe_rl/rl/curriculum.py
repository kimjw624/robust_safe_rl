"""Editable fixed-step curriculum support for residual-SAC training.

The curriculum is intentionally data-driven: stage definitions live in a TOML
file so disturbance ranges and durations can be edited without touching Python.

Anti-forgetting is handled in two complementary ways by the training script:
  * the SAC agent/optimizers/target networks and replay buffer are never reset at
    stage boundaries;
  * a configurable fraction of new episodes rehearse previously seen stages.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    timesteps: int
    env_overrides: dict


@dataclass(frozen=True)
class Curriculum:
    stages: tuple[CurriculumStage, ...]
    rehearsal_probability: float = 0.25
    evaluation_episodes_per_stage: int = 3
    rehearsal_mode: str = "uniform_previous"

    @property
    def total_timesteps(self) -> int:
        return sum(stage.timesteps for stage in self.stages)

    def stage_index(self, step: int) -> int:
        """Scheduled stage index for a 1-based global environment step."""
        if not self.stages:
            raise ValueError("curriculum has no stages")
        step = max(1, int(step))
        end = 0
        for i, stage in enumerate(self.stages):
            end += stage.timesteps
            if step <= end:
                return i
        return len(self.stages) - 1

    def sample_episode_stage(self, scheduled_index: int, rng) -> int:
        """Choose current-stage or rehearsal-stage index for a new episode.

        Stage 0 has nothing to rehearse. From stage 1 onward, a fraction of
        episodes is sampled uniformly from all previously seen stages. This
        keeps old disturbance families in the on-policy data stream even after
        the main curriculum has advanced.
        """
        i = int(scheduled_index)
        if i <= 0 or self.rehearsal_probability <= 0.0:
            return i
        if rng.random() >= self.rehearsal_probability:
            return i
        if self.rehearsal_mode != "uniform_previous":
            raise ValueError(f"unsupported rehearsal_mode: {self.rehearsal_mode!r}")
        return int(rng.integers(0, i))


def load_curriculum(path: str | Path) -> Curriculum:
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)

    rehearsal_probability = float(raw.get("rehearsal_probability", 0.25))
    if not 0.0 <= rehearsal_probability <= 1.0:
        raise ValueError("rehearsal_probability must lie in [0, 1]")

    eval_eps = int(raw.get("evaluation_episodes_per_stage", 3))
    if eval_eps < 1:
        raise ValueError("evaluation_episodes_per_stage must be >= 1")

    rehearsal_mode = str(raw.get("rehearsal_mode", "uniform_previous"))
    if rehearsal_mode != "uniform_previous":
        raise ValueError("only rehearsal_mode='uniform_previous' is currently supported")

    raw_stages = raw.get("stages", [])
    if not raw_stages:
        raise ValueError("curriculum TOML must define at least one [[stages]] entry")

    stages = []
    seen_names = set()
    for i, item in enumerate(raw_stages):
        item = dict(item)
        name = str(item.pop("name", f"stage_{i + 1:02d}"))
        timesteps = int(item.pop("timesteps", 0))
        if timesteps <= 0:
            raise ValueError(f"curriculum stage {name!r} must have timesteps > 0")
        if name in seen_names:
            raise ValueError(f"duplicate curriculum stage name: {name!r}")
        seen_names.add(name)

        disturbances = item.get("disturbances")
        if disturbances is not None:
            item["disturbances"] = tuple(disturbances)

        stages.append(CurriculumStage(name=name, timesteps=timesteps, env_overrides=item))

    return Curriculum(
        stages=tuple(stages),
        rehearsal_probability=rehearsal_probability,
        evaluation_episodes_per_stage=eval_eps,
        rehearsal_mode=rehearsal_mode,
    )


def env_config_for_stage(base_env_cfg, stage: CurriculumStage):
    """Deep-copy an EnvConfig and apply one curriculum stage's overrides."""
    cfg = copy.deepcopy(base_env_cfg)
    for key, value in stage.env_overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(
                f"curriculum stage {stage.name!r} uses unknown EnvConfig field {key!r}"
            )
        setattr(cfg, key, value)
    return cfg
