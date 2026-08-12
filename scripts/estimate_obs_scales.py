"""Estimate fixed normalization scales for the residual-SAC history observation.

The nominal twin remains disturbance-free. The true plant is rolled out with:
  * a per-episode inertial scale k ~ U[k_min, k_max], and
  * a constant per-episode external force F_ext ~ U[-force_max, force_max]^3.

The residual action is fixed at zero so the collected discrepancy is what the
baseline controller alone leaves for SAC to compensate.

Run from the repository root, for example:

    python -m robust_safe_rl.scripts.estimate_obs_scales \
        --episodes 300 --k_min 0.7 --k_max 1.3 --force_max 3.0 \
        --output obs_scale_stats.json

The script reports empirical means/stds for diagnostics, but the recommended
controller-friendly normalization keeps the center at zero and uses one fixed
scale per 3-D error group to preserve axis symmetry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from robust_safe_rl.rl.config import Config
from robust_safe_rl.rl.obs_builders import discrepancy
from robust_safe_rl.rl.residual_env import ResidualTwinEnv


GROUPS = {
    "pos": slice(0, 3),
    "vel": slice(3, 6),
    "att": slice(6, 9),
    "omega": slice(9, 12),
}


def _group_stats(values: np.ndarray) -> dict:
    """Statistics for an (N, 3) error group."""
    values = np.asarray(values, dtype=np.float64)
    abs_components = np.abs(values).reshape(-1)
    norms = np.linalg.norm(values, axis=1)

    # One pooled scale across x/y/z preserves symmetry within the group.
    pooled_rms = float(np.sqrt(np.mean(values ** 2)))
    p95_abs = float(np.percentile(abs_components, 95.0))
    p99_abs = float(np.percentile(abs_components, 99.0))

    return {
        "mean_per_axis": values.mean(axis=0).tolist(),
        "std_per_axis": values.std(axis=0).tolist(),
        "pooled_rms": pooled_rms,
        "abs_component_p95": p95_abs,
        "abs_component_p99": p99_abs,
        "vector_norm_p95": float(np.percentile(norms, 95.0)),
        "vector_norm_p99": float(np.percentile(norms, 99.0)),
    }


def collect(args) -> dict:
    cfg = Config()
    cfg.env.obs_mode = "history"
    cfg.env.k_min = float(args.k_min)
    cfg.env.k_max = float(args.k_max)

    rng = np.random.default_rng(args.seed)
    env = ResidualTwinEnv(cfg.env, seed=args.seed)
    zero_action = np.zeros(env.action_dim, dtype=np.float32)

    errors = []
    episode_lengths = []
    forces = []
    ks = []
    n_terminated = 0
    n_saturated_steps = 0

    for _ in range(args.episodes):
        k = float(rng.uniform(args.k_min, args.k_max))
        env.reset(k=k)

        # Constant episode-level force on the true plant only. This is injected
        # after env.reset() because ResidualTwinEnv currently constructs the SAC
        # training plant with random_force=0.
        f_ext = rng.uniform(-args.force_max, args.force_max, size=3)
        env.dyn_true.external_force = f_ext.copy()
        env.dyn_true.last_external_force = f_ext.copy()

        forces.append(f_ext.copy())
        ks.append(k)

        steps = 0
        while True:
            _, _, terminated, truncated, info = env.step(zero_action)
            e = discrepancy(env.dyn_nom.state(), env.dyn_true.state())
            errors.append(e)
            steps += 1
            n_saturated_steps += int(info["actuator_saturated"])

            if terminated or truncated:
                n_terminated += int(terminated)
                break

        episode_lengths.append(steps)

    errors = np.asarray(errors, dtype=np.float64)
    if errors.size == 0:
        raise RuntimeError("No discrepancy samples were collected.")

    result = {
        "collection": {
            "episodes": int(args.episodes),
            "samples": int(errors.shape[0]),
            "dt": float(cfg.env.dt),
            "episode_steps_max": int(cfg.env.episode_steps),
            "k_min": float(args.k_min),
            "k_max": float(args.k_max),
            "force_distribution": f"per-episode constant U[-{args.force_max}, {args.force_max}] N on each axis",
            "residual_action": "zero",
            "terminated_episodes": int(n_terminated),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "actuator_saturated_steps": int(n_saturated_steps),
            "seed": int(args.seed),
        },
        "full_error_mean": errors.mean(axis=0).tolist(),
        "full_error_std": errors.std(axis=0).tolist(),
        "groups": {
            name: _group_stats(errors[:, sl]) for name, sl in GROUPS.items()
        },
        "normalization_note": {
            "recommended_center": [0.0] * 12,
            "description": "Inspect pooled RMS and the 95th/99th percentiles before choosing one frozen scale per 3-D group. Leave past SAC actions unchanged in [-1, 1].",
        },
    }
    return result


def print_summary(result: dict) -> None:
    c = result["collection"]
    print("\nCollection")
    print("----------")
    print(f"episodes              : {c['episodes']}")
    print(f"samples               : {c['samples']}")
    print(f"terminated episodes   : {c['terminated_episodes']}")
    print(f"mean episode length   : {c['mean_episode_length']:.1f}")
    print(f"actuator sat. steps   : {c['actuator_saturated_steps']}")
    print(f"k range               : [{c['k_min']}, {c['k_max']}]")
    print(f"external force        : {c['force_distribution']}")

    print("\nError-group statistics")
    print("----------------------")
    for name, stats in result["groups"].items():
        mean = np.asarray(stats["mean_per_axis"])
        std = np.asarray(stats["std_per_axis"])
        print(f"{name:>5s}: mean={np.array2string(mean, precision=5)}  "
              f"std={np.array2string(std, precision=5)}")
        print(f"       pooled RMS={stats['pooled_rms']:.6g}  "
              f"|component| p95={stats['abs_component_p95']:.6g}  "
              f"p99={stats['abs_component_p99']:.6g}  "
              f"||group|| p95={stats['vector_norm_p95']:.6g}")

    print("\nNormalization note")
    print("------------------")
    print("Keep the normalization center at zero. Use these statistics to choose one")
    print("frozen scale for each 3-D group; do not blindly use a small empirical std")
    print("without checking the p95/p99 tails. Past SAC actions stay in [-1, 1].")


def parse_args():
    defaults = Config().env
    p = argparse.ArgumentParser(
        description="Estimate fixed residual-SAC observation scales under baseline-only disturbed rollouts."
    )
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--k_min", type=float, default=defaults.k_min)
    p.add_argument("--k_max", type=float, default=defaults.k_max)
    p.add_argument(
        "--force_max",
        type=float,
        default=3.0,
        help="sample one constant external-force vector per episode, each axis U[-force_max, force_max] N",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="obs_scale_stats.json")
    return p.parse_args()


def main():
    args = parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be >= 1")
    if args.k_min <= 0.0 or args.k_max <= 0.0 or args.k_min > args.k_max:
        raise ValueError("require 0 < k_min <= k_max")
    if args.force_max < 0.0:
        raise ValueError("force_max must be >= 0")

    result = collect(args)
    print_summary(result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved full statistics to: {output}")


if __name__ == "__main__":
    main()
