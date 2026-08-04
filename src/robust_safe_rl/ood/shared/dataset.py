"""Rollout-based dataset collection for autoencoder OOD detection.

Two collectors live here so every script shares one rollout implementation:

- :func:`collect_dataset` varies the *external force* disturbance (the training
  and force-OOD setting).
- :func:`collect_parameter_dataset` instead varies *mass and inertia* while
  keeping forces in-distribution (used to test whether a force-trained detector
  generalizes to parameter-type OOD).

In both cases a single controller reads the **true** state and its command is
applied identically to the true and nominal models, matching the feature
alignment described in :mod:`robust_safe_rl.ood.shared.features`.
"""

from collections import deque

import numpy as np

from robust_safe_rl.core import Controller, DesiredTrajectory, Dynamics
from .features import STEP_FEATURE_DIM, make_step_feature

NOMINAL_MASS = 2.0
NOMINAL_J = np.diag([0.022, 0.022, 0.04])


def collect_dataset(
    num_episodes,
    episode_steps,
    dt,
    random_force,
    history_len,
    seed,
    force_sample_each_step=False,
):
    """Collect history-stacked features under a chosen external-force mode.

    Returns
    -------
    x : (N, STEP_FEATURE_DIM * history_len) float32 array of samples.
    disturbance : (N, 3) float32 array of the true external force per sample.
    """
    if num_episodes <= 0 or episode_steps <= 0 or history_len <= 0:
        raise ValueError("num_episodes, episode_steps, and history_len must be positive.")
    if episode_steps < history_len:
        raise ValueError("episode_steps must be at least history_len.")

    rng = np.random.default_rng(seed)
    traj = DesiredTrajectory(radius=0.79, speed=0.5, z0=-1.0)

    samples = []
    forces = []

    for _ in range(num_episodes):
        true_dyn = Dynamics(
            dt=dt,
            random_force=random_force,
            force_sample_each_step=force_sample_each_step,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        nominal_dyn = Dynamics(dt=dt, random_force=0)

        # One controller represents the real command source. Its command is
        # applied identically to the true and nominal dynamics.
        controller = Controller(dt=dt)

        d0 = traj.desired(0.0)
        true_state = true_dyn.reset(x=d0["x"], v=d0["v"])
        nominal_dyn.reset(x=d0["x"], v=d0["v"])
        controller.reset()

        history = deque(maxlen=history_len)

        for k in range(episode_steps):
            t = k * dt
            desired = traj.desired(t)

            f_cmd, M_cmd, _ = controller.compute_control(true_state, desired)
            action = np.array([f_cmd, *M_cmd], dtype=float)

            true_next = true_dyn.step(f_cmd, M_cmd)
            nominal_next = nominal_dyn.step(f_cmd, M_cmd)

            history.append(make_step_feature(nominal_next, true_next, action))

            if len(history) == history_len:
                samples.append(np.concatenate(tuple(history)))
                forces.append(true_dyn.last_external_force.copy())

            true_state = true_next

    x = np.asarray(samples, dtype=np.float32)
    disturbance = np.asarray(forces, dtype=np.float32)
    expected_dim = STEP_FEATURE_DIM * history_len

    if x.ndim != 2 or x.shape[1] != expected_dim:
        raise RuntimeError(f"Expected dataset shape (N, {expected_dim}), got {x.shape}.")
    if not np.all(np.isfinite(x)):
        raise FloatingPointError("Collected dataset contains non-finite values.")

    return x, disturbance


def collect_parameter_dataset(
    *,
    forces,
    mass_scales,
    moi_scales,
    episode_steps,
    dt,
    history_len,
    seed,
):
    """Collect samples with ID forces and specified mass/inertia multipliers.

    The true and nominal systems receive exactly the same commanded action. The
    nominal system always uses nominal mass/inertia and no external force. The
    true system uses the supplied ID external force and parameter scales.
    Parameter scales are fixed within an episode.
    """
    forces = np.asarray(forces, dtype=float)
    mass_scales = np.asarray(mass_scales, dtype=float)
    moi_scales = np.asarray(moi_scales, dtype=float)

    num_episodes = len(forces)
    if forces.shape != (num_episodes, 3):
        raise ValueError(f"forces must have shape (N, 3), got {forces.shape}")
    if mass_scales.shape != (num_episodes,):
        raise ValueError("mass_scales must have shape (N,)")
    if moi_scales.shape not in ((num_episodes,), (num_episodes, 3)):
        raise ValueError("moi_scales must have shape (N,) or (N, 3)")
    if episode_steps < history_len:
        raise ValueError("episode_steps must be at least history_len")

    traj = DesiredTrajectory(radius=0.79, speed=0.5, z0=-1.0)
    rng = np.random.default_rng(seed)

    samples = []
    sample_episode_ids = []
    sample_mass_scales = []
    sample_moi_scales = []
    sample_forces = []

    for episode_id in range(num_episodes):
        mass_scale = float(mass_scales[episode_id])
        if moi_scales.ndim == 1:
            moi_scale_vec = np.full(3, float(moi_scales[episode_id]))
        else:
            moi_scale_vec = np.asarray(moi_scales[episode_id], dtype=float)

        true_mass = NOMINAL_MASS * mass_scale
        true_J = NOMINAL_J @ np.diag(moi_scale_vec)

        true_dyn = Dynamics(
            dt=dt,
            mass=true_mass,
            J=true_J,
            random_force=0,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        nominal_dyn = Dynamics(
            dt=dt,
            mass=NOMINAL_MASS,
            J=NOMINAL_J,
            random_force=0,
        )
        controller = Controller(dt=dt)

        d0 = traj.desired(0.0)
        force = forces[episode_id]
        true_state = true_dyn.reset(x=d0["x"], v=d0["v"], external_force=force)
        nominal_dyn.reset(x=d0["x"], v=d0["v"], external_force=np.zeros(3))
        controller.reset()

        history = deque(maxlen=history_len)

        for k in range(episode_steps):
            # Divergence guard: the light-drone (mass_scale < 1) case makes the
            # nominal-parameter base controller over-thrust and tumble, which
            # overflows the rotation integration and crashes project_to_so3's SVD.
            # Stop the episode before that happens so we log only valid samples.
            if (not np.all(np.isfinite(true_state["x"]))
                    or np.linalg.norm(true_state["v"]) > 10.0
                    or np.linalg.norm(true_state["omega"]) > 10.0
                    or true_state["R"][2, 2] < 0.5):   # >60 deg tilt
                break

            desired = traj.desired(k * dt)
            f_cmd, M_cmd, _ = controller.compute_control(true_state, desired)
            action = np.array([f_cmd, *M_cmd], dtype=float)

            true_next = true_dyn.step(f_cmd, M_cmd)
            nominal_next = nominal_dyn.step(f_cmd, M_cmd)

            history.append(make_step_feature(nominal_next, true_next, action))
            if len(history) == history_len:
                samples.append(np.concatenate(tuple(history)))
                sample_episode_ids.append(episode_id)
                sample_mass_scales.append(mass_scale)
                sample_moi_scales.append(moi_scale_vec.copy())
                sample_forces.append(force.copy())

            true_state = true_next

    x = np.asarray(samples, dtype=np.float32)
    expected_dim = STEP_FEATURE_DIM * history_len
    if x.ndim != 2 or x.shape[1] != expected_dim:
        raise RuntimeError(f"Expected dataset shape (N, {expected_dim}), got {x.shape}")
    if not np.all(np.isfinite(x)):
        raise FloatingPointError("Dataset contains non-finite values")

    return {
        "x": x,
        "episode_id": np.asarray(sample_episode_ids, dtype=np.int32),
        "mass_scale": np.asarray(sample_mass_scales, dtype=np.float32),
        "moi_scale": np.asarray(sample_moi_scales, dtype=np.float32),
        "force": np.asarray(sample_forces, dtype=np.float32),
    }


def standardize(train_x, val_x, ood_x):
    """Fit standardization on train, apply to all three splits. Returns tensors + stats."""
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (
        (train_x - mean) / std,
        (val_x - mean) / std,
        (ood_x - mean) / std,
        mean,
        std,
    )


def apply_standardization(x, mean, std):
    """Apply a previously fitted (mean, std) standardization to a 2-D tensor."""
    if x.ndim != 2:
        raise ValueError(f"Expected a 2-D input tensor, got shape {tuple(x.shape)}.")
    if x.shape[1] != mean.shape[-1] or mean.shape != std.shape:
        raise ValueError(
            f"Input/normalization dimension mismatch: input={x.shape[1]}, "
            f"mean={tuple(mean.shape)}, std={tuple(std.shape)}."
        )
    return (x - mean) / std.clamp_min(1e-6)