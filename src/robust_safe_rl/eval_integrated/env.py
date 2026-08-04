"""Evaluation environment for the integrated detector+residual study.

Extends the twin-plant setup with a time-varying disturbance scenario:

  * mass/MOI:  a per-episode multiplier k ~ U[k_min, k_max] applied from t=0.
  * force:     a constant external force of a sampled magnitude/direction that
               switches ON at a sampled onset time in [onset_min, onset_max]
               (default 3-5 s) and persists to the end of the episode.

The environment is designed for FAIR replay: a "scenario" (k, force vector,
onset step) is sampled once, then the SAME scenario can be run under each of the
four controller cases so differences are due to the controller, not the
disturbance draw.

Unlike the training env, this one does not compute a residual observation
internally in a fixed mode -- instead it exposes everything a controller and the
detectors need each step, and accepts an externally-supplied total control. The
eval loop owns the policy/detector logic; the env owns the physics and the twin.

Per step it makes available:
  - true and nominal post-step states (for AE features and logging)
  - the residual-policy observation (built in the policy's obs_mode)
  - the measured residual acceleration a_res (for the dynamics detector)
  - the active force (0 before onset), k, onset step, tracking errors
"""

import numpy as np

from robust_safe_rl.core import Controller, DesiredTrajectory, Dynamics
from robust_safe_rl.core.so3 import rotation_error
from robust_safe_rl.rl.obs_builders import ObservationBuilder


class Scenario:
    """A fully-specified, replayable disturbance realization for one episode."""

    def __init__(self, k, force_vec, onset_step):
        self.k = float(k)
        self.force_vec = np.asarray(force_vec, dtype=float).reshape(3)
        self.onset_step = int(onset_step)
        self.force_mag = float(np.linalg.norm(self.force_vec))


def sample_scenario(cfg, rng, force_mode=1, onset_range_s=(3.0, 5.0)):
    """Draw a scenario: k, a constant force vector, and its onset step."""
    k = float(rng.uniform(cfg.k_min, cfg.k_max))
    # force vector: reuse the training force bands (mode 1 ID, 2 loose, 3 strict)
    if force_mode == 1:
        fvec = rng.uniform(-3.0, 3.0, size=3)
    elif force_mode == 2:
        fvec = rng.uniform(-5.0, 5.0, size=3)
    else:  # strict OOD: at least one axis beyond +-3
        while True:
            fvec = rng.uniform(-5.0, 5.0, size=3)
            if np.any(np.abs(fvec) > 3.0):
                break
    onset_step = int(rng.uniform(*onset_range_s) / cfg.dt)
    return Scenario(k, fvec, onset_step)


class IntegratedEvalEnv:
    """Twin-plant env with time-varying force, for the integrated evaluation."""

    def __init__(self, cfg, obs_mode="pid", device="cpu"):
        self.cfg = cfg
        self.dt = cfg.dt
        J_nom = np.diag(np.asarray(cfg.J_nom, dtype=float))
        self.mass_nom = cfg.mass_nom
        self.J_nom = J_nom

        self.dyn_true = Dynamics(dt=cfg.dt, mass=cfg.mass_nom, J=J_nom,
                                 gravity=cfg.gravity, random_force=0)
        self.dyn_nom = Dynamics(dt=cfg.dt, mass=cfg.mass_nom, J=J_nom,
                                gravity=cfg.gravity, random_force=0)
        self.ctrl_true = Controller(dt=cfg.dt, mass=cfg.mass_nom, J=J_nom,
                                    gravity=cfg.gravity)
        self.ctrl_nom = Controller(dt=cfg.dt, mass=cfg.mass_nom, J=J_nom,
                                   gravity=cfg.gravity)
        self.traj = DesiredTrajectory(radius=cfg.traj_radius, speed=cfg.traj_speed,
                                      z0=cfg.traj_z0)

        # observation builder matching the trained policy's obs_mode
        obs_cfg = _clone_env_cfg(cfg, obs_mode)
        self.obs_builder = ObservationBuilder(obs_cfg)
        self.obs_dim = self.obs_builder.obs_dim

        self.scenario = None
        self.step_idx = 0
        self.t = 0.0

    def reset(self, scenario):
        """Start an episode with a fixed, replayable scenario."""
        self.scenario = scenario
        self.step_idx = 0
        self.t = 0.0
        self.dyn_true.set_inertial_scale(scenario.k)   # mass/MOI from t=0
        self.dyn_nom.set_inertial_scale(1.0)
        d0 = self.traj.desired(0.0)
        # force is off at reset; injected at onset in step()
        self.dyn_true.reset(x=d0["x"], v=d0["v"], external_force=np.zeros(3))
        self.dyn_nom.reset(x=d0["x"], v=d0["v"], external_force=np.zeros(3))
        self.ctrl_true.reset()
        self.ctrl_nom.reset()
        self.obs_builder.reset()
        return self.obs_builder.get()

    def base_control(self):
        """Base control for the true plant at the current state (nominal params)."""
        st = self.dyn_true.state()
        desired = self.traj.desired(self.t)
        f, M, _ = self.ctrl_true.compute_control(st, desired)
        return f, M

    def step(self, residual):
        """Advance one step given a residual (4,) in physical units.

        Returns a dict with everything the controllers, detectors, and logger
        need. The caller computes the residual (possibly zero after fallback).
        """
        residual = np.asarray(residual, dtype=float).reshape(4)
        sc = self.scenario

        # force active this step?
        force_on = self.step_idx >= sc.onset_step
        force_vec = sc.force_vec if force_on else np.zeros(3)
        self.dyn_true.external_force = force_vec.copy()   # applied to true only

        st = self.dyn_true.state()
        sn = self.dyn_nom.state()
        desired = self.traj.desired(self.t)

        # base control for each plant on its own state
        f_t, M_t, _ = self.ctrl_true.compute_control(st, desired)
        f_n, M_n, _ = self.ctrl_nom.compute_control(sn, desired)

        u_res = residual
        f_true = f_t + u_res[0]
        M_true = M_t + u_res[1:4]
        u_total = np.array([f_true, M_true[0], M_true[1], M_true[2]])
        u_base = np.array([f_t, M_t[0], M_t[1], M_t[2]])

        # ---- residual-acceleration target (for the dynamics detector) ----
        # a_true includes true m,J AND the active force; a_nominal uses nominal
        # m,J and NO force -- same state. This is what the dynamics detector's
        # model (trained on massmoi only) must predict; the force makes it OOD.
        vd_true, wd_true = self.dyn_true.accel_at(
            st["R"], st["v"], st["omega"], f_true, M_true,
            mass=sc.k * self.mass_nom, J=sc.k * self.J_nom, external_force=force_vec)
        vd_nom, wd_nom = self.dyn_true.accel_at(
            st["R"], st["v"], st["omega"], f_true, M_true,
            mass=self.mass_nom, J=self.J_nom, external_force=np.zeros(3))
        a_res = np.concatenate([vd_true - vd_nom, wd_true - wd_nom])

        # advance both plants
        st_next = self.dyn_true.step(f_true, M_true)
        sn_next = self.dyn_nom.step(f_n, M_n)

        self.step_idx += 1
        self.t = self.step_idx * self.dt

        # update the policy observation buffer
        self.obs_builder.push(sn_next, st_next, u_res, u_total, u_base)

        # tracking error: true vs desired (what we ultimately care about) and
        # true vs nominal twin (the DOB reference / detector-relevant signal)
        pos_err_des = float(np.linalg.norm(st_next["x"] - desired["x"]))
        pos_err_nom = float(np.linalg.norm(sn_next["x"] - st_next["x"]))

        # terminated if the true plant diverges
        terminated = (not np.all(np.isfinite(st_next["x"]))
                      or pos_err_nom > 5.0 or st_next["R"][2, 2] < 0.0)
        truncated = self.step_idx >= self.cfg.episode_steps

        return {
            "obs": self.obs_builder.get(),
            "true_next": st_next, "nom_next": sn_next,
            "st_for_accel": st,           # state the a_res was measured at
            "u_total": u_total, "u_base": u_base, "u_res": u_res,
            "a_res": a_res,
            "force_on": force_on, "force_vec": force_vec.copy(),
            "pos_err_des": pos_err_des, "pos_err_nom": pos_err_nom,
            "terminated": terminated, "truncated": truncated,
            "step": self.step_idx,
        }


def _clone_env_cfg(cfg, obs_mode):
    """Shallow clone of the env config with a chosen obs_mode (leaves original)."""
    import copy
    c = copy.copy(cfg)
    c.obs_mode = obs_mode
    return c