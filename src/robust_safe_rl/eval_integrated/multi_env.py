"""Generalized evaluation environment for the multi-disturbance OOD study.

Extends the force-only IntegratedEvalEnv to inject any of three OOD disturbance
types, either constant from t=0 (Phase 1 severity sweeps) or switched on at a
mid-episode onset (Phases 2-3):

  * force         : constant external force vector (N).
  * thrust_factor : motor thrust-coefficient scale via the mixer round-trip
                    (allocate with nominal mixer, reconstruct with true).
  * arm_length    : rotor arm-length scale via the mixer round-trip.

Mass/MOI (the in-distribution family) is always applied from t=0 via the
per-episode multiplier k, matching how the residual policy and detectors were
trained. The OOD disturbance is layered on top.

The env exposes, per step, everything the controllers and detectors need: the
policy observation, the twin states, and the residual acceleration a_res
computed with the true (disturbed) vs nominal model at the same state.
"""

import copy
import numpy as np

from robust_safe_rl.core import Controller, DesiredTrajectory, Dynamics
from robust_safe_rl.rl.obs_builders import ObservationBuilder
from robust_safe_rl.rl.mixer import Mixer, NOMINAL_ARM


class DisturbanceScenario:
    """A replayable disturbance realization: mass/MOI k + one OOD disturbance.

    dist_type : "none" | "force" | "thrust_factor" | "arm_length"
    severity  : force -> 3-vector (N); thrust_factor/arm_length -> scalar multiplier
    onset_step: step index at which the OOD disturbance switches on (0 = from start)
    """

    def __init__(self, k, dist_type, severity, onset_step):
        self.k = float(k)
        self.dist_type = dist_type
        self.onset_step = int(onset_step)
        if dist_type == "force":
            self.force_vec = np.asarray(severity, dtype=float).reshape(3)
            self.severity_scalar = float(np.linalg.norm(self.force_vec))
        elif dist_type in ("thrust_factor", "arm_length"):
            self.force_vec = np.zeros(3)
            self.severity_scalar = float(severity)
        else:  # none
            self.force_vec = np.zeros(3)
            self.severity_scalar = 0.0


def sample_scenario(cfg, rng, dist_type="force", onset_range_s=(3.0, 5.0),
                    force_mode=3, param_min=0.7, param_max=1.3, onset_step=None,
                    k=None):
    """Draw a scenario. onset_step overrides the sampled onset when given
    (e.g. 0 for constant-from-t=0 Phase 1 runs)."""
    k = float(rng.uniform(cfg.k_min, cfg.k_max)) if k is None else float(k)
    if dist_type == "force":
        if force_mode == 1:
            sev = rng.uniform(-3.0, 3.0, size=3)
        elif force_mode == 2:
            sev = rng.uniform(-5.0, 5.0, size=3)
        else:
            while True:
                sev = rng.uniform(-5.0, 5.0, size=3)
                if np.any(np.abs(sev) > 3.0):
                    break
    elif dist_type in ("thrust_factor", "arm_length"):
        sev = float(rng.uniform(param_min, param_max))
    else:
        sev = 0.0
    if onset_step is None:
        onset_step = int(rng.uniform(*onset_range_s) / cfg.dt)
    return DisturbanceScenario(k, dist_type, sev, onset_step)


class MultiDisturbanceEvalEnv:
    """Twin-plant eval env supporting force / thrust_factor / arm_length OOD."""

    def __init__(self, cfg, obs_mode="pid"):
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
        self.mixer = Mixer()

        obs_cfg = copy.copy(cfg); obs_cfg.obs_mode = obs_mode
        self.obs_builder = ObservationBuilder(obs_cfg)
        self.obs_dim = self.obs_builder.obs_dim

        self.scenario = None
        self.step_idx = 0
        self.t = 0.0

    def reset(self, scenario):
        self.scenario = scenario
        self.step_idx = 0
        self.t = 0.0
        self.dyn_true.set_inertial_scale(scenario.k)   # mass/MOI from t=0
        self.dyn_nom.set_inertial_scale(1.0)
        self.mixer.reset_true()
        d0 = self.traj.desired(0.0)
        self.dyn_true.reset(x=d0["x"], v=d0["v"], external_force=np.zeros(3))
        self.dyn_nom.reset(x=d0["x"], v=d0["v"], external_force=np.zeros(3))
        self.ctrl_true.reset()
        self.ctrl_nom.reset()
        self.obs_builder.reset()
        return self.obs_builder.get()

    def _apply_disturbance_wrench(self, f, M, active):
        """Return the wrench actually applied to the true plant, given the OOD
        disturbance state. thrust_factor/arm_length go through the mixer;
        force is handled separately (as an external force on the dynamics)."""
        sc = self.scenario
        if active and sc.dist_type == "thrust_factor":
            self.mixer.set_true(kf_scale=sc.severity_scalar)
            return self.mixer.apply(f, M)
        if active and sc.dist_type == "arm_length":
            self.mixer.set_true(arm_x=NOMINAL_ARM * sc.severity_scalar)
            return self.mixer.apply(f, M)
        # force or inactive: mixer is identity
        self.mixer.reset_true()
        return float(f), np.asarray(M, dtype=float)

    def step(self, residual):
        residual = np.asarray(residual, dtype=float).reshape(4)
        sc = self.scenario
        active = self.step_idx >= sc.onset_step

        # external force disturbance (only for dist_type == force)
        force_vec = sc.force_vec if (active and sc.dist_type == "force") else np.zeros(3)
        self.dyn_true.external_force = force_vec.copy()

        st = self.dyn_true.state()
        sn = self.dyn_nom.state()
        desired = self.traj.desired(self.t)

        f_t, M_t, _ = self.ctrl_true.compute_control(st, desired)
        f_n, M_n, _ = self.ctrl_nom.compute_control(sn, desired)

        # residual added to the commanded wrench (before allocation)
        u_res = residual
        f_cmd = f_t + u_res[0]
        M_cmd = M_t + u_res[1:4]

        # apply mixer round-trip for parameter disturbances
        f_true, M_true = self._apply_disturbance_wrench(f_cmd, M_cmd, active)

        u_total = np.array([f_true, M_true[0], M_true[1], M_true[2]])
        u_base = np.array([f_t, M_t[0], M_t[1], M_t[2]])

        # residual acceleration: true (disturbed params/force + actual wrench) vs
        # nominal model prediction for the COMMANDED wrench, same state.
        vd_true, wd_true = self.dyn_true.accel_at(
            st["R"], st["v"], st["omega"], f_true, M_true,
            mass=sc.k * self.mass_nom, J=sc.k * self.J_nom, external_force=force_vec)
        vd_nom, wd_nom = self.dyn_true.accel_at(
            st["R"], st["v"], st["omega"], f_cmd, M_cmd,
            mass=self.mass_nom, J=self.J_nom, external_force=np.zeros(3))
        a_res = np.concatenate([vd_true - vd_nom, wd_true - wd_nom])

        st_next = self.dyn_true.step(f_true, M_true)
        sn_next = self.dyn_nom.step(f_n, M_n)

        self.step_idx += 1
        self.t = self.step_idx * self.dt
        self.obs_builder.push(sn_next, st_next, u_res, u_total, u_base)

        pos_err_des = float(np.linalg.norm(st_next["x"] - desired["x"]))
        pos_err_nom = float(np.linalg.norm(sn_next["x"] - st_next["x"]))
        terminated = (not np.all(np.isfinite(st_next["x"]))
                      or pos_err_nom > 5.0 or st_next["R"][2, 2] < 0.0)
        truncated = self.step_idx >= self.cfg.episode_steps

        return {
            "obs": self.obs_builder.get(),
            "true_next": st_next, "nom_next": sn_next, "st_for_accel": st,
            "u_total": u_total, "u_base": u_base, "u_res": u_res, "a_res": a_res,
            "dist_active": active, "dist_type": sc.dist_type,
            "pos_err_des": pos_err_des, "pos_err_nom": pos_err_nom,
            "terminated": terminated, "truncated": truncated, "step": self.step_idx,
        }