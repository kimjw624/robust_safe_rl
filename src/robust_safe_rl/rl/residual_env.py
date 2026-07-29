"""Twin-plant residual-RL environment (disturbance-observer style).

Two rigid-body plants are stepped in lockstep from the same initial condition:

  * nominal twin : nominal mass/inertia, NO disturbance, driven by its OWN base
                   controller only. This is the reference behaviour -- "what the
                   ideal controller does in an ideal world."
  * true plant   : mass and inertia scaled by a per-episode factor k ~ U[k_min,
                   k_max]. Driven by the base controller (which still uses the
                   NOMINAL m, J -- it does not know k) PLUS the learned residual.

The agent observes a length-H history of frames, each frame being the
nominal-minus-true state discrepancy together with the residual and total
control that produced it. The reward drives the true plant to track the nominal
twin (not the mathematical desired trajectory) -- the DOB objective. The
residual exists solely to cancel the effect of k.

Each plant gets its OWN Controller instance: the geometric controller keeps
internal finite-difference state (Rd_prev, omega_d_prev), so sharing one
instance across the two diverging plants would corrupt both. Both controllers
are reset every episode.

The environment follows the common Gym-style reset()/step() contract but does
not depend on gymnasium, to keep the core light. Observations and actions are
float32 numpy arrays.
"""

import numpy as np

from robust_safe_rl.core import Controller, DesiredTrajectory, Dynamics
from robust_safe_rl.core.so3 import rotation_error


# Per-frame layout: [ dx(3) dv(3) dR(3) domega(3) | residual(4) | u_total(4) ]
FRAME_STATE = 12
FRAME_ACTION = 4
FRAME_TOTAL_U = 4
FRAME_DIM = FRAME_STATE + FRAME_ACTION + FRAME_TOTAL_U  # 20
ACTION_DIM = 4


class ResidualTwinEnv:
    """Residual-RL env with a nominal reference twin and a disturbed true plant."""

    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.H = cfg.history
        self.obs_dim = self.H * FRAME_DIM
        self.action_dim = ACTION_DIM

        self.action_scale = np.asarray(cfg.action_scale, dtype=np.float64)

        J_nom = np.diag(np.asarray(cfg.J_nom, dtype=float))

        # True and nominal plants. The true plant carries the mass/MOI disturbance;
        # the nominal twin never does. random_force stays 0 for both -- the only
        # disturbance in this phase is the inertial scaling k.
        self.dyn_true = Dynamics(
            dt=cfg.dt, mass=cfg.mass_nom, J=J_nom, gravity=cfg.gravity, random_force=0,
        )
        self.dyn_nom = Dynamics(
            dt=cfg.dt, mass=cfg.mass_nom, J=J_nom, gravity=cfg.gravity, random_force=0,
        )

        # One controller per plant (independent finite-difference state). Both use
        # NOMINAL m, J -- the controller does not get to see the true k.
        self.ctrl_true = Controller(dt=cfg.dt, mass=cfg.mass_nom, J=J_nom, gravity=cfg.gravity)
        self.ctrl_nom = Controller(dt=cfg.dt, mass=cfg.mass_nom, J=J_nom, gravity=cfg.gravity)

        self.traj = DesiredTrajectory(
            radius=cfg.traj_radius, speed=cfg.traj_speed, z0=cfg.traj_z0,
        )

        self.rng = np.random.default_rng(seed)

        self._tilt_cos_thresh = np.cos(np.deg2rad(cfg.term_tilt_deg))

        self.t = 0.0
        self.step_idx = 0
        self.k = 1.0
        self._frames = None  # ring of the last H frames, oldest first

    # ------------------------------------------------------------------ reset
    def reset(self, k=None):
        """Start a new episode. Optionally force the disturbance factor k."""
        self.step_idx = 0
        self.t = 0.0

        # Sample the per-episode inertial multiplier and apply it to the true plant.
        self.k = float(k) if k is not None else float(
            self.rng.uniform(self.cfg.k_min, self.cfg.k_max)
        )
        self.dyn_true.set_inertial_scale(self.k)
        self.dyn_nom.set_inertial_scale(1.0)  # twin stays nominal

        # Both plants start on the desired trajectory with matching state.
        d0 = self.traj.desired(0.0)
        self.dyn_true.reset(x=d0["x"], v=d0["v"])
        self.dyn_nom.reset(x=d0["x"], v=d0["v"])

        self.ctrl_true.reset()
        self.ctrl_nom.reset()

        # History starts zero-padded.
        self._frames = np.zeros((self.H, FRAME_DIM), dtype=np.float64)

        return self._get_obs()

    # ------------------------------------------------------------------- step
    def step(self, action):
        """Apply a residual action (post-tanh, in [-1, 1]^4) and advance one step.

        Returns (obs, reward, terminated, truncated, info).
        """
        action = np.asarray(action, dtype=np.float64).reshape(ACTION_DIM)
        residual = np.clip(action, -1.0, 1.0) * self.action_scale  # -> physical units

        desired = self.traj.desired(self.t)

        st = self.dyn_true.state()
        sn = self.dyn_nom.state()

        # Base control for each plant on its OWN state (nominal params inside both).
        f_t, M_t, _ = self.ctrl_true.compute_control(st, desired)
        f_n, M_n, _ = self.ctrl_nom.compute_control(sn, desired)

        # Residual is added to the TRUE plant only.
        u_res = residual                          # [df, dMx, dMy, dMz]
        f_true = f_t + u_res[0]
        M_true = M_t + u_res[1:4]

        # Advance both plants.
        st_next = self.dyn_true.step(f_true, M_true)
        sn_next = self.dyn_nom.step(f_n, M_n)      # base-only twin

        self.step_idx += 1
        self.t = self.step_idx * self.cfg.dt

        # Build the new history frame from the post-step discrepancy.
        u_total = np.array([f_true, M_true[0], M_true[1], M_true[2]])
        frame = self._make_frame(sn_next, st_next, u_res, u_total)
        self._frames = np.roll(self._frames, -1, axis=0)
        self._frames[-1] = frame

        reward = self._reward(sn_next, st_next)

        terminated = self._check_terminated(sn_next, st_next)
        truncated = self.step_idx >= self.cfg.episode_steps

        info = {"k": self.k, "pos_err": float(np.linalg.norm(sn_next["x"] - st_next["x"]))}
        return self._get_obs(), reward, terminated, truncated, info

    # -------------------------------------------------------------- internals
    def _make_frame(self, sn, st, u_res, u_total):
        dx = sn["x"] - st["x"]
        dv = sn["v"] - st["v"]
        dR = rotation_error(st["R"], sn["R"])       # attitude discrepancy (nominal vs true)
        dw = sn["omega"] - st["omega"]
        return np.concatenate([dx, dv, dR, dw, u_res, u_total]).astype(np.float64)

    def _get_obs(self):
        return self._frames.reshape(-1).astype(np.float32)

    def _reward(self, sn, st):
        c = self.cfg
        dx = np.linalg.norm(sn["x"] - st["x"]) ** 2
        dv = np.linalg.norm(sn["v"] - st["v"]) ** 2
        dR = np.linalg.norm(rotation_error(st["R"], sn["R"])) ** 2
        dw = np.linalg.norm(sn["omega"] - st["omega"]) ** 2

        r = (
            c.w_pos * np.exp(-dx / c.tau_pos ** 2)
            + c.w_vel * np.exp(-dv / c.tau_vel ** 2)
            + c.w_att * np.exp(-dR / c.tau_att ** 2)
            + c.w_omega * np.exp(-dw / c.tau_omega ** 2)
        )
        # --- optional shaping (left off for now; enable + tune after trial 1) ---
        # effort   = -w_effort * np.sum(u_res ** 2)
        # smooth   = -w_smooth * np.sum((u_res - u_res_prev) ** 2)
        # r += effort + smooth
        return float(r / c.reward_norm)

    def _check_terminated(self, sn, st):
        pos_err = np.linalg.norm(sn["x"] - st["x"])
        if pos_err > self.cfg.term_pos_error:
            return True
        # tilt of the true plant: body z-axis vs world down (e3). R[:,2] . e3 = R[2,2].
        cos_tilt = st["R"][2, 2]
        if cos_tilt < self._tilt_cos_thresh:
            return True
        return False