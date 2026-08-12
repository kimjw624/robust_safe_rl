"""Twin-plant environment for the residual SAC policy.

The nominal twin is a disturbance-free reference closed loop. The true plant
uses the same nominal controller law on its own state plus one of two residual
interfaces selected by ``cfg.residual_interface``:

  * ``"wrench"``: legacy post-controller [df, dMx, dMy, dMz] residual,
  * ``"force_vector"``: 3-D NED/world ``delta_A`` inserted before the geometric
    controller constructs desired attitude and moments.

The true plant can carry any configured combination of episode-constant disturbances:

  * mass/MOI scale,
  * external force,
  * motor thrust-coefficient scale,
  * rotor momentConstant scale,
  * arm-length/geometry scale.

Actuator/geometry uncertainty is applied through a nominal-allocation / true-
plant mixer round trip. The controller therefore computes a desired wrench with
nominal parameters, the NOMINAL mixer converts it to motor commands, and the
TRUE mixer converts those same motor commands back to the wrench actually
produced by the uncertain vehicle.
"""

import numpy as np

from robust_safe_rl.core import Controller, DesiredTrajectory, Dynamics
from robust_safe_rl.core.so3 import rotation_error
from robust_safe_rl.rl.obs_builders import ObservationBuilder
from robust_safe_rl.rl.mixer import F_MAX, M_MAX, Mixer, NOMINAL_ARM

LEGACY_WRENCH_ACTION_DIM = 4
FORCE_VECTOR_ACTION_DIM = 3
_ALLOWED_DISTURBANCES = {
    "none", "massmoi", "force", "motor_coeff", "moment_coeff", "arm_length"
}


class ResidualTwinEnv:
    """Residual-RL env with a nominal reference twin and disturbed true plant."""

    def __init__(self, cfg, seed=None):
        self.cfg = cfg
        self.residual_interface = str(getattr(cfg, "residual_interface", "wrench"))
        if self.residual_interface == "wrench":
            self.action_dim = LEGACY_WRENCH_ACTION_DIM
        elif self.residual_interface == "force_vector":
            self.action_dim = FORCE_VECTOR_ACTION_DIM
        else:
            raise ValueError(
                "residual_interface must be 'wrench' or 'force_vector', got "
                f"{self.residual_interface!r}"
            )

        disturbances = getattr(cfg, "disturbances", ("massmoi",))
        if isinstance(disturbances, str):
            disturbances = (disturbances,)
        self.disturbances = tuple(dict.fromkeys(disturbances))
        unknown = set(self.disturbances) - _ALLOWED_DISTURBANCES
        if unknown:
            raise ValueError(f"unknown disturbance(s): {sorted(unknown)}")
        if "none" in self.disturbances and len(self.disturbances) != 1:
            raise ValueError("'none' cannot be combined with other disturbances")

        self._validate_range("k", cfg.k_min, cfg.k_max)
        self._validate_range("motor_coeff", cfg.motor_coeff_min, cfg.motor_coeff_max)
        self._validate_range("moment_coeff", cfg.moment_coeff_min, cfg.moment_coeff_max)
        self._validate_range("arm_length", cfg.arm_length_min, cfg.arm_length_max)
        if float(cfg.external_force_max) < 0.0:
            raise ValueError("external_force_max must be nonnegative")

        # SAC always lives in normalized [-1, 1]^d action space. Physical
        # scaling depends on the residual interface.
        if self.residual_interface == "wrench":
            self.residual_authority = float(cfg.residual_authority)
            if not 0.0 <= self.residual_authority <= 1.0:
                raise ValueError("residual_authority must lie in [0, 1]")
            beta = float(getattr(cfg, "wrench_thrust_filter_beta", 1.0))
            if not np.isfinite(beta) or not 0.0 < beta <= 1.0:
                raise ValueError("wrench_thrust_filter_beta must satisfy 0 < beta <= 1")
            self.wrench_thrust_filter_beta = beta
            physical_envelope = np.concatenate(([F_MAX], np.asarray(M_MAX, dtype=float)))
            self.action_scale = self.residual_authority * physical_envelope
        else:
            limit = float(getattr(cfg, "force_vector_limit_N", 4.0))
            if not np.isfinite(limit) or limit <= 0.0:
                raise ValueError("force_vector_limit_N must be a finite positive number")
            beta = float(getattr(cfg, "force_vector_filter_beta", 0.01))
            if not np.isfinite(beta) or not 0.0 < beta <= 1.0:
                raise ValueError("force_vector_filter_beta must satisfy 0 < beta <= 1")
            self.force_vector_filter_beta = beta
            self.residual_authority = None
            self.action_scale = np.full(FORCE_VECTOR_ACTION_DIM, limit, dtype=float)

        if getattr(cfg, "obs_mode", "history") == "pred_ares":
            if self.residual_interface != "wrench":
                raise ValueError("obs_mode='pred_ares' is only implemented for residual_interface='wrench'")
            from robust_safe_rl.rl.pred_ares_obs import PredAresObsBuilder
            self.obs_builder = PredAresObsBuilder(cfg, device=getattr(cfg, "device", "cpu"))
            self._pred_ares_mode = True
        else:
            self.obs_builder = ObservationBuilder(cfg, action_dim=self.action_dim)
            self._pred_ares_mode = False
        self.obs_dim = self.obs_builder.obs_dim

        J_nom = np.diag(np.asarray(cfg.J_nom, dtype=float))

        # Force is injected explicitly at reset so its magnitude is controlled by
        # EnvConfig and the same environment RNG as every other disturbance.
        self.dyn_true = Dynamics(
            dt=cfg.dt, mass=cfg.mass_nom, J=J_nom, gravity=cfg.gravity, random_force=0,
        )
        self.dyn_nom = Dynamics(
            dt=cfg.dt, mass=cfg.mass_nom, J=J_nom, gravity=cfg.gravity, random_force=0,
        )

        # One controller instance per plant because the geometric controller has
        # finite-difference internal state. Both controllers remain nominal.
        self.ctrl_true = Controller(dt=cfg.dt, mass=cfg.mass_nom, J=J_nom, gravity=cfg.gravity)
        self.ctrl_nom = Controller(dt=cfg.dt, mass=cfg.mass_nom, J=J_nom, gravity=cfg.gravity)

        self.traj = DesiredTrajectory(
            radius=cfg.traj_radius, speed=cfg.traj_speed, z0=cfg.traj_z0,
        )

        # One nominal allocator, with B_true changed episode-by-episode. The
        # nominal twin is reconstructed with B_nom through apply_nominal().
        self.mixer = Mixer()

        self.rng = np.random.default_rng(seed)
        self._tilt_cos_thresh = np.cos(np.deg2rad(cfg.term_tilt_deg))

        self.t = 0.0
        self.step_idx = 0
        self.k = 1.0
        self.external_force = np.zeros(3, dtype=float)
        self.motor_coeff_scale = np.ones(4, dtype=float)
        self.moment_coeff_scale = np.ones(4, dtype=float)
        self.arm_length_scale = np.ones(4, dtype=float)
        # ``prev_action_norm`` is the action that actually reached the physical
        # residual interface. History observations store this applied action so
        # the first-order command state is visible to the policy. The separate
        # requested-action state keeps the legacy effort/smoothness reward on the
        # SAC command itself, preventing hidden +1/-1 chatter behind the filter.
        self.prev_action_norm = np.zeros(self.action_dim, dtype=np.float64)
        self.prev_action_requested_norm = np.zeros(self.action_dim, dtype=np.float64)
        self._wrench_thrust_filter_initialized = False
        self._last_reward_terms = {}

    @staticmethod
    def _validate_range(name, lo, hi):
        lo, hi = float(lo), float(hi)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo <= 0.0 or hi <= 0.0 or lo > hi:
            raise ValueError(f"{name} range must satisfy 0 < min <= max")

    def _sample_rotor_scale(self, lo, hi):
        if bool(getattr(self.cfg, "per_motor_params", False)):
            return self.rng.uniform(float(lo), float(hi), size=4)
        value = float(self.rng.uniform(float(lo), float(hi)))
        return np.full(4, value, dtype=float)

    def _sample_episode_disturbances(self, k_override=None):
        enabled = set() if self.disturbances == ("none",) else set(self.disturbances)

        if k_override is not None:
            self.k = float(k_override)
        elif "massmoi" in enabled:
            self.k = float(self.rng.uniform(self.cfg.k_min, self.cfg.k_max))
        else:
            self.k = 1.0

        if "force" in enabled:
            fmax = float(self.cfg.external_force_max)
            self.external_force = self.rng.uniform(-fmax, fmax, size=3)
        else:
            self.external_force = np.zeros(3, dtype=float)

        self.motor_coeff_scale = (
            self._sample_rotor_scale(self.cfg.motor_coeff_min, self.cfg.motor_coeff_max)
            if "motor_coeff" in enabled else np.ones(4, dtype=float)
        )
        self.moment_coeff_scale = (
            self._sample_rotor_scale(self.cfg.moment_coeff_min, self.cfg.moment_coeff_max)
            if "moment_coeff" in enabled else np.ones(4, dtype=float)
        )
        self.arm_length_scale = (
            self._sample_rotor_scale(self.cfg.arm_length_min, self.cfg.arm_length_max)
            if "arm_length" in enabled else np.ones(4, dtype=float)
        )

        self.mixer.set_true(
            kf_scale=self.motor_coeff_scale,
            moment_scale=self.moment_coeff_scale,
            arm_x=NOMINAL_ARM * self.arm_length_scale,
        )

    # ------------------------------------------------------------------ reset
    def reset(self, k=None):
        """Start a new episode.

        ``k`` is an optional explicit mass/MOI override used by controlled
        evaluation. Training normally leaves it as ``None`` so the configured
        disturbance set is sampled once per episode.
        """
        self.step_idx = 0
        self.t = 0.0
        self._sample_episode_disturbances(k_override=k)

        self.dyn_true.set_inertial_scale(self.k)
        self.dyn_nom.set_inertial_scale(1.0)

        d0 = self.traj.desired(0.0)
        self.dyn_true.reset(
            x=d0["x"], v=d0["v"], external_force=self.external_force,
        )
        self.dyn_nom.reset(
            x=d0["x"], v=d0["v"], external_force=np.zeros(3),
        )

        self.ctrl_true.reset()
        self.ctrl_nom.reset()
        self.prev_action_norm.fill(0.0)
        self.prev_action_requested_norm.fill(0.0)
        self._wrench_thrust_filter_initialized = False
        self._last_reward_terms = {}
        self.obs_builder.reset()
        return self.obs_builder.get()

    # ------------------------------------------------------------------- step
    def step(self, action):
        """Apply one normalized residual action and advance the twin system.

        ``wrench`` mode uses a 4-D direct [df, dMx, dMy, dMz] residual.
        ``force_vector`` mode uses a 3-D ``delta_A`` in NED/world newtons and
        lets the geometric controller convert the modified force vector into
        desired attitude, collective thrust, and body moment.
        """
        action = np.asarray(action, dtype=np.float64).reshape(self.action_dim)
        action_requested_norm = np.clip(action, -1.0, 1.0)
        if self.residual_interface == "force_vector":
            beta = self.force_vector_filter_beta
            action_norm = (1.0 - beta) * self.prev_action_norm + beta * action_requested_norm
            reward_action_norm = action_norm
            reward_prev_action_norm = self.prev_action_norm
        else:
            # Only the direct-wrench collective-thrust channel is bandwidth
            # limited. Roll/pitch/yaw residual moments remain exactly the raw SAC
            # command. Match the validated diagnostic intervention by seeding the
            # filter from the first requested action rather than from zero.
            action_norm = action_requested_norm.copy()
            beta = self.wrench_thrust_filter_beta
            if not self._wrench_thrust_filter_initialized:
                action_norm[0] = action_requested_norm[0]
            else:
                action_norm[0] = (
                    (1.0 - beta) * self.prev_action_norm[0]
                    + beta * action_requested_norm[0]
                )
            reward_action_norm = action_requested_norm
            reward_prev_action_norm = self.prev_action_requested_norm
        residual = action_norm * self.action_scale
        residual_requested = action_requested_norm * self.action_scale

        desired = self.traj.desired(self.t)
        st = self.dyn_true.state()
        sn = self.dyn_nom.state()

        # Nominal branch is always the untouched nominal closed-loop controller.
        f_n, M_n, nom_ctrl_info = self.ctrl_nom.compute_control(sn, desired)

        if self.residual_interface == "force_vector":
            # Preview the no-residual command for logging / PID-observation use,
            # but do not advance the controller's finite-difference reference
            # history. The actual call below advances history exactly once.
            f_base, M_base, base_ctrl_info = self.ctrl_true.compute_control(
                st, desired, force_residual=None, update_history=False,
            )
            f_cmd, M_cmd, true_ctrl_info = self.ctrl_true.compute_control(
                st, desired, force_residual=residual, update_history=True,
            )
            u_base = np.array([f_base, M_base[0], M_base[1], M_base[2]], dtype=float)
            residual_wrench = None
            residual_force_vector = residual.copy()
        else:
            # Legacy direct-wrench residual: evaluate the geometric controller
            # normally, then add the residual before nominal motor allocation.
            f_base, M_base, true_ctrl_info = self.ctrl_true.compute_control(st, desired)
            f_cmd = f_base + residual[0]
            M_cmd = M_base + residual[1:4]
            u_base = np.array([f_base, M_base[0], M_base[1], M_base[2]], dtype=float)
            base_ctrl_info = true_ctrl_info
            residual_wrench = residual.copy()
            residual_force_vector = None

        # Both plants use the nominal allocator. The true plant reconstructs
        # wrench with B_true; the nominal twin reconstructs with B_nom.
        f_true, M_true, true_mix = self.mixer.apply(f_cmd, M_cmd, return_info=True)
        f_nom, M_nom, nom_mix = self.mixer.apply_nominal(f_n, M_n, return_info=True)

        st_next = self.dyn_true.step(f_true, M_true)
        sn_next = self.dyn_nom.step(f_nom, M_nom)

        self.step_idx += 1
        self.t = self.step_idx * self.cfg.dt

        u_total = np.array([f_true, M_true[0], M_true[1], M_true[2]], dtype=float)
        u_cmd = np.array([f_cmd, M_cmd[0], M_cmd[1], M_cmd[2]], dtype=float)
        if self._pred_ares_mode:
            self.obs_builder.push(st_next, u_total, residual)
        else:
            self.obs_builder.push(sn_next, st_next, action_norm, u_total, u_base)

        reward = self._reward(
            sn_next, st_next, reward_action_norm, reward_prev_action_norm
        )
        self.prev_action_requested_norm = action_requested_norm.copy()
        self.prev_action_norm = action_norm.copy()
        if self.residual_interface == "wrench":
            self._wrench_thrust_filter_initialized = True

        terminated = self._check_terminated(sn_next, st_next)
        truncated = self.step_idx >= self.cfg.episode_steps

        info = {
            "k": self.k,
            "external_force": self.external_force.copy(),
            "motor_coeff_scale": self.motor_coeff_scale.copy(),
            "moment_coeff_scale": self.moment_coeff_scale.copy(),
            "arm_length_scale": self.arm_length_scale.copy(),
            "disturbances": self.disturbances,
            "residual_interface": self.residual_interface,
            "pos_err": float(np.linalg.norm(sn_next["x"] - st_next["x"])),
            "action_requested_norm": action_requested_norm.copy(),
            "action_applied_norm": action_norm.copy(),
            # Generic physical residual. Shape is 4 for wrench mode and 3 for A mode.
            "residual": residual.copy(),
            "residual_requested": residual_requested.copy(),
            "residual_wrench": residual_wrench,
            "residual_requested_wrench": (
                residual_requested.copy() if self.residual_interface == "wrench" else None
            ),
            "residual_force_vector": residual_force_vector,
            "u_base": u_base.copy(),
            "u_cmd": u_cmd.copy(),
            "u_total": u_total.copy(),
            "A_base": np.asarray(base_ctrl_info.get("A_base", np.zeros(3)), dtype=float).copy(),
            "A_cmd": np.asarray(true_ctrl_info.get("A_cmd", np.zeros(3)), dtype=float).copy(),
            "Rd_cmd": np.asarray(true_ctrl_info.get("Rd"), dtype=float).copy(),
            "omega_d_cmd": np.asarray(true_ctrl_info.get("omega_d", np.zeros(3)), dtype=float).copy(),
            # Do not infer saturation from u_cmd != u_total: actuator/geometry
            # mismatch itself changes wrench even when no motor is saturated.
            "actuator_saturated": bool(true_mix["saturated"]),
            "nominal_actuator_saturated": bool(nom_mix["saturated"]),
            "motor_cmd": true_mix["motor_cmd"].copy(),
            "motor_sat": true_mix["motor_sat"].copy(),
            "reward_state": self._last_reward_terms.get("state", 0.0),
            "reward_effort_penalty": self._last_reward_terms.get("effort", 0.0),
            "reward_smooth_penalty": self._last_reward_terms.get("smooth", 0.0),
        }
        return self.obs_builder.get(), reward, terminated, truncated, info

    # -------------------------------------------------------------- internals
    def _reward(self, sn, st, action_norm, prev_action_norm=None):
        c = self.cfg

        ep = (sn["x"] - st["x"]) / c.obs_pos_scale
        ev = (sn["v"] - st["v"]) / c.obs_vel_scale
        eR = rotation_error(sn["R"], st["R"]) / c.obs_att_scale
        ew = (sn["omega"] - st["omega"]) / c.obs_omega_scale

        ep2 = float(np.dot(ep, ep))
        ev2 = float(np.dot(ev, ev))
        eR2 = float(np.dot(eR, eR))
        ew2 = float(np.dot(ew, ew))

        state_reward_raw = (
            c.w_pos * np.exp(-ep2 / c.tau_pos ** 2)
            + c.w_vel * np.exp(-ev2 / c.tau_vel ** 2)
            + c.w_att * np.exp(-eR2 / c.tau_att ** 2)
            + c.w_omega * np.exp(-ew2 / c.tau_omega ** 2)
        )
        state_reward = float(state_reward_raw / c.reward_norm)

        a = np.asarray(action_norm, dtype=np.float64).reshape(self.action_dim)
        if prev_action_norm is None:
            prev_action_norm = self.prev_action_norm
        a_prev = np.asarray(prev_action_norm, dtype=np.float64).reshape(self.action_dim)
        effort_penalty = c.w_action_effort * float(np.linalg.norm(a))
        smooth_penalty = c.w_action_smooth * float(np.linalg.norm(a - a_prev))

        self._last_reward_terms = {
            "state": state_reward,
            "effort": effort_penalty,
            "smooth": smooth_penalty,
        }
        return float(state_reward - effort_penalty - smooth_penalty)

    def _check_terminated(self, sn, st):
        pos_err = np.linalg.norm(sn["x"] - st["x"])
        if pos_err > self.cfg.term_pos_error:
            return True
        cos_tilt = st["R"][2, 2]
        if cos_tilt < self._tilt_cos_thresh:
            return True
        return False
