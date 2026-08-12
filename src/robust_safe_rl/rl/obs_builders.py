"""Observation builders for the residual environment.

Three modes, all computed from the nominal-minus-true twin discrepancy so the
disturbance-observer framing is preserved:

  * "history"  : a causal interleaved history
                 [e~_{t-H+1}, a_{t-H+1}, ..., e~_{t-1}, a_{t-1}, e~_t],
                 where e~ is the 12-D zero-centered, fixed-scale normalized twin
                 discrepancy and each past action is the normalized command that
                 actually reached the residual interface in [-1, 1]^d. This equals
                 the raw SAC action when no command filter is active. The current
                 action a_t is deliberately NOT part of s_t;
                 it is chosen from s_t and enters the critic separately as
                 Q(s_t,a_t). For H=10 this is 156-D for the legacy 4-D wrench
                 residual and 147-D for the 3-D force-vector residual.

  * "pid"      : a single frame of PID errors on position and attitude plus the
                 base control:
                   position  P = dx,  D = dv,  I = leaky-integral of dx
                   attitude  P = e_R, D = domega, I = leaky-integral of e_R
                   u_base (4)
                 -> 9 + 9 + 4 = 22 dims. The only genuinely new signal versus the
                 baseline is the leaky integral; P and D are reused from the
                 discrepancy states directly (no numerical differencing).

  * "pid_hist" : the 22 PID dims plus a short tail of the last `pid_hist_frames`
                 raw discrepancy vectors (12 each), to test whether history
                 carries disturbance-identification information beyond PID.
                 -> 22 + 12*pid_hist_frames dims.

The D terms use the velocity / angular-rate discrepancy states directly
(dv, domega), which are the clean derivatives of the position / attitude errors
and avoid differencing noise. The integral is leaky (I <- lambda*I + e*dt) for
anti-windup: under actuator saturation a plain integral would grow without
bound, but the leak keeps it bounded and well-scaled for the network. The
integral state is reset to zero every episode.
"""

import numpy as np

from robust_safe_rl.core.so3 import rotation_error


# ---- history mode layout ----
FRAME_STATE = 12          # discrepancy e: dx(3) dv(3) e_R(3) domega(3)
FRAME_ACTION = 4          # legacy/default normalized residual SAC action dimension

# ---- pid mode block sizes ----
PID_BLOCK = 9             # P(3) + I(3) + D(3) per tracked group
N_PID_GROUPS = 2          # position, attitude
UBASE_DIM = 4


def action_dim_for(cfg):
    """Residual action dimension implied by the configured control interface."""
    interface = str(getattr(cfg, "residual_interface", "wrench"))
    if interface == "wrench":
        return 4
    if interface == "force_vector":
        return 3
    raise ValueError(f"unknown residual_interface: {interface!r}")


def obs_dim_for(cfg, action_dim=None):
    """Return the flattened observation dimension for the configured mode."""
    mode = cfg.obs_mode
    action_dim = action_dim_for(cfg) if action_dim is None else int(action_dim)
    if action_dim < 1:
        raise ValueError("action_dim must be >= 1")
    if mode == "history":
        if cfg.history < 1:
            raise ValueError("history must be >= 1")
        # H errors and only H-1 past actions: the current action is not known yet.
        return cfg.history * FRAME_STATE + (cfg.history - 1) * action_dim
    if mode == "pid":
        return N_PID_GROUPS * PID_BLOCK + UBASE_DIM               # 22
    if mode == "pid_hist":
        return N_PID_GROUPS * PID_BLOCK + UBASE_DIM + cfg.pid_hist_frames * FRAME_STATE
    raise ValueError(f"unknown obs_mode: {mode!r}")


def discrepancy(sn, st):
    """Raw nominal-minus-true discrepancy [dx, dv, e_R, domega] (12,)."""
    dx = sn["x"] - st["x"]
    dv = sn["v"] - st["v"]
    # Keep the same nominal-minus-true sign convention as dx, dv, and dw.
    # rotation_error(R, Rd) is locally current-minus-desired, so pass
    # (nominal, true) to obtain nominal-minus-true.
    e_R = rotation_error(sn["R"], st["R"])
    dw = sn["omega"] - st["omega"]
    return np.concatenate([dx, dv, e_R, dw]).astype(np.float64)


def normalize_discrepancy(disc, cfg):
    """Fixed zero-centered physical scaling used by the SAC history state."""
    scales = np.array(
        [cfg.obs_pos_scale] * 3
        + [cfg.obs_vel_scale] * 3
        + [cfg.obs_att_scale] * 3
        + [cfg.obs_omega_scale] * 3,
        dtype=np.float64,
    )
    if np.any(scales <= 0.0):
        raise ValueError("observation scales must all be positive")
    return np.asarray(disc, dtype=np.float64) / scales


class ObservationBuilder:
    """Stateful per-episode observation builder for one of the three modes.

    Call reset() at episode start, then push() once per step with the freshly
    stepped states and the control that was applied; get() returns the current
    observation vector.
    """

    def __init__(self, cfg, action_dim=None):
        self.cfg = cfg
        self.mode = cfg.obs_mode
        self.dt = cfg.dt
        self.leak = cfg.pid_integral_leak
        self.action_dim = action_dim_for(cfg) if action_dim is None else int(action_dim)
        self.obs_dim = obs_dim_for(cfg, action_dim=self.action_dim)
        self.reset()

    def reset(self):
        # history-mode buffers. There is one more error than action because
        # s_t ends in the current discrepancy e_t; a_t has not been selected yet.
        self._errors = np.zeros((self.cfg.history, FRAME_STATE), dtype=np.float64)
        self._actions = np.zeros((max(self.cfg.history - 1, 0), self.action_dim), dtype=np.float64)
        # pid-mode integral accumulators (position, attitude)
        self._int_pos = np.zeros(3, dtype=np.float64)
        self._int_att = np.zeros(3, dtype=np.float64)
        # pid_hist short tail
        self._tail = np.zeros((self.cfg.pid_hist_frames, FRAME_STATE), dtype=np.float64)
        # latest bits needed to assemble a pid observation
        self._last_disc = np.zeros(FRAME_STATE, dtype=np.float64)
        self._last_ubase = np.zeros(UBASE_DIM, dtype=np.float64)

    def push(self, sn, st, action, u_total, u_base):
        """Record one completed transition.

        ``action`` is the normalized residual command actually applied at step
        t in [-1, 1]^d. It is stored only as a *past* action for the next
        observation s_{t+1}. With command dynamics this is intentionally the
        applied command, not the hidden raw SAC request. ``u_total``
        and ``u_base`` are physical wrench quantities used by other observation
        modes; history mode does not include them.
        """
        disc = discrepancy(sn, st)
        disc_norm = normalize_discrepancy(disc, self.cfg)
        self._last_disc = disc
        self._last_ubase = np.asarray(u_base, dtype=np.float64).reshape(UBASE_DIM)

        # Causal history update. This push happens after action a_t has been
        # applied and the new discrepancy e_{t+1} is available. At the next
        # decision time, a_t is therefore a past action and may enter s_{t+1}.
        self._errors = np.roll(self._errors, -1, axis=0)
        self._errors[-1] = disc_norm
        if self.cfg.history > 1:
            self._actions = np.roll(self._actions, -1, axis=0)
            self._actions[-1] = np.asarray(action, dtype=np.float64).reshape(self.action_dim)

        # leaky integrals on position (dx) and attitude (e_R) errors
        dx = disc[0:3]
        e_R = disc[6:9]
        self._int_pos = self.leak * self._int_pos + dx * self.dt
        self._int_att = self.leak * self._int_att + e_R * self.dt

        # pid_hist tail
        if self.cfg.pid_hist_frames > 0:
            self._tail = np.roll(self._tail, -1, axis=0)
            self._tail[-1] = disc

    def get(self):
        if self.mode == "history":
            # Interleave each historical error with the action taken from that
            # state, then append the current error with no current action:
            # [e_{t-H+1}, a_{t-H+1}, ..., e_{t-1}, a_{t-1}, e_t].
            pieces = []
            for i in range(self.cfg.history - 1):
                pieces.extend((self._errors[i], self._actions[i]))
            pieces.append(self._errors[-1])
            return np.concatenate(pieces).astype(np.float32)

        # pid / pid_hist share the 22-dim core
        dx = self._last_disc[0:3]
        dv = self._last_disc[3:6]      # position D (velocity discrepancy)
        e_R = self._last_disc[6:9]
        dw = self._last_disc[9:12]     # attitude D (angular-rate discrepancy)

        pos_pid = np.concatenate([dx, self._int_pos, dv])      # P, I, D
        att_pid = np.concatenate([e_R, self._int_att, dw])     # P, I, D
        core = np.concatenate([pos_pid, att_pid, self._last_ubase])

        if self.mode == "pid":
            return core.astype(np.float32)
        if self.mode == "pid_hist":
            return np.concatenate([core, self._tail.reshape(-1)]).astype(np.float32)
        raise ValueError(f"unknown obs_mode: {self.mode!r}")