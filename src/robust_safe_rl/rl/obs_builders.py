"""Observation builders for the residual environment.

Three modes, all computed from the nominal-minus-true twin discrepancy so the
disturbance-observer framing is preserved:

  * "history"  : the original H-frame stack of
                 [discrepancy(12), residual(4), u_total(4)] -> H*20 dims.
                 This is the baseline; the discrepancy already contains P and D
                 information (position P = dx, position D = dv, attitude P = e_R,
                 attitude D = domega), stacked over time.

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


# ---- history mode frame layout (unchanged from the original env) ----
FRAME_STATE = 12          # dx(3) dv(3) e_R(3) domega(3)
FRAME_ACTION = 4          # residual
FRAME_TOTAL_U = 4         # total control applied to the true plant
FRAME_DIM = FRAME_STATE + FRAME_ACTION + FRAME_TOTAL_U  # 20

# ---- pid mode block sizes ----
PID_BLOCK = 9             # P(3) + I(3) + D(3) per tracked group
N_PID_GROUPS = 2          # position, attitude
UBASE_DIM = 4


def obs_dim_for(cfg):
    """Return the flattened observation dimension for the configured mode."""
    mode = cfg.obs_mode
    if mode == "history":
        return cfg.history * FRAME_DIM
    if mode == "pid":
        return N_PID_GROUPS * PID_BLOCK + UBASE_DIM               # 22
    if mode == "pid_hist":
        return N_PID_GROUPS * PID_BLOCK + UBASE_DIM + cfg.pid_hist_frames * FRAME_STATE
    raise ValueError(f"unknown obs_mode: {mode!r}")


def discrepancy(sn, st):
    """Nominal-minus-true discrepancy vector [dx, dv, e_R, domega] (12,)."""
    dx = sn["x"] - st["x"]
    dv = sn["v"] - st["v"]
    e_R = rotation_error(st["R"], sn["R"])       # attitude discrepancy (true vs nominal)
    dw = sn["omega"] - st["omega"]
    return np.concatenate([dx, dv, e_R, dw]).astype(np.float64)


class ObservationBuilder:
    """Stateful per-episode observation builder for one of the three modes.

    Call reset() at episode start, then push() once per step with the freshly
    stepped states and the control that was applied; get() returns the current
    observation vector.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.mode = cfg.obs_mode
        self.dt = cfg.dt
        self.leak = cfg.pid_integral_leak
        self.obs_dim = obs_dim_for(cfg)
        self.reset()

    def reset(self):
        # history-mode ring buffer
        self._frames = np.zeros((self.cfg.history, FRAME_DIM), dtype=np.float64)
        # pid-mode integral accumulators (position, attitude)
        self._int_pos = np.zeros(3, dtype=np.float64)
        self._int_att = np.zeros(3, dtype=np.float64)
        # pid_hist short tail
        self._tail = np.zeros((self.cfg.pid_hist_frames, FRAME_STATE), dtype=np.float64)
        # latest bits needed to assemble a pid observation
        self._last_disc = np.zeros(FRAME_STATE, dtype=np.float64)
        self._last_ubase = np.zeros(UBASE_DIM, dtype=np.float64)

    def push(self, sn, st, residual, u_total, u_base):
        """Record one step. sn/st are nominal/true post-step states; residual and
        u_total are the applied wrench pieces; u_base is the base control (4,)."""
        disc = discrepancy(sn, st)
        self._last_disc = disc
        self._last_ubase = np.asarray(u_base, dtype=np.float64).reshape(UBASE_DIM)

        # history frame
        frame = np.concatenate([disc, residual, u_total]).astype(np.float64)
        self._frames = np.roll(self._frames, -1, axis=0)
        self._frames[-1] = frame

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
            return self._frames.reshape(-1).astype(np.float32)

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