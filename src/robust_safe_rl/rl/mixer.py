"""x500 control-allocation mixer, for motor/geometry disturbance modeling.

Physically-faithful disturbance loop (matches how sim-to-real parameter error
actually enters):

    controller wants (f, M)
      -> allocate to motor thrusts with the NOMINAL mixer:  u = B_nom^{-1} (f,M)
      -> reconstruct the actual wrench with the TRUE mixer:  (f,M)_actual = B_true u
      -> feed (f,M)_actual to the dynamics

The residual/disturbance is the gap opened by B_true != B_nom. When the true
parameters equal nominal, the round-trip is the identity and the residual is
exactly zero.

Geometry (x500, physics-faithful from the gz model):
    arm position L = 0.174 m along EACH body axis (symmetric X quad),
    k_f = 8.54858e-6 N/(rad/s)^2 (motor thrust coefficient),
    km_ratio = 0.016 (drag-torque-to-thrust ratio).

Note the arm value: 0.174 is the gz model's per-axis rotor position, not the
PX4 allocator's (inconsistent) 0.13/0.22, and not the 0.246 diagonal.

Disturbance parameters (perturb the TRUE mixer only):
    k_f scale   : motor strength. Global (scalar) or per-motor (length-4).
                  "thrust factor" scales k_f, which affects BOTH f and M through
                  the allocation.
    arm scale   : rotor arm position. Global (scalar) or per-rotor (length-4);
                  scales the moment arms, so it affects M but not f.
"""

import numpy as np


# x500 nominal mixer constants
NOMINAL_ARM = 0.174           # m, per-axis rotor position (gz model)
KF = 8.54858e-06              # N/(rad/s)^2
KM_RATIO = 0.016              # drag torque = KM_RATIO * thrust

# rotor sign pattern for the X configuration: (px_sign, py_sign, spin)
# spin +1 = CCW, -1 = CW; motors 0..3
_ROTOR_SIGNS = [(+1, +1, +1), (-1, -1, +1), (+1, -1, -1), (-1, +1, -1)]


def build_mixer(arm_x, arm_y=None, kf_scale=None, km_ratio=KM_RATIO):
    """Build the 4x4 mixer B mapping motor COMMANDS to wrench [f, Mx, My, Mz].

    arm_x, arm_y : scalar or length-4 arm positions (m). arm_y defaults to arm_x.
    kf_scale     : scalar or length-4 multiplier on each motor's thrust coeff
                   (1.0 = nominal). Scales that motor's contribution to f and M.
    Returns B such that (f, Mx, My, Mz)^T = B @ motor_commands.
    """
    arm_x = np.atleast_1d(np.asarray(arm_x, dtype=float))
    arm_y = arm_x if arm_y is None else np.atleast_1d(np.asarray(arm_y, dtype=float))
    if arm_x.size == 1:
        arm_x = np.full(4, arm_x[0])
    if arm_y.size == 1:
        arm_y = np.full(4, arm_y[0])
    if kf_scale is None:
        kf = np.ones(4)
    else:
        kf = np.atleast_1d(np.asarray(kf_scale, dtype=float))
        if kf.size == 1:
            kf = np.full(4, kf[0])

    B = np.zeros((4, 4))
    for i, (sx, sy, spin) in enumerate(_ROTOR_SIGNS):
        px, py = sx * arm_x[i], sy * arm_y[i]
        B[0, i] = kf[i]                 # thrust
        B[1, i] = kf[i] * py            # roll  Mx
        B[2, i] = -kf[i] * px           # pitch My
        B[3, i] = -spin * km_ratio * kf[i]   # yaw Mz (rotor drag reaction)
    return B


class Mixer:
    """Nominal/true mixer pair implementing the allocate-then-reconstruct loop."""

    def __init__(self):
        self.B_nom = build_mixer(NOMINAL_ARM)
        self.B_nom_inv = np.linalg.inv(self.B_nom)
        self.B_true = self.B_nom.copy()

    def set_true(self, kf_scale=None, arm_x=None, arm_y=None):
        """Set the TRUE mixer's perturbed parameters (nominal stays fixed).

        kf_scale : scalar or length-4 motor-strength multiplier (thrust factor).
        arm_x/y  : scalar or length-4 arm positions; default to NOMINAL_ARM.
        """
        ax = NOMINAL_ARM if arm_x is None else arm_x
        ay = arm_y
        self.B_true = build_mixer(ax, ay, kf_scale=kf_scale)
        return self

    def reset_true(self):
        self.B_true = self.B_nom.copy()
        return self

    def apply(self, f, M):
        """Allocate (f,M) with the nominal mixer, reconstruct with the true one.

        Returns (f_actual, M_actual) -- the wrench the plant actually experiences.
        Identity when B_true == B_nom.
        """
        w = np.array([f, M[0], M[1], M[2]], dtype=float)
        motors = self.B_nom_inv @ w          # what the controller commands
        w_actual = self.B_true @ motors      # what the plant produces
        return float(w_actual[0]), w_actual[1:4].copy()