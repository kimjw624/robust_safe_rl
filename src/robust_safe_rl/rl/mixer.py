"""x500 control-allocation mixer for actuator/geometry disturbance modeling.

The controller requests a body wrench ``w_cmd = [f, Mx, My, Mz]`` using the
NOMINAL vehicle model.  The nominal allocator converts that wrench to nominal
per-rotor thrust commands.  Those commands correspond to motor-speed-squared
commands through ``T_nom = k_f,nom * omega^2``.  The same motor commands are
then interpreted by the TRUE actuator/geometry model:

    w_cmd
      -> nominal allocation:  T_cmd = B_nom^{-1} w_cmd
      -> rotor saturation:    0 <= T_cmd <= k_f,nom * omega_max^2
      -> actual wrench:       w_actual = B_true T_cmd

This is the important physical relationship for coefficient/arm uncertainty:
the flight controller/allocation logic still assumes nominal coefficients and
geometry, while the true vehicle produces a different wrench from those same
motor commands.

Nominal x500 constants used here:
    arm coordinate L = 0.174 m along each body x/y axis,
    k_f = 8.54858e-6 N/(rad/s)^2,
    momentConstant = 0.016, implemented as yaw drag torque / rotor thrust,
    omega_max = 1000 rad/s.

Supported TRUE-model perturbations (scalar or one value per rotor):
    kf_scale       : scales motorConstant / thrust coefficient k_f.  Because
                     yaw drag is momentConstant * thrust, this also scales yaw
                     torque for a fixed momentConstant.
    moment_scale   : independently scales momentConstant, affecting yaw torque
                     only (on top of any k_f scaling).
    arm_x, arm_y   : true rotor x/y moment-arm magnitudes. Scaling both by the
                     same factor models an arm-length/geometry scale and affects
                     roll/pitch moments but not collective thrust or yaw drag.
"""

import numpy as np


# x500 nominal actuator / mixer constants.
NOMINAL_ARM = 0.174           # m, per-axis rotor position
KF = 8.54858e-06              # N/(rad/s)^2
KM_RATIO = 0.016              # momentConstant: yaw drag torque / rotor thrust
MAX_ROT_VELOCITY = 1000.0     # rad/s

# The mixer uses nominal rotor thrust [N] as its motor-command coordinate. This
# is k_f,nom * omega^2, so the physical motor-speed limit maps to this fixed cap
# even when the TRUE k_f is uncertain.
MAX_MOTOR_THRUST = KF * MAX_ROT_VELOCITY ** 2
F_MAX = 4.0 * MAX_MOTOR_THRUST
MX_MAX = 2.0 * NOMINAL_ARM * MAX_MOTOR_THRUST
MY_MAX = MX_MAX
MZ_MAX = 2.0 * KM_RATIO * MAX_MOTOR_THRUST
M_MAX = np.array([MX_MAX, MY_MAX, MZ_MAX], dtype=float)

# Rotor geometry in motor-number order:
#   0: (+x, -y), CCW
#   1: (-x, +y), CCW
#   2: (+x, +y), CW
#   3: (-x, -y), CW
# spin +1 = CCW, -1 = CW.
_ROTOR_SIGNS = [(+1, -1, +1), (-1, +1, +1), (+1, +1, -1), (-1, -1, -1)]


def _rotor_array(value, name, default=1.0):
    """Return a finite length-4 rotor parameter array."""
    if value is None:
        arr = np.full(4, float(default), dtype=float)
    else:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size == 1:
            arr = np.full(4, float(arr[0]), dtype=float)
        elif arr.size != 4:
            raise ValueError(f"{name} must be a scalar or length-4 array")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def clip_wrench(f, M):
    """Component-wise nominal-x500 wrench guard for code paths bypassing allocation.

    This is only a box constraint. Coupled actuator feasibility is handled by
    :class:`Mixer`, which clips the individual rotor commands instead.
    """
    f_sat = float(np.clip(float(f), 0.0, F_MAX))
    M = np.asarray(M, dtype=float).reshape(3)
    M_sat = np.clip(M, -M_MAX, M_MAX)
    return f_sat, M_sat.copy()


def build_mixer(arm_x, arm_y=None, kf_scale=None, moment_scale=None,
                km_ratio=KM_RATIO):
    """Build ``B`` mapping nominal rotor-thrust commands to body wrench.

    Parameters
    ----------
    arm_x, arm_y:
        Scalar or length-4 positive rotor-coordinate magnitudes [m]. ``arm_y``
        defaults to ``arm_x``. Rotor quadrant signs are applied internally.
    kf_scale:
        Scalar or length-4 scale on the TRUE rotor thrust coefficient k_f.
    moment_scale:
        Scalar or length-4 scale on ``momentConstant`` (yaw torque / thrust).
        A k_f change still changes yaw torque because yaw torque is proportional
        to the actual rotor thrust; ``moment_scale`` changes the ratio itself.

    Returns
    -------
    B : ndarray, shape (4, 4)
        ``[f, Mx, My, Mz]^T = B @ T_nom_cmd``, where ``T_nom_cmd`` is the
        nominal-equivalent per-rotor thrust ``k_f,nom * omega^2``.
    """
    ax = _rotor_array(arm_x, "arm_x")
    ay = ax.copy() if arm_y is None else _rotor_array(arm_y, "arm_y")
    kf = _rotor_array(kf_scale, "kf_scale", default=1.0)
    km = _rotor_array(moment_scale, "moment_scale", default=1.0)

    if np.any(ax < 0.0) or np.any(ay < 0.0):
        raise ValueError("arm magnitudes must be nonnegative")
    if np.any(kf < 0.0) or np.any(km < 0.0):
        raise ValueError("coefficient scales must be nonnegative")

    B = np.zeros((4, 4), dtype=float)
    for i, (sx, sy, spin) in enumerate(_ROTOR_SIGNS):
        px, py = sx * ax[i], sy * ay[i]
        thrust_gain = kf[i]
        B[0, i] = thrust_gain
        B[1, i] = thrust_gain * py
        B[2, i] = -thrust_gain * px
        B[3, i] = -spin * km_ratio * km[i] * thrust_gain
    return B


class Mixer:
    """Nominal allocator plus a configurable true actuator/geometry model.

    ``max_motor_thrust`` defaults to the physical x500 limit.  The argument is
    exposed primarily for controlled diagnostic experiments: increasing or
    decreasing actuator headroom lets us test whether clipping/saturation is a
    *necessary* trigger for a closed-loop oscillation.  Normal training and
    evaluation code does not pass this argument, so its behavior is unchanged.
    """

    def __init__(self, max_motor_thrust=MAX_MOTOR_THRUST):
        self.B_nom = build_mixer(NOMINAL_ARM)
        self.B_nom_inv = np.linalg.inv(self.B_nom)
        self.B_true = self.B_nom.copy()
        self.max_motor_thrust = float(max_motor_thrust)
        if not np.isfinite(self.max_motor_thrust) or self.max_motor_thrust <= 0.0:
            raise ValueError("max_motor_thrust must be finite and positive")

    def set_motor_thrust_limit(self, max_motor_thrust):
        """Set the per-rotor command limit used by the nominal allocator.

        This is a diagnostic hook.  The true/nominal mixer matrices are not
        modified; only the actuator clipping boundary changes.
        """
        value = float(max_motor_thrust)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("max_motor_thrust must be finite and positive")
        self.max_motor_thrust = value
        return self

    def set_true(self, kf_scale=None, moment_scale=None, arm_x=None, arm_y=None):
        """Set TRUE actuator/geometry parameters while keeping allocation nominal.

        ``kf_scale`` and ``moment_scale`` may be scalars or length-4 arrays.
        ``arm_x``/``arm_y`` are true positive coordinate magnitudes in metres;
        when only ``arm_x`` is provided, the same magnitudes are used on y so a
        scalar/per-rotor arm scale preserves each rotor's radial direction.
        """
        ax = NOMINAL_ARM if arm_x is None else arm_x
        self.B_true = build_mixer(
            ax,
            arm_y,
            kf_scale=kf_scale,
            moment_scale=moment_scale,
        )
        return self

    def reset_true(self):
        self.B_true = self.B_nom.copy()
        return self

    def allocate(self, f, M):
        """Allocate a requested wrench with the NOMINAL mixer and saturate rotors."""
        M = np.asarray(M, dtype=float).reshape(3)
        w_cmd = np.array([float(f), M[0], M[1], M[2]], dtype=float)
        motor_cmd = self.B_nom_inv @ w_cmd
        motor_sat = np.clip(motor_cmd, 0.0, self.max_motor_thrust)
        saturated = bool(not np.allclose(motor_cmd, motor_sat, rtol=0.0, atol=1e-12))
        return w_cmd, motor_cmd, motor_sat, saturated

    def _apply_matrix(self, f, M, B, return_info=False):
        w_cmd, motor_cmd, motor_sat, saturated = self.allocate(f, M)
        w_actual = B @ motor_sat
        result = (float(w_actual[0]), w_actual[1:4].copy())
        if not return_info:
            return result
        info = {
            "w_cmd": w_cmd.copy(),
            "motor_cmd": motor_cmd.copy(),
            "motor_sat": motor_sat.copy(),
            "saturated": saturated,
            "w_actual": w_actual.copy(),
        }
        return result[0], result[1], info

    def apply(self, f, M, return_info=False):
        """Apply nominal allocation followed by the TRUE actuator/geometry model."""
        return self._apply_matrix(f, M, self.B_true, return_info=return_info)

    def apply_nominal(self, f, M, return_info=False):
        """Apply nominal allocation and reconstruct with the NOMINAL mixer.

        This is useful for the reference twin: it receives the same physical
        rotor saturation model but no actuator/geometry parameter mismatch.
        """
        return self._apply_matrix(f, M, self.B_nom, return_info=return_info)
