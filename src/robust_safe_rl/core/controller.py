"""Geometric SE(3) tracking controller for the NED quadrotor.

This is the baseline (nominal) controller. It follows the structure of Lee,
Leok, and McClamroch (CDC 2010), adapted to the NED convention used by
:class:`~robust_safe_rl.core.dynamics.Dynamics`. In the residual-RL framework
this controller is the base policy that a learned residual is added on top of.

The desired angular velocity and its derivative are obtained by finite-
differencing the desired attitude ``Rd``, so the first one or two steps after a
reset use zero feedforward until the history fills in.
"""

import numpy as np

from .so3 import hat, vee, normalize, project_to_so3, rotation_error, so3_log


class Controller:
    """Minimal geometric SE(3) controller for the NED dynamics."""

    def __init__(self, dt=0.01, mass=2.0, J=None, gravity=9.807):
        self.dt = float(dt)
        self.mass = float(mass)
        self.J = np.diag([0.022, 0.022, 0.04]) if J is None else np.asarray(J, dtype=float)
        self.gravity = float(gravity)

        # kx and kv are already mass-scaled, matching the original gain style.
        self.kx = self.mass * 18.0 * np.eye(3)
        self.kv = self.mass * 10.0 * np.eye(3)
        self.kr = 8.81
        self.komega = 2.54

        self.e3 = np.array([0.0, 0.0, 1.0])

        self.Rd_prev = None
        self.omega_d_prev = np.zeros(3)

    def reset(self):
        self.Rd_prev = None
        self.omega_d_prev = np.zeros(3)

    def compute_control(self, state, desired):
        x = state["x"]
        v = state["v"]
        R = state["R"]
        omega = state["omega"]

        xd = desired["x"]
        vd = desired["v"]
        ad = desired["a"]
        b1d = desired["b1d"]

        ex = x - xd
        ev = v - vd

        # NED geometric controller:
        # A = kx ex + kv ev + m g e3 - m ad
        A = self.kx @ ex + self.kv @ ev + self.mass * self.gravity * self.e3 - self.mass * ad

        # Total thrust.
        f = float(A @ (R @ self.e3))

        # Keep thrust physically nonnegative.
        f = max(0.0, f)

        # Desired attitude. Desired body z-axis aligns with A.
        b3d = normalize(A, fallback=self.e3)
        b2d = normalize(
            np.cross(b3d, b1d),
            fallback=np.array([0.0, 1.0, 0.0]),
        )
        b1d_real = np.cross(b2d, b3d)

        Rd = project_to_so3(np.column_stack((b1d_real, b2d, b3d)))

        # Desired angular velocity by finite difference of Rd.
        if self.Rd_prev is None:
            omega_d = np.zeros(3)
            omega_d_dot = np.zeros(3)
        else:
            omega_d = vee(so3_log(self.Rd_prev.T @ Rd)) / self.dt
            omega_d_dot = (omega_d - self.omega_d_prev) / self.dt

        self.Rd_prev = Rd.copy()
        self.omega_d_prev = omega_d.copy()

        eR = rotation_error(R, Rd)
        eOmega = omega - R.T @ Rd @ omega_d

        M = (
            -self.kr * eR
            - self.komega * eOmega
            + np.cross(omega, self.J @ omega)
            - self.J @ (
                hat(omega) @ R.T @ Rd @ omega_d
                - R.T @ Rd @ omega_d_dot
            )
        )

        info = {
            "ex": ex,
            "ev": ev,
            "eR": eR,
            "eOmega": eOmega,
            "Rd": Rd,
        }

        return f, M, info
