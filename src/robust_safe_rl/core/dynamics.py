"""Minimal quadrotor rigid-body dynamics in NED coordinates.

The equations of motion follow the SE(3) quadrotor model of Lee, Leok, and
McClamroch (CDC 2010), expressed here in a North-East-Down frame:

    x_dot     = v
    v_dot     = g e3 - (f / m) R e3 + f_ext / m
    R_dot     = R hat(omega)
    omega_dot = J^-1 (M - omega x J omega)

Integration is done with a fixed-step RK4 scheme. An optional external force
disturbance models wind or unmodeled aerodynamic effects, and can be used to
generate in-distribution vs. out-of-distribution episodes for OOD detection.
"""

import numpy as np

from .so3 import hat, project_to_so3


class Dynamics:
    """NED quadrotor dynamics with an optional external force disturbance.

    random_force:
        0: no external force.
        1: in-distribution force, each axis uniform in [-3, 3] N.
        2: loose OOD force, each axis uniform in [-5, 5] N (can overlap ID).
        3: strict OOD force, sampled in [-5, 5] N but requiring at least one
           axis to exceed the ID limit of 3 N.

    By default the force is sampled once at reset and held constant for the whole
    episode. Set ``force_sample_each_step=True`` to resample every step.
    """

    def __init__(
        self,
        dt=0.01,
        mass=2.0,
        J=None,
        gravity=9.807,
        random_force=0,
        force_sample_each_step=False,
        seed=None,
    ):
        self.dt = float(dt)
        self.mass = float(mass)
        self.J = np.diag([0.022, 0.022, 0.04]) if J is None else np.asarray(J, dtype=float)
        self.gravity = float(gravity)

        self.random_force = int(random_force)
        self.force_sample_each_step = bool(force_sample_each_step)

        self.rng = np.random.default_rng(seed)

        self.e3 = np.array([0.0, 0.0, 1.0])

        self.external_force = np.zeros(3)
        self.last_external_force = np.zeros(3)

        self.reset()

    def reset(self, x=None, v=None, R=None, omega=None, external_force=None):
        self.x = np.zeros(3) if x is None else np.asarray(x, dtype=float).reshape(3)
        self.v = np.zeros(3) if v is None else np.asarray(v, dtype=float).reshape(3)
        self.R = np.eye(3) if R is None else project_to_so3(np.asarray(R, dtype=float).reshape(3, 3))
        self.omega = np.zeros(3) if omega is None else np.asarray(omega, dtype=float).reshape(3)

        if external_force is None:
            self.external_force = self.sample_external_force()
        else:
            self.external_force = np.asarray(external_force, dtype=float).reshape(3)

        self.last_external_force = self.external_force.copy()

        return self.state()

    def state(self):
        return {
            "x": self.x.copy(),
            "v": self.v.copy(),
            "R": self.R.copy(),
            "omega": self.omega.copy(),
        }

    def sample_external_force(self):
        if self.random_force == 0:
            return np.zeros(3)

        if self.random_force == 1:
            # In-distribution force. Each axis is inside [-3, 3] N.
            return self.rng.uniform(-3.0, 3.0, size=3)

        if self.random_force == 2:
            # Loose OOD force. Each axis is inside [-5, 5] N.
            # Note: this can still generate forces inside [-3, 3].
            return self.rng.uniform(-5.0, 5.0, size=3)

        if self.random_force == 3:
            # Strict OOD force. Sample from [-5, 5] N, but require at least one
            # axis to be outside the ID limit of [-3, 3] N.
            while True:
                force = self.rng.uniform(-5.0, 5.0, size=3)

                if np.any(np.abs(force) > 3.0):
                    return force

        raise ValueError("random_force must be 0, 1, 2, or 3")

    def derivative(self, y, f, M, external_force):
        v = y[3:6]
        R = y[6:15].reshape(3, 3)
        omega = y[15:18]

        M = np.asarray(M, dtype=float).reshape(3)
        external_force = np.asarray(external_force, dtype=float).reshape(3)

        x_dot = v

        # NED dynamics:
        # v_dot = g e3 - f/m R e3 + disturbance_force/m
        v_dot = (
            self.gravity * self.e3
            - (float(f) / self.mass) * (R @ self.e3)
            + external_force / self.mass
        )

        R_dot = R @ hat(omega)

        omega_dot = np.linalg.solve(
            self.J,
            M - np.cross(omega, self.J @ omega),
        )

        return np.concatenate([
            x_dot,
            v_dot,
            R_dot.reshape(9),
            omega_dot,
        ])

    def step(self, f, M):
        if self.force_sample_each_step:
            self.external_force = self.sample_external_force()

        external_force = self.external_force.copy()
        self.last_external_force = external_force.copy()

        y = np.concatenate([
            self.x,
            self.v,
            self.R.reshape(9),
            self.omega,
        ])

        h = self.dt

        k1 = self.derivative(y, f, M, external_force)
        k2 = self.derivative(y + 0.5 * h * k1, f, M, external_force)
        k3 = self.derivative(y + 0.5 * h * k2, f, M, external_force)
        k4 = self.derivative(y + h * k3, f, M, external_force)

        y_next = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.x = y_next[0:3]
        self.v = y_next[3:6]
        self.R = project_to_so3(y_next[6:15].reshape(3, 3))
        self.omega = y_next[15:18]

        return self.state()
