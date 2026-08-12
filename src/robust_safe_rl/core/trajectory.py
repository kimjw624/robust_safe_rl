"""Analytic desired trajectories for the NED quadrotor simulation.

The controller consumes desired position, velocity, acceleration, and an
inertial-frame heading direction ``b1d``.  Every trajectory therefore returns

    {"x": ..., "v": ..., "a": ..., "b1d": ...}

Coordinates use North-East-Down (NED).  Positive z is Down.
"""

import numpy as np


class DesiredTrajectory:
    """Generate figure-eight, hover, or circular reference trajectories.

    The default is the 10 s three-dimensional figure-eight used in the supplied
    reference:

        x = a sin(w t)
        y = (b/2) sin(2 w t)
        z = z0 + z_amp sin(w t)

    with ``a=b=z_amp=1``, ``w=2*pi/10``, and ``z0=0``.

    ``radius`` and ``speed`` are used only by the circle trajectory.  ``z0`` is
    shared by all modes.  Desired yaw is fixed at zero through
    ``b1d=[1, 0, 0]``.
    """

    _ALIASES = {
        "circle": "circle",
        "hover": "hover",
        "hovering": "hover",
        "figure8": "figure8",
        "figure_8": "figure8",
        "figure-8": "figure8",
        "figure eight": "figure8",
    }

    def __init__(
        self,
        radius=0.79,
        speed=0.5,
        z0=0.0,
        trajectory_type="figure8",
        figure8_a=1.0,
        figure8_b=1.0,
        figure8_z_amp=1.0,
        figure8_omega=2.0 * np.pi / 10.0,
    ):
        self.radius = float(radius)
        self.speed = float(speed)
        self.z0 = float(z0)
        self.figure8_a = float(figure8_a)
        self.figure8_b = float(figure8_b)
        self.figure8_z_amp = float(figure8_z_amp)
        self.figure8_omega = float(figure8_omega)

        key = str(trajectory_type).strip().lower()
        if key not in self._ALIASES:
            valid = ", ".join(sorted({"circle", "hover", "figure8"}))
            raise ValueError(
                f"Unknown trajectory_type={trajectory_type!r}. Expected one of: {valid}."
            )
        self.trajectory_type = self._ALIASES[key]

        scalar_values = {
            "radius": self.radius,
            "speed": self.speed,
            "z0": self.z0,
            "figure8_a": self.figure8_a,
            "figure8_b": self.figure8_b,
            "figure8_z_amp": self.figure8_z_amp,
            "figure8_omega": self.figure8_omega,
        }
        for name, value in scalar_values.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if self.trajectory_type == "circle" and self.radius <= 0.0:
            raise ValueError("radius must be > 0 for a circle trajectory")
        if self.trajectory_type == "figure8" and self.figure8_omega < 0.0:
            raise ValueError("figure8_omega must be >= 0")

        self._b1d = np.array([1.0, 0.0, 0.0], dtype=float)

    def desired(self, t):
        """Return desired position, velocity, acceleration, and heading at time t."""
        t = float(t)
        if not np.isfinite(t):
            raise ValueError("t must be finite")

        if self.trajectory_type == "hover":
            return self._hover()
        if self.trajectory_type == "circle":
            return self._circle(t)
        return self._figure8(t)

    def _pack(self, x, v, a):
        return {
            "x": np.asarray(x, dtype=float),
            "v": np.asarray(v, dtype=float),
            "a": np.asarray(a, dtype=float),
            "b1d": self._b1d.copy(),
        }

    def _hover(self):
        x = np.array([0.0, 0.0, self.z0])
        v = np.zeros(3)
        a = np.zeros(3)
        return self._pack(x, v, a)

    def _circle(self, t):
        # Existing circle retained unchanged.  It starts at (0, 0, z0), moves
        # North, and circles around the centre (0, radius, z0).
        w = self.speed / self.radius
        s = np.sin(w * t)
        c = np.cos(w * t)

        x = np.array([
            self.radius * s,
            -self.radius * c + self.radius,
            self.z0,
        ])
        v = np.array([
            self.radius * w * c,
            self.radius * w * s,
            0.0,
        ])
        a = np.array([
            -self.radius * w**2 * s,
            self.radius * w**2 * c,
            0.0,
        ])
        return self._pack(x, v, a)

    def _figure8(self, t):
        a_xy = self.figure8_a
        b_xy = self.figure8_b
        z_amp = self.figure8_z_amp
        w = self.figure8_omega

        theta = w * t
        s1 = np.sin(theta)
        c1 = np.cos(theta)
        s2 = np.sin(2.0 * theta)
        c2 = np.cos(2.0 * theta)

        x = np.array([
            a_xy * s1,
            0.5 * b_xy * s2,
            self.z0 + z_amp * s1,
        ])
        v = np.array([
            a_xy * w * c1,
            b_xy * w * c2,
            z_amp * w * c1,
        ])
        acc = np.array([
            -a_xy * w**2 * s1,
            -2.0 * b_xy * w**2 * s2,
            -z_amp * w**2 * s1,
        ])
        return self._pack(x, v, acc)
