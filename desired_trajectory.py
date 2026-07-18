import numpy as np


class DesiredTrajectory:
    """
    Circular trajectory in NED with constant zero desired yaw.

    Since this simulation uses NED:
        x[0] = North
        x[1] = East
        x[2] = Down

    Therefore, z0 = -1.0 means 1 meter above the origin.
    """

    def __init__(self, radius=0.79, speed=0.5, z0=-1.0):
        self.radius = float(radius)
        self.speed = float(speed)
        self.z0 = float(z0)

    def desired(self, t):
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

        # Constant zero desired yaw in NED.
        # Desired body x-axis points North.
        b1d = np.array([
            1.0,
            0.0,
            0.0,
        ])

        return {
            "x": x,
            "v": v,
            "a": a,
            "b1d": b1d,
        }