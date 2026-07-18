import numpy as np
import matplotlib.pyplot as plt

from dynamics import Dynamics
from controller import Controller
from desired_trajectory import DesiredTrajectory


def main():
    dt = 0.01
    tf = 20.0
    steps = int(tf / dt)

    traj = DesiredTrajectory(
        radius=0.79,
        speed=0.5,
        z0=-1.0,
    )

    dyn = Dynamics(dt=dt)
    ctrl = Controller(dt=dt)

    # Start on the desired trajectory.
    d0 = traj.desired(0.0)
    state = dyn.reset(
        x=d0["x"],
        v=d0["v"],
    )

    t_hist = []
    x_hist = []
    xd_hist = []
    err_hist = []

    f_hist = []
    M_hist = []

    for k in range(steps):
        t = k * dt

        desired = traj.desired(t)

        f, M, info = ctrl.compute_control(state, desired)

        state = dyn.step(f, M)

        t_hist.append(t)
        x_hist.append(state["x"].copy())
        xd_hist.append(desired["x"].copy())
        err_hist.append(np.linalg.norm(state["x"] - desired["x"]))

        f_hist.append(f)
        M_hist.append(M.copy())

    t_hist = np.asarray(t_hist)
    x_hist = np.asarray(x_hist)
    xd_hist = np.asarray(xd_hist)
    err_hist = np.asarray(err_hist)

    f_hist = np.asarray(f_hist)
    M_hist = np.asarray(M_hist)

    print(f"final position error: {err_hist[-1]:.4f} m")
    print(f"mean position error:  {err_hist.mean():.4f} m")
    print(f"max position error:   {err_hist.max():.4f} m")

    # ------------------------------------------------------------
    # 1. 3D trajectory
    # ------------------------------------------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        xd_hist[:, 0],
        xd_hist[:, 1],
        xd_hist[:, 2],
        "--",
        label="desired",
    )

    ax.plot(
        x_hist[:, 0],
        x_hist[:, 1],
        x_hist[:, 2],
        label="actual",
    )

    ax.set_xlabel("North x [m]")
    ax.set_ylabel("East y [m]")
    ax.set_zlabel("Down z [m]")
    ax.set_title("3D circular trajectory tracking")
    ax.legend()
    ax.grid(True)

    # Makes the axes visually more proportional.
    set_axes_equal(ax)

    # ------------------------------------------------------------
    # 2. Top-view trajectory
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(xd_hist[:, 0], xd_hist[:, 1], "--", label="desired")
    plt.plot(x_hist[:, 0], x_hist[:, 1], label="actual")
    plt.axis("equal")
    plt.xlabel("North x [m]")
    plt.ylabel("East y [m]")
    plt.title("Circular trajectory tracking, top view")
    plt.legend()
    plt.grid(True)

    # ------------------------------------------------------------
    # 3. Time vs desired/true position: North
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_hist, xd_hist[:, 0], "--", label="desired x")
    plt.plot(t_hist, x_hist[:, 0], label="actual x")
    plt.xlabel("time [s]")
    plt.ylabel("North x [m]")
    plt.title("North position tracking")
    plt.legend()
    plt.grid(True)

    # ------------------------------------------------------------
    # 4. Time vs desired/true position: East
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_hist, xd_hist[:, 1], "--", label="desired y")
    plt.plot(t_hist, x_hist[:, 1], label="actual y")
    plt.xlabel("time [s]")
    plt.ylabel("East y [m]")
    plt.title("East position tracking")
    plt.legend()
    plt.grid(True)

    # ------------------------------------------------------------
    # 5. Time vs desired/true position: Down
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_hist, xd_hist[:, 2], "--", label="desired z")
    plt.plot(t_hist, x_hist[:, 2], label="actual z")
    plt.xlabel("time [s]")
    plt.ylabel("Down z [m]")
    plt.title("Down position tracking")
    plt.legend()
    plt.grid(True)

    # ------------------------------------------------------------
    # 6. Position tracking error
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_hist, err_hist)
    plt.xlabel("time [s]")
    plt.ylabel("position error [m]")
    plt.title("Tracking error")
    plt.grid(True)

    # ------------------------------------------------------------
    # 7. Total thrust input
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_hist, f_hist)
    plt.xlabel("time [s]")
    plt.ylabel("thrust f [N]")
    plt.title("Total thrust input")
    plt.grid(True)

    # ------------------------------------------------------------
    # 8. Moment inputs
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(t_hist, M_hist[:, 0], label="Mx")
    plt.plot(t_hist, M_hist[:, 1], label="My")
    plt.plot(t_hist, M_hist[:, 2], label="Mz")
    plt.xlabel("time [s]")
    plt.ylabel("moment [N m]")
    plt.title("Body moment inputs")
    plt.legend()
    plt.grid(True)

    plt.show()


def set_axes_equal(ax):
    """
    Make 3D plot axes have equal scale.

    This is useful because matplotlib does not automatically make 3D axes
    equal, so circular trajectories can visually look distorted.
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


if __name__ == "__main__":
    main()