import argparse
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import AutoEncoder
from controller import Controller
from desired_trajectory import DesiredTrajectory
from dynamics import Dynamics

FEATURE_VERSION = 2
STEP_FEATURE_DIM = 16
FEATURE_NAMES = (
    "position_error_next[3]",
    "velocity_error_next[3]",
    "relative_rotation_vector_next[3]",
    "angular_velocity_error_next[3]",
    "commanded_action[f,Mx,My,Mz][4]",
)


def vee(S):
    return np.array([-S[1, 2], S[0, 2], -S[0, 1]], dtype=float)


def so3_log_vector(R):
    """Return the rotation vector whose exponential is R."""
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < 1e-7:
        return 0.5 * vee(R - R.T)

    if np.pi - theta < 1e-5:
        # Stable axis extraction near pi.
        A = 0.5 * (R + np.eye(3))
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        axis[0] = np.copysign(axis[0], R[2, 1] - R[1, 2])
        axis[1] = np.copysign(axis[1], R[0, 2] - R[2, 0])
        axis[2] = np.copysign(axis[2], R[1, 0] - R[0, 1])
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            return np.zeros(3)
        return theta * axis / norm

    return theta * vee(R - R.T) / (2.0 * np.sin(theta))


def make_step_feature(nominal_next_state, true_next_state, action):
    """
    Build one transition-aligned feature vector.

    The same commanded action is applied to both systems over [t, t+dt].
    The feature then pairs that action with the state discrepancy observed at
    t+dt. This prevents controller/action mismatch from being mislabeled as a
    disturbance.
    """
    action = np.asarray(action, dtype=float).reshape(4)

    ex = nominal_next_state["x"] - true_next_state["x"]
    ev = nominal_next_state["v"] - true_next_state["v"]

    # Rotation taking the true attitude into the nominal attitude.
    relative_R = true_next_state["R"].T @ nominal_next_state["R"]
    eR = so3_log_vector(relative_R)

    eomega = nominal_next_state["omega"] - true_next_state["omega"]

    feature = np.concatenate((ex, ev, eR, eomega, action)).astype(np.float64)
    if feature.shape != (STEP_FEATURE_DIM,):
        raise RuntimeError(f"Unexpected step feature shape: {feature.shape}")
    if not np.all(np.isfinite(feature)):
        raise FloatingPointError("Non-finite value found in a step feature.")
    return feature


def feature_metadata(history_len):
    history_len = int(history_len)
    return {
        "feature_version": FEATURE_VERSION,
        "history_len": history_len,
        "step_feature_dim": STEP_FEATURE_DIM,
        "input_dim": STEP_FEATURE_DIM * history_len,
        "feature_names": list(FEATURE_NAMES),
        "error_sign": "nominal_minus_true",
        "transition_alignment": "action_t_with_state_error_t_plus_1",
        "action_source": "single_true_state_feedback_command_applied_to_both_models",
        "rotation_error": "Log(R_true_next.T @ R_nominal_next)",
    }


def collect_dataset(
    num_episodes,
    episode_steps,
    dt,
    random_force,
    history_len,
    seed,
    force_sample_each_step=False,
):
    if num_episodes <= 0 or episode_steps <= 0 or history_len <= 0:
        raise ValueError("num_episodes, episode_steps, and history_len must be positive.")
    if episode_steps < history_len:
        raise ValueError("episode_steps must be at least history_len.")

    rng = np.random.default_rng(seed)
    traj = DesiredTrajectory(radius=0.79, speed=0.5, z0=-1.0)

    samples = []
    forces = []

    for _ in range(num_episodes):
        true_dyn = Dynamics(
            dt=dt,
            random_force=random_force,
            force_sample_each_step=force_sample_each_step,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        nominal_dyn = Dynamics(dt=dt, random_force=0)

        # One controller represents the real command source. Its command is
        # applied identically to the true and nominal dynamics.
        controller = Controller(dt=dt)

        d0 = traj.desired(0.0)
        true_state = true_dyn.reset(x=d0["x"], v=d0["v"])
        nominal_state = nominal_dyn.reset(x=d0["x"], v=d0["v"])
        controller.reset()

        history = deque(maxlen=history_len)

        for k in range(episode_steps):
            t = k * dt
            desired = traj.desired(t)

            f_cmd, M_cmd, _ = controller.compute_control(true_state, desired)
            action = np.array([f_cmd, *M_cmd], dtype=float)

            true_next = true_dyn.step(f_cmd, M_cmd)
            nominal_next = nominal_dyn.step(f_cmd, M_cmd)

            history.append(make_step_feature(nominal_next, true_next, action))

            if len(history) == history_len:
                samples.append(np.concatenate(tuple(history)))
                forces.append(true_dyn.last_external_force.copy())

            true_state = true_next
            nominal_state = nominal_next

    x = np.asarray(samples, dtype=np.float32)
    disturbance = np.asarray(forces, dtype=np.float32)
    expected_dim = STEP_FEATURE_DIM * history_len

    if x.ndim != 2 or x.shape[1] != expected_dim:
        raise RuntimeError(f"Expected dataset shape (N, {expected_dim}), got {x.shape}.")
    if not np.all(np.isfinite(x)):
        raise FloatingPointError("Collected dataset contains non-finite values.")

    return x, disturbance


def standardize(train_x, val_x, ood_x):
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (
        (train_x - mean) / std,
        (val_x - mean) / std,
        (ood_x - mean) / std,
        mean,
        std,
    )


def apply_standardization(x, mean, std):
    if x.ndim != 2:
        raise ValueError(f"Expected a 2-D input tensor, got shape {tuple(x.shape)}.")
    if x.shape[1] != mean.shape[-1] or mean.shape != std.shape:
        raise ValueError(
            f"Input/normalization dimension mismatch: input={x.shape[1]}, "
            f"mean={tuple(mean.shape)}, std={tuple(std.shape)}."
        )
    return (x - mean) / std.clamp_min(1e-6)


def evaluate_errors(model, x, batch_size, device):
    model.eval()
    errors, latents = [], []
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for (xb,) in loader:
            err, z = model.reconstruction_error(xb.to(device), reduction="none")
            errors.append(err.cpu())
            latents.append(z.cpu())

    return torch.cat(errors).numpy(), torch.cat(latents).numpy()


def detection_report(id_errors, ood_errors):
    thresholds = {
        "id_max": float(np.max(id_errors)),
        "id_p95": float(np.percentile(id_errors, 95.0)),
        "id_p99": float(np.percentile(id_errors, 99.0)),
        "id_p995": float(np.percentile(id_errors, 99.5)),
        "id_mean_plus_3std": float(np.mean(id_errors) + 3.0 * np.std(id_errors)),
    }
    lines = []
    for name, threshold in thresholds.items():
        lines.append((
            name,
            threshold,
            float(np.mean(id_errors > threshold)),
            float(np.mean(ood_errors > threshold)),
        ))
    return thresholds, lines


def get_next_indexed_path(directory, suffix):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    used = []
    for path in directory.glob(f"*_{suffix}"):
        prefix = path.name.split("_", 1)[0]
        if prefix.isdigit():
            used.append(int(prefix))
    index = 1 if not used else max(used) + 1
    return directory / f"{index:03d}_{suffix}", index


def _axis_limits(a, b, i, j):
    x = np.concatenate((a[:, i], b[:, i]))
    y = np.concatenate((a[:, j], b[:, j]))
    x_pad = max(0.05 * np.ptp(x), 1e-6)
    y_pad = max(0.05 * np.ptp(y), 1e-6)
    return (x.min() - x_pad, x.max() + x_pad), (y.min() - y_pad, y.max() + y_pad)


def _save_latent_2d_triplet(plt, plot_dir, prefix, id_latent, ood_latent, i, j):
    xlim, ylim = _axis_limits(id_latent, ood_latent, i, j)
    specs = (("id", ((id_latent, "ID"),)),
             ("ood", ((ood_latent, "Strict OOD"),)),
             ("both", ((id_latent, "ID"), (ood_latent, "Strict OOD"))))
    for tag, groups in specs:
        plt.figure()
        for points, label in groups:
            plt.scatter(points[:, i], points[:, j], s=3, alpha=0.35, label=label)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.xlabel(f"z{i + 1}")
        plt.ylabel(f"z{j + 1}")
        plt.title(f"Latent space z{i + 1} vs z{j + 1}: {tag.upper()}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{prefix}_latent_z{i + 1}_z{j + 1}_{tag}.png", dpi=200)
        plt.close()


def _save_latent_3d_triplet(plt, plot_dir, prefix, id_latent, ood_latent):
    combined = np.vstack((id_latent[:, :3], ood_latent[:, :3]))
    mins, maxs = combined.min(axis=0), combined.max(axis=0)
    pads = np.maximum(0.05 * (maxs - mins), 1e-6)
    specs = (("id", ((id_latent, "ID"),)),
             ("ood", ((ood_latent, "Strict OOD"),)),
             ("both", ((id_latent, "ID"), (ood_latent, "Strict OOD"))))
    for tag, groups in specs:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for points, label in groups:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=3, alpha=0.25, label=label)
        ax.set_xlim(mins[0] - pads[0], maxs[0] + pads[0])
        ax.set_ylim(mins[1] - pads[1], maxs[1] + pads[1])
        ax.set_zlim(mins[2] - pads[2], maxs[2] + pads[2])
        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_zlabel("z3")
        ax.set_title(f"3D latent space: {tag.upper()}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{prefix}_latent_3d_{tag}.png", dpi=200)
        plt.close()


def save_training_plots(plot_dir, run_idx, id_errors, ood_errors, id_latent, ood_latent, thresholds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{run_idx:03d}"

    plt.figure()
    plt.hist(id_errors, bins=100, alpha=0.6, label="ID")
    plt.hist(ood_errors, bins=100, alpha=0.6, label="Strict OOD")
    plt.axvline(thresholds["id_p99"], linestyle="--", label="ID p99 threshold")
    plt.xlabel("reconstruction MSE")
    plt.ylabel("count")
    plt.title("Autoencoder anomaly score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{prefix}_reconstruction_hist.png", dpi=200)
    plt.close()

    if id_latent.shape[1] >= 2:
        _save_latent_2d_triplet(plt, plot_dir, prefix, id_latent, ood_latent, 0, 1)
    if id_latent.shape[1] >= 3:
        _save_latent_2d_triplet(plt, plot_dir, prefix, id_latent, ood_latent, 0, 2)
        _save_latent_2d_triplet(plt, plot_dir, prefix, id_latent, ood_latent, 1, 2)
        _save_latent_3d_triplet(plt, plot_dir, prefix, id_latent, ood_latent)

    plt.figure()
    plt.plot(np.sort(id_errors), label="ID")
    plt.plot(np.sort(ood_errors), label="Strict OOD")
    plt.axhline(thresholds["id_p99"], linestyle="--", label="ID p99 threshold")
    plt.xlabel("sorted sample index")
    plt.ylabel("reconstruction MSE")
    plt.title("Sorted reconstruction errors")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{prefix}_sorted_reconstruction_error.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tf", type=float, default=10.0)
    parser.add_argument("--train_episodes", type=int, default=200)
    parser.add_argument("--val_episodes", type=int, default=50)
    parser.add_argument("--ood_episodes", type=int, default=50)
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--hidden_dims", type=str, default="256,128,64")
    parser.add_argument("--latent_dim", type=int, default=3)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force_sample_each_step", action="store_true")
    parser.add_argument("--run_dir", type=str, default="runs")
    parser.add_argument("--plot_dir", type=str, default="runs/plots")
    parser.add_argument("--save_suffix", type=str, default="ae_disturbance.pt")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    episode_steps = int(round(args.tf / args.dt))
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())
    save_path, run_idx = get_next_indexed_path(args.run_dir, args.save_suffix)

    train_np, train_forces = collect_dataset(args.train_episodes, episode_steps, args.dt, 1, args.history_len, args.seed, args.force_sample_each_step)
    val_np, val_forces = collect_dataset(args.val_episodes, episode_steps, args.dt, 1, args.history_len, args.seed + 1000, args.force_sample_each_step)
    ood_np, ood_forces = collect_dataset(args.ood_episodes, episode_steps, args.dt, 3, args.history_len, args.seed + 2000, args.force_sample_each_step)

    train_x, val_x, ood_x, mean, std = standardize(
        torch.from_numpy(train_np), torch.from_numpy(val_np), torch.from_numpy(ood_np)
    )

    model = AutoEncoder(train_x.shape[1], hidden_dims, args.latent_dim, args.activation).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()
    loader = DataLoader(TensorDataset(train_x), batch_size=args.batch_size, shuffle=True)

    best_val, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for (xb,) in loader:
            xb = xb.to(device)
            x_hat, _ = model(xb)
            loss = criterion(x_hat, xb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.shape[0]
        train_loss = total / len(train_x)
        val_errors, _ = evaluate_errors(model, val_x, args.batch_size, device)
        val_loss = float(val_errors.mean())
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch {epoch:04d} | train mse {train_loss:.6e} | val mse {val_loss:.6e} | best {best_val:.6e}")

    if best_state is not None:
        model.load_state_dict(best_state)

    id_errors, id_latent = evaluate_errors(model, val_x, args.batch_size, device)
    ood_errors, ood_latent = evaluate_errors(model, ood_x, args.batch_size, device)
    thresholds, lines = detection_report(id_errors, ood_errors)

    metadata = feature_metadata(args.history_len)
    metadata.update({
        "run_idx": run_idx,
        "dt": args.dt,
        "tf": args.tf,
        "train_episodes": args.train_episodes,
        "val_episodes": args.val_episodes,
        "ood_episodes": args.ood_episodes,
        "force_sample_each_step": args.force_sample_each_step,
        "train_force_mode": 1,
        "ood_force_mode": 3,
    })
    model.save(save_path, mean=mean, std=std, thresholds=thresholds, metadata=metadata)
    save_training_plots(args.plot_dir, run_idx, id_errors, ood_errors, id_latent, ood_latent, thresholds)

    print(f"Saved model to: {save_path}")
    print(f"Input dimension: {train_x.shape[1]} ({STEP_FEATURE_DIM} x history_len {args.history_len})")
    for name, threshold, false_alarm, detected in lines:
        print(f"{name:18s} {threshold:.6e} | ID false alarm {false_alarm:.3%} | OOD detected {detected:.3%}")


if __name__ == "__main__":
    main()
