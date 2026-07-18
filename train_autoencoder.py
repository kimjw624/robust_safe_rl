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


def rotation_vector_by_columns(R):
    return R.reshape(9, order="F")


def make_step_feature(nominal_state, true_state, action):
    """
    One-step feature:

        error = nominal - true

        ex      : 3
        ev      : 3
        eR      : 9
        eomega  : 3
        action  : 4, [f, Mx, My, Mz]

    one-step dim = 22
    history_len=3 gives input_dim = 66
    """

    ex = nominal_state["x"] - true_state["x"]
    ev = nominal_state["v"] - true_state["v"]

    eR = (
        rotation_vector_by_columns(nominal_state["R"])
        - rotation_vector_by_columns(true_state["R"])
    )

    eomega = nominal_state["omega"] - true_state["omega"]

    return np.concatenate([
        ex,
        ev,
        eR,
        eomega,
        action,
    ])


def collect_dataset(
    num_episodes,
    episode_steps,
    dt,
    random_force,
    history_len,
    seed,
    force_sample_each_step=False,
):
    rng = np.random.default_rng(seed)

    traj = DesiredTrajectory(
        radius=0.79,
        speed=0.5,
        z0=-1.0,
    )

    samples = []
    forces = []

    for ep in range(num_episodes):
        true_dyn = Dynamics(
            dt=dt,
            random_force=random_force,
            force_sample_each_step=force_sample_each_step,
            seed=int(rng.integers(0, 2**31 - 1)),
        )

        nominal_dyn = Dynamics(
            dt=dt,
            random_force=0,
        )

        true_ctrl = Controller(dt=dt)
        nominal_ctrl = Controller(dt=dt)

        d0 = traj.desired(0.0)

        true_state = true_dyn.reset(
            x=d0["x"],
            v=d0["v"],
        )

        nominal_state = nominal_dyn.reset(
            x=d0["x"],
            v=d0["v"],
        )

        true_ctrl.reset()
        nominal_ctrl.reset()

        hist = deque(maxlen=history_len)

        for k in range(episode_steps):
            t = k * dt
            desired = traj.desired(t)

            f_true, M_true, _ = true_ctrl.compute_control(true_state, desired)
            f_nom, M_nom, _ = nominal_ctrl.compute_control(nominal_state, desired)

            action = np.array([
                f_true,
                M_true[0],
                M_true[1],
                M_true[2],
            ], dtype=float)

            one_step_feature = make_step_feature(
                nominal_state=nominal_state,
                true_state=true_state,
                action=action,
            )

            hist.append(one_step_feature)

            if len(hist) == history_len:
                samples.append(np.concatenate(list(hist)))
                forces.append(true_dyn.external_force.copy())

            true_state = true_dyn.step(f_true, M_true)
            nominal_state = nominal_dyn.step(f_nom, M_nom)

    return (
        np.asarray(samples, dtype=np.float32),
        np.asarray(forces, dtype=np.float32),
    )


def standardize(train_x, val_x, ood_x):
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True)

    std = torch.clamp(std, min=1e-6)

    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std
    ood_x = (ood_x - mean) / std

    return train_x, val_x, ood_x, mean, std


def apply_standardization(x, mean, std):
    return (x - mean) / torch.clamp(std, min=1e-6)


def evaluate_errors(model, x, batch_size, device):
    model.eval()

    errors = []
    latents = []

    loader = DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        shuffle=False,
    )

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)

            err, z = model.reconstruction_error(
                xb,
                reduction="none",
            )

            errors.append(err.cpu())
            latents.append(z.cpu())

    return (
        torch.cat(errors).numpy(),
        torch.cat(latents).numpy(),
    )


def detection_report(id_errors, ood_errors):
    thresholds = {
        "id_max": float(np.max(id_errors)),
        "id_p95": float(np.percentile(id_errors, 95.0)),
        "id_p99": float(np.percentile(id_errors, 99.0)),
        "id_p995": float(np.percentile(id_errors, 99.5)),
        "id_mean_plus_3std": float(np.mean(id_errors) + 3.0 * np.std(id_errors)),
    }

    lines = []

    for name, th in thresholds.items():
        id_false_alarm = float(np.mean(id_errors > th))
        ood_detected = float(np.mean(ood_errors > th))

        lines.append((
            name,
            th,
            id_false_alarm,
            ood_detected,
        ))

    return thresholds, lines


def get_next_indexed_path(directory, suffix):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    existing = sorted(directory.glob(f"*_{suffix}"))

    used_indices = []

    for path in existing:
        prefix = path.name.split("_", 1)[0]

        if prefix.isdigit():
            used_indices.append(int(prefix))

    next_idx = 1 if len(used_indices) == 0 else max(used_indices) + 1

    return directory / f"{next_idx:03d}_{suffix}", next_idx


def save_training_plots(
    plot_dir,
    run_idx,
    id_errors,
    ood_errors,
    id_latent,
    ood_latent,
    thresholds,
):
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 001 reconstruction histogram
    plt.figure()
    plt.hist(id_errors, bins=100, alpha=0.6, label="ID force in [-3, 3] N")
    plt.hist(ood_errors, bins=100, alpha=0.6, label="Strict OOD force, at least one axis > 3 N")
    plt.axvline(
        thresholds["id_p99"],
        linestyle="--",
        label="ID p99 threshold",
    )
    plt.xlabel("reconstruction MSE")
    plt.ylabel("count")
    plt.title("Autoencoder anomaly score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{run_idx:03d}_001_reconstruction_hist.png", dpi=200)
    plt.close()

    # 002 latent z1-z2
    if id_latent.shape[1] >= 2:
        plt.figure()
        plt.scatter(id_latent[:, 0], id_latent[:, 1], s=3, alpha=0.35, label="ID")
        plt.scatter(ood_latent[:, 0], ood_latent[:, 1], s=3, alpha=0.35, label="Strict OOD")
        plt.xlabel("z1")
        plt.ylabel("z2")
        plt.title("Latent space: z1 vs z2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{run_idx:03d}_002_latent_z1_z2.png", dpi=200)
        plt.close()

    # 003 latent z1-z3
    if id_latent.shape[1] >= 3:
        plt.figure()
        plt.scatter(id_latent[:, 0], id_latent[:, 2], s=3, alpha=0.35, label="ID")
        plt.scatter(ood_latent[:, 0], ood_latent[:, 2], s=3, alpha=0.35, label="Strict OOD")
        plt.xlabel("z1")
        plt.ylabel("z3")
        plt.title("Latent space: z1 vs z3")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{run_idx:03d}_003_latent_z1_z3.png", dpi=200)
        plt.close()

    # 004 latent z2-z3
    if id_latent.shape[1] >= 3:
        plt.figure()
        plt.scatter(id_latent[:, 1], id_latent[:, 2], s=3, alpha=0.35, label="ID")
        plt.scatter(ood_latent[:, 1], ood_latent[:, 2], s=3, alpha=0.35, label="Strict OOD")
        plt.xlabel("z2")
        plt.ylabel("z3")
        plt.title("Latent space: z2 vs z3")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"{run_idx:03d}_004_latent_z2_z3.png", dpi=200)
        plt.close()

    # 005 3D latent plot
    if id_latent.shape[1] >= 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            id_latent[:, 0],
            id_latent[:, 1],
            id_latent[:, 2],
            s=3,
            alpha=0.25,
            label="ID",
        )

        ax.scatter(
            ood_latent[:, 0],
            ood_latent[:, 1],
            ood_latent[:, 2],
            s=3,
            alpha=0.25,
            label="Strict OOD",
        )

        ax.set_xlabel("z1")
        ax.set_ylabel("z2")
        ax.set_zlabel("z3")
        ax.set_title("3D latent space")
        ax.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"{run_idx:03d}_005_latent_3d.png", dpi=200)
        plt.close()

    # 006 sorted reconstruction error
    id_sorted = np.sort(id_errors)
    ood_sorted = np.sort(ood_errors)

    plt.figure()
    plt.plot(id_sorted, label="ID")
    plt.plot(ood_sorted, label="Strict OOD")
    plt.axhline(
        thresholds["id_p99"],
        linestyle="--",
        label="ID p99 threshold",
    )
    plt.xlabel("sorted sample index")
    plt.ylabel("reconstruction MSE")
    plt.title("Sorted reconstruction errors")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{run_idx:03d}_006_sorted_reconstruction_error.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tf", type=float, default=10.0)

    parser.add_argument("--train_episodes", type=int, default=200)
    parser.add_argument("--val_episodes", type=int, default=50)
    parser.add_argument("--ood_episodes", type=int, default=50)

    parser.add_argument("--history_len", type=int, default=3)

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

    episode_steps = int(args.tf / args.dt)

    hidden_dims = tuple(
        int(x) for x in args.hidden_dims.split(",") if x.strip()
    )

    save_path, run_idx = get_next_indexed_path(
        directory=args.run_dir,
        suffix=args.save_suffix,
    )

    print(f"Run index: {run_idx:03d}")
    print(f"Checkpoint will be saved to: {save_path}")

    print("Collecting ID training data: random_force=1, [-3, 3] N")

    train_np, train_forces = collect_dataset(
        num_episodes=args.train_episodes,
        episode_steps=episode_steps,
        dt=args.dt,
        random_force=1,
        history_len=args.history_len,
        seed=args.seed,
        force_sample_each_step=args.force_sample_each_step,
    )

    print("Collecting ID validation data: random_force=1, [-3, 3] N")

    val_np, val_forces = collect_dataset(
        num_episodes=args.val_episodes,
        episode_steps=episode_steps,
        dt=args.dt,
        random_force=1,
        history_len=args.history_len,
        seed=args.seed + 1000,
        force_sample_each_step=args.force_sample_each_step,
    )

    print("Collecting strict OOD evaluation data: random_force=3")
    print("Strict OOD means force sampled from [-5, 5] N, but at least one axis is outside [-3, 3] N")

    ood_np, ood_forces = collect_dataset(
        num_episodes=args.ood_episodes,
        episode_steps=episode_steps,
        dt=args.dt,
        random_force=3,
        history_len=args.history_len,
        seed=args.seed + 2000,
        force_sample_each_step=args.force_sample_each_step,
    )

    print(f"train samples: {train_np.shape[0]}")
    print(f"val samples:   {val_np.shape[0]}")
    print(f"OOD samples:   {ood_np.shape[0]}")

    train_x = torch.tensor(train_np, dtype=torch.float32)
    val_x = torch.tensor(val_np, dtype=torch.float32)
    ood_x = torch.tensor(ood_np, dtype=torch.float32)

    train_x, val_x, ood_x, mean, std = standardize(
        train_x,
        val_x,
        ood_x,
    )

    input_dim = train_x.shape[1]

    model = AutoEncoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=args.latent_dim,
        activation=args.activation,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(train_x),
        batch_size=args.batch_size,
        shuffle=True,
    )

    print(
        f"input_dim={input_dim}, "
        f"hidden_dims={hidden_dims}, "
        f"latent_dim={args.latent_dim}, "
        f"device={device}"
    )

    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for (xb,) in train_loader:
            xb = xb.to(device)

            x_hat, _ = model(xb)
            loss = criterion(x_hat, xb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * xb.shape[0]
            train_count += xb.shape[0]

        train_loss = train_loss_sum / train_count

        val_errors, _ = evaluate_errors(
            model=model,
            x=val_x,
            batch_size=args.batch_size,
            device=device,
        )

        val_loss = float(np.mean(val_errors))

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:04d} | "
                f"train mse {train_loss:.6e} | "
                f"val mse {val_loss:.6e} | "
                f"best val {best_val:.6e}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    id_errors, id_latent = evaluate_errors(
        model=model,
        x=val_x,
        batch_size=args.batch_size,
        device=device,
    )

    ood_errors, ood_latent = evaluate_errors(
        model=model,
        x=ood_x,
        batch_size=args.batch_size,
        device=device,
    )

    thresholds, lines = detection_report(
        id_errors=id_errors,
        ood_errors=ood_errors,
    )

    print("\nReconstruction-error summary")

    print(
        "ID  "
        f"mean={id_errors.mean():.6e}, "
        f"p95={np.percentile(id_errors, 95):.6e}, "
        f"p99={np.percentile(id_errors, 99):.6e}, "
        f"p99.5={np.percentile(id_errors, 99.5):.6e}, "
        f"max={id_errors.max():.6e}"
    )

    print(
        "OOD "
        f"mean={ood_errors.mean():.6e}, "
        f"p95={np.percentile(ood_errors, 95):.6e}, "
        f"p99={np.percentile(ood_errors, 99):.6e}, "
        f"p99.5={np.percentile(ood_errors, 99.5):.6e}, "
        f"max={ood_errors.max():.6e}"
    )

    print("\nDetection report")
    print("threshold_name       threshold       ID false alarm       OOD detected")

    for name, th, fa, detected in lines:
        print(
            f"{name:18s} "
            f"{th:.6e} "
            f"{fa:18.3%} "
            f"{detected:18.3%}"
        )

    metadata = {
        "run_idx": run_idx,
        "dt": args.dt,
        "tf": args.tf,
        "train_episodes": args.train_episodes,
        "val_episodes": args.val_episodes,
        "ood_episodes": args.ood_episodes,
        "history_len": args.history_len,
        "force_sample_each_step": args.force_sample_each_step,
        "train_force_mode": 1,
        "ood_force_mode": 3,
    }

    model.save(
        path=save_path,
        mean=mean,
        std=std,
        thresholds=thresholds,
    )

    # Add metadata to saved checkpoint.
    payload = torch.load(save_path, map_location="cpu")
    payload["metadata"] = metadata
    torch.save(payload, save_path)

    save_training_plots(
        plot_dir=args.plot_dir,
        run_idx=run_idx,
        id_errors=id_errors,
        ood_errors=ood_errors,
        id_latent=id_latent,
        ood_latent=ood_latent,
        thresholds=thresholds,
    )

    print(f"\nSaved model to: {save_path}")
    print(f"Saved plots to: {args.plot_dir}")
    print(f"Recommended first threshold: id_p99 = {thresholds['id_p99']:.6e}")


if __name__ == "__main__":
    main()