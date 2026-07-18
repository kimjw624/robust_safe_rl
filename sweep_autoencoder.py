import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from autoencoder import AutoEncoder
from train_autoencoder import (
    collect_dataset,
    standardize,
    evaluate_errors,
    detection_report,
    save_training_plots,
)


def parse_hidden_dims_grid(text):
    """
    Example input:
        "128,64;256,128,64;512,256,128,64"

    Returns:
        [(128, 64), (256, 128, 64), (512, 256, 128, 64)]
    """

    configs = []

    for block in text.split(";"):
        block = block.strip()

        if not block:
            continue

        dims = tuple(int(x.strip()) for x in block.split(",") if x.strip())
        configs.append(dims)

    return configs


def parse_list(text, dtype):
    return [dtype(x.strip()) for x in text.split(",") if x.strip()]


def make_config_grid(args):
    hidden_dims_grid = parse_hidden_dims_grid(args.hidden_dims_grid)
    latent_dims = parse_list(args.latent_dims, int)
    activations = parse_list(args.activations, str)
    learning_rates = parse_list(args.learning_rates, float)
    weight_decays = parse_list(args.weight_decays, float)
    batch_sizes = parse_list(args.batch_sizes, int)

    configs = []

    for hidden_dims, latent_dim, activation, lr, weight_decay, batch_size in itertools.product(
        hidden_dims_grid,
        latent_dims,
        activations,
        learning_rates,
        weight_decays,
        batch_sizes,
    ):
        configs.append({
            "hidden_dims": hidden_dims,
            "latent_dim": latent_dim,
            "activation": activation,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        })

    return configs


def train_one_model(
    config,
    train_x,
    val_x,
    ood_x,
    epochs,
    device,
    run_dir,
    plot_dir,
    run_idx,
    seed,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = train_x.shape[1]

    model = AutoEncoder(
        input_dim=input_dim,
        hidden_dims=config["hidden_dims"],
        latent_dim=config["latent_dim"],
        activation=config["activation"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    criterion = torch.nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(train_x),
        batch_size=config["batch_size"],
        shuffle=True,
    )

    history = {
        "epoch": [],
        "train_mse": [],
        "val_mse": [],
    }

    best_val = float("inf")
    best_state = None
    best_epoch = 0

    print("")
    print("=" * 80)
    print(f"Run {run_idx:03d}")
    print(config)
    print("=" * 80)

    for epoch in range(1, epochs + 1):
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
            batch_size=config["batch_size"],
            device=device,
        )

        val_loss = float(np.mean(val_errors))

        history["epoch"].append(epoch)
        history["train_mse"].append(train_loss)
        history["val_mse"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(
                f"run {run_idx:03d} | "
                f"epoch {epoch:04d} | "
                f"train mse {train_loss:.6e} | "
                f"val mse {val_loss:.6e} | "
                f"best val {best_val:.6e} at epoch {best_epoch}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    id_errors, id_latent = evaluate_errors(
        model=model,
        x=val_x,
        batch_size=config["batch_size"],
        device=device,
    )

    ood_errors, ood_latent = evaluate_errors(
        model=model,
        x=ood_x,
        batch_size=config["batch_size"],
        device=device,
    )

    thresholds, threshold_lines = detection_report(
        id_errors=id_errors,
        ood_errors=ood_errors,
    )

    metrics = {
        "run_idx": run_idx,
        "input_dim": int(input_dim),
        "hidden_dims": list(config["hidden_dims"]),
        "latent_dim": int(config["latent_dim"]),
        "activation": config["activation"],
        "lr": float(config["lr"]),
        "weight_decay": float(config["weight_decay"]),
        "batch_size": int(config["batch_size"]),
        "epochs": int(epochs),
        "best_epoch": int(best_epoch),
        "best_val_mse": float(best_val),

        "id_mean": float(np.mean(id_errors)),
        "id_p95": float(np.percentile(id_errors, 95)),
        "id_p99": float(np.percentile(id_errors, 99)),
        "id_p995": float(np.percentile(id_errors, 99.5)),
        "id_max": float(np.max(id_errors)),

        "ood_mean": float(np.mean(ood_errors)),
        "ood_p95": float(np.percentile(ood_errors, 95)),
        "ood_p99": float(np.percentile(ood_errors, 99)),
        "ood_p995": float(np.percentile(ood_errors, 99.5)),
        "ood_max": float(np.max(ood_errors)),
    }

    for name, threshold, id_false_alarm, ood_detected in threshold_lines:
        metrics[f"{name}_threshold"] = float(threshold)
        metrics[f"{name}_id_false_alarm"] = float(id_false_alarm)
        metrics[f"{name}_ood_detected"] = float(ood_detected)

    checkpoint_path = run_dir / f"{run_idx:03d}_ae_disturbance.pt"
    config_path = run_dir / f"{run_idx:03d}_config.json"
    metrics_path = run_dir / f"{run_idx:03d}_metrics.json"
    history_path = run_dir / f"{run_idx:03d}_history.csv"

    model.save(
        path=checkpoint_path,
        mean=train_one_model.mean,
        std=train_one_model.std,
        thresholds=thresholds,
    )

    payload = torch.load(checkpoint_path, map_location="cpu")
    payload["sweep_config"] = config
    payload["sweep_metrics"] = metrics
    torch.save(payload, checkpoint_path)

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse", "val_mse"])

        for e, tr, va in zip(
            history["epoch"],
            history["train_mse"],
            history["val_mse"],
        ):
            writer.writerow([e, tr, va])

    save_training_plots(
        plot_dir=plot_dir,
        run_idx=run_idx,
        id_errors=id_errors,
        ood_errors=ood_errors,
        id_latent=id_latent,
        ood_latent=ood_latent,
        thresholds=thresholds,
    )

    print("")
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved config:     {config_path}")
    print(f"Saved metrics:    {metrics_path}")
    print(f"Saved history:    {history_path}")

    print("")
    print("Main detection results")
    print(
        f"id_p95  | false alarm {metrics['id_p95_id_false_alarm']:.3%} | "
        f"OOD detected {metrics['id_p95_ood_detected']:.3%}"
    )
    print(
        f"id_p99  | false alarm {metrics['id_p99_id_false_alarm']:.3%} | "
        f"OOD detected {metrics['id_p99_ood_detected']:.3%}"
    )
    print(
        f"id_p995 | false alarm {metrics['id_p995_id_false_alarm']:.3%} | "
        f"OOD detected {metrics['id_p995_ood_detected']:.3%}"
    )
    print(
        f"id_max  | false alarm {metrics['id_max_id_false_alarm']:.3%} | "
        f"OOD detected {metrics['id_max_ood_detected']:.3%}"
    )

    return metrics


def save_master_summary(summary_path, all_metrics):
    if len(all_metrics) == 0:
        return

    fieldnames = sorted(all_metrics[0].keys())

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_metrics:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tf", type=float, default=10.0)

    parser.add_argument("--train_episodes", type=int, default=1000)
    parser.add_argument("--val_episodes", type=int, default=200)
    parser.add_argument("--ood_episodes", type=int, default=200)

    parser.add_argument("--history_len", type=int, default=3)

    parser.add_argument(
        "--hidden_dims_grid",
        type=str,
        default="128,64;256,128,64;512,256,128,64",
    )

    parser.add_argument("--latent_dims", type=str, default="2,3,4,6")
    parser.add_argument("--activations", type=str, default="relu,elu")
    parser.add_argument("--learning_rates", type=str, default="5e-4,3e-4")
    parser.add_argument("--weight_decays", type=str, default="1e-6")
    parser.add_argument("--batch_sizes", type=str, default="2048")

    parser.add_argument("--epochs", type=int, default=2000)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--force_sample_each_step", action="store_true")

    parser.add_argument("--sweep_dir", type=str, default="runs/sweep_001")

    parser.add_argument("--max_runs", type=int, default=-1)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sweep_dir = Path(args.sweep_dir)
    run_dir = sweep_dir / "checkpoints"
    plot_dir = sweep_dir / "plots"

    sweep_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    configs = make_config_grid(args)

    if args.max_runs > 0:
        configs = configs[:args.max_runs]

    print("")
    print("=" * 80)
    print("Autoencoder hyperparameter sweep")
    print("=" * 80)
    print(f"device: {device}")
    print(f"number of configs: {len(configs)}")
    print(f"sweep_dir: {sweep_dir}")
    print("")

    with open(sweep_dir / "sweep_args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    episode_steps = int(args.tf / args.dt)

    print("Collecting one shared large ID training dataset")
    print("ID force mode: random_force=1, force in [-3, 3] N")

    train_np, train_forces = collect_dataset(
        num_episodes=args.train_episodes,
        episode_steps=episode_steps,
        dt=args.dt,
        random_force=1,
        history_len=args.history_len,
        seed=args.seed,
        force_sample_each_step=args.force_sample_each_step,
    )

    print("Collecting one shared large ID validation dataset")

    val_np, val_forces = collect_dataset(
        num_episodes=args.val_episodes,
        episode_steps=episode_steps,
        dt=args.dt,
        random_force=1,
        history_len=args.history_len,
        seed=args.seed + 1000,
        force_sample_each_step=args.force_sample_each_step,
    )

    print("Collecting one shared strict OOD evaluation dataset")
    print("Strict OOD force mode: random_force=3")

    ood_np, ood_forces = collect_dataset(
        num_episodes=args.ood_episodes,
        episode_steps=episode_steps,
        dt=args.dt,
        random_force=3,
        history_len=args.history_len,
        seed=args.seed + 2000,
        force_sample_each_step=args.force_sample_each_step,
    )

    print("")
    print(f"train samples: {train_np.shape[0]}")
    print(f"val samples:   {val_np.shape[0]}")
    print(f"OOD samples:   {ood_np.shape[0]}")
    print(f"input dim:      {train_np.shape[1]}")

    np.savez_compressed(
        sweep_dir / "dataset_info_and_forces.npz",
        train_forces=train_forces,
        val_forces=val_forces,
        ood_forces=ood_forces,
    )

    train_x = torch.tensor(train_np, dtype=torch.float32)
    val_x = torch.tensor(val_np, dtype=torch.float32)
    ood_x = torch.tensor(ood_np, dtype=torch.float32)

    train_x, val_x, ood_x, mean, std = standardize(
        train_x,
        val_x,
        ood_x,
    )

    train_one_model.mean = mean
    train_one_model.std = std

    train_x = train_x.to(device)
    val_x = val_x.to(device)
    ood_x = ood_x.to(device)

    all_metrics = []

    for i, config in enumerate(configs, start=1):
        metrics = train_one_model(
            config=config,
            train_x=train_x,
            val_x=val_x,
            ood_x=ood_x,
            epochs=args.epochs,
            device=device,
            run_dir=run_dir,
            plot_dir=plot_dir,
            run_idx=i,
            seed=args.seed + i,
        )

        all_metrics.append(metrics)

        save_master_summary(
            summary_path=sweep_dir / "summary.csv",
            all_metrics=all_metrics,
        )

    save_master_summary(
        summary_path=sweep_dir / "summary.csv",
        all_metrics=all_metrics,
    )

    with open(sweep_dir / "summary.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("")
    print("=" * 80)
    print("Sweep complete")
    print("=" * 80)
    print(f"Master CSV:  {sweep_dir / 'summary.csv'}")
    print(f"Master JSON: {sweep_dir / 'summary.json'}")
    print(f"Checkpoints: {run_dir}")
    print(f"Plots:       {plot_dir}")


if __name__ == "__main__":
    main()