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
    feature_metadata,
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
    regularizers = parse_list(args.regularizers, str)
    sparse_lambdas = parse_list(args.sparse_lambdas, float)
    contractive_lambdas = parse_list(args.contractive_lambdas, float)

    configs = []
    common = itertools.product(
        hidden_dims_grid,
        latent_dims,
        activations,
        learning_rates,
        weight_decays,
        batch_sizes,
    )

    for hidden_dims, latent_dim, activation, lr, weight_decay, batch_size in common:
        base = {
            "hidden_dims": hidden_dims,
            "latent_dim": latent_dim,
            "activation": activation,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        }
        for regularizer in regularizers:
            regularizer = regularizer.lower()
            if regularizer == "none":
                configs.append({**base, "regularizer": "none", "reg_lambda": 0.0})
            elif regularizer == "sparse":
                for value in sparse_lambdas:
                    configs.append({**base, "regularizer": "sparse", "reg_lambda": value})
            elif regularizer == "contractive":
                for value in contractive_lambdas:
                    configs.append({**base, "regularizer": "contractive", "reg_lambda": value})
            else:
                raise ValueError(
                    f"Unsupported regularizer '{regularizer}'. "
                    "Use none, sparse, or contractive."
                )
    return configs


def regularization_penalty(model, xb, z, regularizer, contractive_samples):
    if regularizer == "none":
        return z.new_zeros(())

    if regularizer == "sparse":
        # L1 activity penalty: encourages each sample to use only a small
        # subset of latent coordinates. The latent layer is linear, so this
        # works with signed disturbance representations.
        return z.abs().mean()

    if regularizer == "contractive":
        # Hutchinson estimate of ||dz/dx||_F^2. For speed, estimate it on a
        # subset of each mini-batch. Division by latent dimension makes the
        # coefficient less sensitive to the chosen bottleneck width.
        n = min(int(contractive_samples), xb.shape[0])
        x_sub = xb[:n]
        z_sub = model.encode(x_sub)
        probe = torch.empty_like(z_sub).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        projected = (z_sub * probe).sum()
        grad_x = torch.autograd.grad(
            projected,
            x_sub,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return grad_x.square().sum(dim=1).mean() / z_sub.shape[1]

    raise ValueError(f"Unsupported regularizer: {regularizer}")


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
    contractive_samples,
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
        "train_regularizer": [],
        "train_total_loss": [],
        "val_mse": [],
    }

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    print("")
    print("=" * 80)
    print(f"Run {run_idx:03d}")
    print(config)
    print("=" * 80)

    for epoch in range(1, epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_count = 0

        reg_sum = 0.0
        total_loss_sum = 0.0

        for (xb,) in train_loader:
            xb = xb.to(device)
            if config["regularizer"] == "contractive":
                xb = xb.detach().requires_grad_(True)

            x_hat, z = model(xb)
            reconstruction_loss = criterion(x_hat, xb)
            reg_penalty = regularization_penalty(
                model=model,
                xb=xb,
                z=z,
                regularizer=config["regularizer"],
                contractive_samples=contractive_samples,
            )
            loss = reconstruction_loss + config["reg_lambda"] * reg_penalty

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_n = xb.shape[0]
            train_loss_sum += reconstruction_loss.detach().item() * batch_n
            reg_sum += reg_penalty.detach().item() * batch_n
            total_loss_sum += loss.detach().item() * batch_n
            train_count += batch_n

        train_loss = train_loss_sum / train_count
        train_reg = reg_sum / train_count
        train_total = total_loss_sum / train_count

        val_errors, _ = evaluate_errors(
            model=model,
            x=val_x,
            batch_size=config["batch_size"],
            device=device,
        )

        val_loss = float(np.mean(val_errors))

        history["epoch"].append(epoch)
        history["train_mse"].append(train_loss)
        history["train_regularizer"].append(train_reg)
        history["train_total_loss"].append(train_total)
        history["val_mse"].append(val_loss)

        improved = val_loss < (best_val - train_one_model.min_delta)

        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print(
                f"run {run_idx:03d} | "
                f"epoch {epoch:04d} | "
                f"train mse {train_loss:.6e} | "
                f"reg {train_reg:.6e} | "
                f"total {train_total:.6e} | "
                f"val mse {val_loss:.6e} | "
                f"best val {best_val:.6e} at epoch {best_epoch}"
            )

        if epochs_without_improvement >= train_one_model.patience:
            print(
                f"run {run_idx:03d} | early stopping at epoch {epoch}; "
                f"best validation MSE was {best_val:.6e} at epoch {best_epoch}"
            )
            break

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
        "regularizer": config["regularizer"],
        "reg_lambda": float(config["reg_lambda"]),
        "contractive_samples": int(contractive_samples),
        "epochs_requested": int(epochs),
        "epochs_completed": int(history["epoch"][-1]),
        "early_stopping_patience": int(train_one_model.patience),
        "early_stopping_min_delta": float(train_one_model.min_delta),
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
        metadata=feature_metadata(train_one_model.history_len),
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
        writer.writerow([
            "epoch", "train_mse", "train_regularizer",
            "train_total_loss", "val_mse"
        ])

        for e, tr, rg, total, va in zip(
            history["epoch"],
            history["train_mse"],
            history["train_regularizer"],
            history["train_total_loss"],
            history["val_mse"],
        ):
            writer.writerow([e, tr, rg, total, va])

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

    parser.add_argument("--history_len", type=int, default=10)

    parser.add_argument(
        "--hidden_dims_grid",
        type=str,
        default="256,128,64",
    )

    parser.add_argument("--latent_dims", type=str, default="3,6,12")
    parser.add_argument("--activations", type=str, default="elu")
    parser.add_argument("--learning_rates", type=str, default="3e-4")
    parser.add_argument("--weight_decays", type=str, default="1e-6")
    parser.add_argument("--batch_sizes", type=str, default="2048")

    parser.add_argument(
        "--regularizers", type=str, default="sparse,contractive",
        help="Comma-separated: none,sparse,contractive",
    )
    parser.add_argument(
        "--sparse_lambdas", type=str, default="1e-4,1e-3",
        help="L1 latent activity coefficients.",
    )
    parser.add_argument(
        "--contractive_lambdas", type=str, default="1e-4,1e-3",
        help="Encoder-Jacobian coefficients.",
    )
    parser.add_argument(
        "--contractive_samples", type=int, default=256,
        help="Samples per batch used for the Jacobian estimate.",
    )

    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--min_delta", type=float, default=1e-7)

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
    print("Regularized autoencoder sweep")
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
    train_one_model.history_len = args.history_len
    train_one_model.patience = args.patience
    train_one_model.min_delta = args.min_delta

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
            contractive_samples=args.contractive_samples,
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