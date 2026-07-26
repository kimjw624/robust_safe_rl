"""Train a reconstruction autoencoder to detect force-disturbance OOD.

Collects ID (force in [-3,3] N) training/validation rollouts and a strict-OOD
(at least one axis > 3 N) evaluation set, standardizes with train statistics,
trains the autoencoder, then reports ID-threshold-based detection rates and
saves the checkpoint (with normalization stats, thresholds, and feature
metadata) plus diagnostic plots.

Example:
    python -m robust_safe_rl.scripts.train_autoencoder \
        --history_len 10 --latent_dim 6 --run_dir runs --plot_dir runs/plots
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from robust_safe_rl.ood import (
    AutoEncoder,
    collect_dataset,
    standardize,
    evaluate_errors,
    detection_report,
    feature_metadata,
    STEP_FEATURE_DIM,
)
from robust_safe_rl.ood.shared.io_utils import get_next_indexed_path
from robust_safe_rl.ood.shared.plots import save_training_plots


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
