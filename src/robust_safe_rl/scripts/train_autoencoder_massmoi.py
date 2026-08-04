"""Train a reconstruction autoencoder with MASS/MOI as the in-distribution family.

This is the mass/MOI-ID counterpart to ``train_autoencoder.py`` (which uses
force as ID). Here the in-distribution family is episode-constant mass and
moment-of-inertia scaling -- matching the disturbance the residual policy and the
residual-dynamics detector were trained on -- so the autoencoder learns the same
notion of "normal" as the rest of the pipeline. Out-of-distribution is a strict
external force.

ID  : mass/MOI multipliers ~ U[k_min, k_max], forces kept small (ID band).
OOD : strict external force (mode 3), mass/MOI nominal.

Artifacts land in <runs_root>/<run_name>/trial_<N>/ with plots, mirroring the
other components.

Usage:
    python -m robust_safe_rl.scripts.train_autoencoder_massmoi \
        --history_len 10 --latent_dim 6 --run_name massmoi_ae
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from robust_safe_rl.ood import (
    AutoEncoder,
    standardize,
    evaluate_errors,
    detection_report,
    feature_metadata,
    STEP_FEATURE_DIM,
)
from robust_safe_rl.ood.shared.dataset import collect_parameter_dataset, collect_dataset
from robust_safe_rl.ood.shared.io_utils import get_next_indexed_path
from robust_safe_rl.ood.shared.plots import save_training_plots


def _massmoi_forces_scales(n_episodes, dt, k_min, k_max, id_force_band, seed):
    """Sample per-episode mass/MOI multipliers and small ID forces."""
    rng = np.random.default_rng(seed)
    mass_scales = rng.uniform(k_min, k_max, size=n_episodes)
    moi_scales = mass_scales.copy()   # couple mass and MOI by the same k (as in training)
    # small in-distribution force per episode (or zero if band == 0)
    if id_force_band > 0:
        forces = rng.uniform(-id_force_band, id_force_band, size=(n_episodes, 3))
    else:
        forces = np.zeros((n_episodes, 3))
    return forces, mass_scales, moi_scales


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tf", type=float, default=10.0)
    parser.add_argument("--train_episodes", type=int, default=200)
    parser.add_argument("--val_episodes", type=int, default=50)
    parser.add_argument("--ood_episodes", type=int, default=50)
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--hidden_dims", type=str, default="256,128,64")
    parser.add_argument("--latent_dim", type=int, default=6)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1)
    # ID mass/MOI range and the (small) ID force band
    parser.add_argument("--k_min", type=float, default=0.5)
    parser.add_argument("--k_max", type=float, default=1.5)
    parser.add_argument("--id_force_band", type=float, default=0.0,
                        help="magnitude of small ID forces (0 = no force in ID)")
    # run organization
    parser.add_argument("--runs_root", type=str, default="runs_autoencoder")
    parser.add_argument("--run_name", type=str, default="massmoi_ae")
    parser.add_argument("--trial", type=int, default=None)
    parser.add_argument("--save_suffix", type=str, default="ae_massmoi.pt")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    episode_steps = int(round(args.tf / args.dt))
    hidden_dims = tuple(int(x) for x in args.hidden_dims.split(",") if x.strip())

    trial = args.trial
    if trial is None:
        trial = 1
        while os.path.exists(os.path.join(args.runs_root, args.run_name, f"trial_{trial}")):
            trial += 1
    run_dir = os.path.join(args.runs_root, args.run_name, f"trial_{trial}")
    plot_dir = os.path.join(run_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"run: {args.run_name}  trial: {trial}")
    print(f"artifacts -> {run_dir}")
    save_path, run_idx = get_next_indexed_path(run_dir, args.save_suffix)

    # ---- ID data: mass/MOI scaling (this is the "normal" the AE learns) ----
    tr_f, tr_m, tr_j = _massmoi_forces_scales(args.train_episodes, args.dt,
                                              args.k_min, args.k_max, args.id_force_band, args.seed)
    va_f, va_m, va_j = _massmoi_forces_scales(args.val_episodes, args.dt,
                                              args.k_min, args.k_max, args.id_force_band, args.seed + 1000)
    train_np = collect_parameter_dataset(
        forces=tr_f, mass_scales=tr_m, moi_scales=tr_j,
        episode_steps=episode_steps, dt=args.dt, history_len=args.history_len, seed=args.seed)["x"]
    val_np = collect_parameter_dataset(
        forces=va_f, mass_scales=va_m, moi_scales=va_j,
        episode_steps=episode_steps, dt=args.dt, history_len=args.history_len, seed=args.seed + 1000)["x"]

    # ---- OOD data: strict force, nominal mass/MOI ----
    ood_np, _ = collect_dataset(args.ood_episodes, episode_steps, args.dt, 3,
                                args.history_len, args.seed + 2000, False)

    train_x, val_x, ood_x, mean, std = standardize(
        torch.from_numpy(train_np), torch.from_numpy(val_np), torch.from_numpy(ood_np))

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
        "run_idx": run_idx, "dt": args.dt, "tf": args.tf,
        "id_family": "massmoi", "k_min": args.k_min, "k_max": args.k_max,
        "id_force_band": args.id_force_band, "ood_force_mode": 3,
    })
    model.save(save_path, mean=mean, std=std, thresholds=thresholds, metadata=metadata)
    save_training_plots(plot_dir, run_idx, id_errors, ood_errors, id_latent, ood_latent, thresholds)

    print(f"Saved model to: {save_path}")
    print(f"Input dimension: {train_x.shape[1]} ({STEP_FEATURE_DIM} x history_len {args.history_len})")
    for name, threshold, false_alarm, detected in lines:
        print(f"{name:18s} {threshold:.6e} | ID false alarm {false_alarm:.3%} | OOD detected {detected:.3%}")


if __name__ == "__main__":
    main()