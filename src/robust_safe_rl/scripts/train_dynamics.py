"""Train the residual dynamics model on collected (history, a_res) data.

Supervised regression with MSE loss (spectral-norm regularization optional via a
flag). Reports the in-distribution prediction-error distribution, which is the
key diagnostic: it tells us (a) whether the history successfully resolved the
k-ambiguity -- sharp predictions, low error -- versus collapsing to a washed-out
mean, and (b) what a future OOD-detection threshold would look like (e.g. the
99th percentile of the ID error becomes a natural threshold).

Usage:
  python -m robust_safe_rl.scripts.train_dynamics \
      --data data/dyn_massmoi.npz --out runs_dynamics/massmoi --epochs 100
  # add --spectral_norm to enable Lipschitz regularization
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from robust_safe_rl.rl.residual_dynamics import ResidualDynamics
from robust_safe_rl.rl.run_utils import resolve_run_dir_args, checkpoint_name


def load_data(path, val_frac=0.15, seed=0):
    d = np.load(path)
    X, Y, k = d["X"], d["Y"], d["k"]
    history = int(d["history"])
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return (X[train_idx], Y[train_idx], k[train_idx],
            X[val_idx], Y[val_idx], k[val_idx], history)


def train(args):
    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available())
                          else "cpu")

    # Organized run directory: <runs_root>/<run_name>/trial_<N>/ with auto-
    # incrementing trial index (nothing is overwritten). Mirrors the residual-
    # policy scheme so dynamics runs are laid out the same way.
    run_dir, trial = resolve_run_dir_args(args.runs_root, args.run_name, args.trial)
    print(f"run: {args.run_name}  trial: {trial}")
    print(f"artifacts -> {run_dir}")

    # dump the run config for reproducibility
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({
            "data": os.path.abspath(args.data), "epochs": args.epochs,
            "batch_size": args.batch_size, "lr": args.lr, "hidden": list(args.hidden),
            "spectral_norm": args.spectral_norm, "seed": args.seed,
            "run_name": args.run_name, "trial": trial, "runs_root": args.runs_root,
        }, f, indent=2)

    Xtr, Ytr, ktr, Xva, Yva, kva, history = load_data(args.data, seed=args.seed)
    print(f"train {Xtr.shape[0]}  val {Xva.shape[0]}  history {history}")

    # Standardize targets per-dimension using TRAIN stats, so all 6 output dims
    # contribute comparably to the loss (raw a_res dims span ~0.07 to ~30 in std,
    # which otherwise lets the large-scale dims swamp the small ones). Predictions
    # are un-standardized before reporting errors in physical units. Stats are
    # saved with the model so it can be used downstream.
    y_mean = Ytr.mean(axis=0)
    y_std = Ytr.std(axis=0) + 1e-6
    Ytr_n = (Ytr - y_mean) / y_std
    Yva_n = (Yva - y_mean) / y_std

    model = ResidualDynamics(history=history, hidden=tuple(args.hidden),
                             spectral_norm=args.spectral_norm).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.as_tensor(Xtr, device=device)
    Ytr_t = torch.as_tensor(Ytr_n, device=device)
    Xva_t = torch.as_tensor(Xva, device=device)
    Yva_t = torch.as_tensor(Yva_n, device=device)   # normalized (for loss)
    Yva_phys = torch.as_tensor(Yva, device=device)  # physical (for reporting)
    y_std_t = torch.as_tensor(y_std, device=device)
    y_mean_t = torch.as_tensor(y_mean, device=device)

    n = Xtr_t.shape[0]
    bs = args.batch_size
    log = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        ep_loss = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            pred = model(Xtr_t[b])
            loss = loss_fn(pred, Ytr_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item() * b.numel()
        ep_loss /= n

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                pred_n = model(Xva_t)                      # normalized prediction
                pred_va = pred_n * y_std_t + y_mean_t      # back to physical units
                # per-sample L2 error over the 6-dim target (physical units)
                err = torch.norm(pred_va - Yva_phys, dim=1).cpu().numpy()
                err_v = torch.norm(pred_va[:, :3] - Yva_phys[:, :3], dim=1).cpu().numpy()
                err_w = torch.norm(pred_va[:, 3:] - Yva_phys[:, 3:], dim=1).cpu().numpy()
            rec = {
                "epoch": epoch,
                "train_mse": ep_loss,
                "val_err_mean": float(err.mean()),
                "val_err_p50": float(np.percentile(err, 50)),
                "val_err_p99": float(np.percentile(err, 99)),
                "val_err_trans_mean": float(err_v.mean()),
                "val_err_ang_mean": float(err_w.mean()),
            }
            log.append(rec)
            print(f"epoch {epoch:4d}  train_mse {ep_loss:.5f}  "
                  f"val_err mean {rec['val_err_mean']:.5f}  p99 {rec['val_err_p99']:.5f}  "
                  f"(trans {rec['val_err_trans_mean']:.5f}  ang {rec['val_err_ang_mean']:.5f})")

    ckpt_path = os.path.join(run_dir, checkpoint_name(args.run_name, trial))
    torch.save({"model": model.state_dict(), "history": history,
                "hidden": list(args.hidden), "spectral_norm": args.spectral_norm,
                "y_mean": y_mean, "y_std": y_std},
               ckpt_path)
    with open(os.path.join(run_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    # Save the in-distribution error distribution for later OOD threshold setting.
    model.eval()
    with torch.no_grad():
        pred_all = model(Xva_t) * y_std_t + y_mean_t
        err_all = torch.norm(pred_all - Yva_phys, dim=1).cpu().numpy()
    np.savez(os.path.join(run_dir, "id_error_dist.npz"), err=err_all, k=kva)
    print(f"\nID val error: mean {err_all.mean():.5f}  p95 {np.percentile(err_all,95):.5f}  "
          f"p99 {np.percentile(err_all,99):.5f}  max {err_all.max():.5f}")
    print(f"  -> a natural OOD threshold candidate is around p99 = {np.percentile(err_all,99):.5f}")
    print(f"checkpoint -> {ckpt_path}")
    print(f"artifacts  -> {run_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--runs_root", default="runs_dynamics",
                   help="root folder for dynamics-model runs")
    p.add_argument("--run_name", default="dynamics",
                   help="experiment name; artifacts go to <runs_root>/<run_name>/trial_<N>/")
    p.add_argument("--trial", type=int, default=None,
                   help="force a trial index (default: auto-increment)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, nargs="+", default=[256, 256])
    p.add_argument("--spectral_norm", action="store_true",
                   help="enable spectral-norm (Lipschitz) regularization")
    p.add_argument("--eval_every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()