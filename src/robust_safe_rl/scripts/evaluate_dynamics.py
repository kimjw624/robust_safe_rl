"""Evaluate a trained residual-dynamics model and organize the results.

Loads a dynamics-model checkpoint and measures its residual-acceleration
prediction error on one or more datasets. The intended use is out-of-
distribution (OOD) detection analysis: compare the prediction-error
distribution on in-distribution data (the disturbance family the model was
trained on) against error on OOD data (a disturbance type it never saw). If the
OOD errors sit clearly above the ID error's high percentile, the model doubles
as an OOD detector.

Artifacts are written into <checkpoint_dir>/eval/<eval_tag>/, mirroring the
residual-policy evaluation layout, so everything for one evaluated model stays
in one organized folder:

    summary.json          per-dataset error stats + suggested threshold
    error_hist.png        overlaid ID vs OOD error histograms
    error_vs_k.png        prediction error against the disturbance factor k
    <dataset>_err.npz     raw per-sample errors for custom analysis

Usage:
  # ID only (characterize the training distribution's error)
  python -m robust_safe_rl.scripts.evaluate_dynamics \
      --checkpoint runs_dynamics/massmoi/trial_1/massmoi_trial_1.pt \
      --id_data data/dyn_massmoi_val.npz

  # ID vs OOD (the detection test)
  python -m robust_safe_rl.scripts.evaluate_dynamics \
      --checkpoint runs_dynamics/massmoi/trial_1/massmoi_trial_1.pt \
      --id_data data/dyn_massmoi.npz --ood_data data/dyn_force.npz \
      --eval_tag massmoi_vs_force
"""

import argparse
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robust_safe_rl.rl.residual_dynamics import ResidualDynamics


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = ResidualDynamics(history=ckpt["history"], hidden=tuple(ckpt["hidden"]),
                             spectral_norm=ckpt.get("spectral_norm", False)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    y_mean = torch.as_tensor(ckpt["y_mean"], device=device)
    y_std = torch.as_tensor(ckpt["y_std"], device=device)
    return model, y_mean, y_std, ckpt["history"]


def prediction_error(model, y_mean, y_std, data_path, device, batch=8192):
    """Per-sample L2 error (physical units) of the model on a dataset."""
    d = np.load(data_path)
    X, Y, k = d["X"], d["Y"], d["k"]
    errs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = torch.as_tensor(X[i:i + batch], device=device)
            yb = torch.as_tensor(Y[i:i + batch], device=device)
            pred = model(xb) * y_std + y_mean          # un-standardize to physical
            errs.append(torch.norm(pred - yb, dim=1).cpu().numpy())
    return np.concatenate(errs), k


def stats(err):
    return {
        "n": int(err.size),
        "mean": float(err.mean()),
        "p50": float(np.percentile(err, 50)),
        "p95": float(np.percentile(err, 95)),
        "p99": float(np.percentile(err, 99)),
        "max": float(err.max()),
    }


def evaluate(checkpoint, id_data, ood_data=None, eval_tag=None, device="cpu"):
    device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint))
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint))[0]
    eval_tag = eval_tag or f"eval_{ckpt_stem}"
    out_dir = os.path.join(ckpt_dir, "eval", eval_tag)
    os.makedirs(out_dir, exist_ok=True)

    model, y_mean, y_std, history = load_model(checkpoint, device)

    id_err, id_k = prediction_error(model, y_mean, y_std, id_data, device)
    np.savez(os.path.join(out_dir, "id_err.npz"), err=id_err, k=id_k)
    result = {"checkpoint": os.path.abspath(checkpoint), "history": history,
              "id": stats(id_err)}

    # A natural OOD threshold candidate: high percentile of the ID error.
    threshold = float(np.percentile(id_err, 99))
    result["suggested_threshold_p99"] = threshold

    ood_err = ood_k = None
    if ood_data is not None:
        ood_err, ood_k = prediction_error(model, y_mean, y_std, ood_data, device)
        np.savez(os.path.join(out_dir, "ood_err.npz"), err=ood_err, k=ood_k)
        result["ood"] = stats(ood_err)
        # detection rate: fraction of OOD samples above the ID-p99 threshold
        result["ood_detect_rate_at_p99"] = float(np.mean(ood_err > threshold))
        result["id_false_positive_at_p99"] = float(np.mean(id_err > threshold))

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(result, f, indent=2)

    # ---- plots ----
    fig = plt.figure(figsize=(8, 5))
    bins = np.linspace(0, np.percentile(id_err, 99.5) * (3 if ood_err is not None else 1.2), 80)
    plt.hist(id_err, bins=bins, alpha=0.6, label="in-distribution", density=True)
    if ood_err is not None:
        plt.hist(ood_err, bins=bins, alpha=0.6, label="out-of-distribution", density=True)
    plt.axvline(threshold, color="k", ls="--", lw=1, label="ID p99 threshold")
    plt.xlabel("prediction error  ||a_res_pred - a_res_true||")
    plt.ylabel("density"); plt.title("Residual-dynamics prediction error")
    plt.legend(); plt.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "error_hist.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.scatter(id_k, id_err, s=3, alpha=0.3, label="in-distribution")
    if ood_err is not None:
        plt.scatter(ood_k, ood_err, s=3, alpha=0.3, label="out-of-distribution")
    plt.axhline(threshold, color="k", ls="--", lw=1, label="ID p99 threshold")
    plt.xlabel("disturbance factor k"); plt.ylabel("prediction error")
    plt.title("Prediction error vs disturbance"); plt.legend(); plt.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, "error_vs_k.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"ID error   : mean {result['id']['mean']:.5f}  p99 {result['id']['p99']:.5f}")
    if ood_err is not None:
        print(f"OOD error  : mean {result['ood']['mean']:.5f}  p99 {result['ood']['p99']:.5f}")
        print(f"OOD detect rate @ ID-p99 threshold: {result['ood_detect_rate_at_p99']:.3f}  "
              f"(ID false-positive: {result['id_false_positive_at_p99']:.3f})")
    print(f"eval artifacts -> {out_dir}")
    return out_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="dynamics-model .pt checkpoint")
    p.add_argument("--id_data", required=True, help="in-distribution dataset .npz")
    p.add_argument("--ood_data", default=None, help="optional OOD dataset .npz")
    p.add_argument("--eval_tag", default=None,
                   help="subfolder under eval/ (default: eval_<ckpt stem>)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    evaluate(args.checkpoint, args.id_data, args.ood_data, args.eval_tag, args.device)


if __name__ == "__main__":
    main()