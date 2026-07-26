"""Evaluate a force-trained autoencoder on unseen mass / inertia OOD.

Tests whether a detector trained only on external-force disturbances flags a
different disturbance type: episode-constant mass and moment-of-inertia scaling
(with forces kept in-distribution). Reports AUROC / average precision,
threshold-based confusion metrics, per-episode error-vs-severity summaries, and
an extensive set of latent-space diagnostic plots (including ID-scale zoom and
symlog views, since parameter-OOD points can sit far from the ID cluster).
"""

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from robust_safe_rl.ood import (
    AutoEncoder,
    STEP_FEATURE_DIM,
    apply_standardization,
    evaluate_errors,
    collect_parameter_dataset,
    binary_curves,
    summarize,
    threshold_metrics,
    NOMINAL_MASS,
    NOMINAL_J,
)
from robust_safe_rl.ood.shared.io_utils import get_checkpoint_by_index


def per_episode_summary(errors, episode_ids, mass_scales, moi_scales, forces):
    rows = []
    for episode_id in np.unique(episode_ids):
        mask = episode_ids == episode_id
        rows.append({
            "episode_id": int(episode_id),
            "mass_scale": float(mass_scales[mask][0]),
            "moi_scale_x": float(moi_scales[mask][0, 0]),
            "moi_scale_y": float(moi_scales[mask][0, 1]),
            "moi_scale_z": float(moi_scales[mask][0, 2]),
            "force_x": float(forces[mask][0, 0]),
            "force_y": float(forces[mask][0, 1]),
            "force_z": float(forces[mask][0, 2]),
            "severity": float(max(
                abs(mass_scales[mask][0] - 1.0),
                np.max(np.abs(moi_scales[mask][0] - 1.0)),
            )),
            "mean_error": float(np.mean(errors[mask])),
            "median_error": float(np.median(errors[mask])),
            "p95_error": float(np.percentile(errors[mask], 95)),
            "max_error": float(np.max(errors[mask])),
        })
    return rows


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plots(out_dir, prefix, id_errors, ood_errors, id_latent, ood_latent,
               threshold_rows, curves, episode_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p995 = float(np.percentile(np.concatenate([id_errors, ood_errors]), 99.5))
    p995 = max(p995, np.finfo(float).eps)
    p99_row = next((r for r in threshold_rows if r["threshold_name"] == "id_p99"), None)
    threshold = p99_row["threshold"] if p99_row else float(np.percentile(id_errors, 99))

    plt.figure()
    plt.hist(id_errors[id_errors <= p995], bins=100, alpha=0.6, label="ID force only")
    plt.hist(ood_errors[ood_errors <= p995], bins=100, alpha=0.6, label="Mass + MOI OOD")
    plt.axvline(threshold, linestyle="--", label="checkpoint ID p99")
    plt.xlabel("reconstruction MSE")
    plt.ylabel("count")
    plt.title("Reconstruction error (x-axis clipped at combined p99.5)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_reconstruction_hist_clipped.png", dpi=200)
    plt.close()

    positive = np.concatenate([id_errors, ood_errors])
    positive = positive[positive > 0]
    min_positive = float(np.min(positive)) if len(positive) else 1e-12
    bins = np.logspace(np.log10(min_positive), np.log10(max(float(np.max(positive)), min_positive * 10)), 100)
    plt.figure()
    plt.hist(np.maximum(id_errors, min_positive), bins=bins, alpha=0.6, label="ID force only")
    plt.hist(np.maximum(ood_errors, min_positive), bins=bins, alpha=0.6, label="Mass + MOI OOD")
    plt.axvline(max(threshold, min_positive), linestyle="--", label="checkpoint ID p99")
    plt.xscale("log")
    plt.xlabel("reconstruction MSE (log scale)")
    plt.ylabel("count")
    plt.title("Reconstruction error: ID vs mass/MOI OOD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_reconstruction_hist_log.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(np.sort(id_errors), label="ID force only")
    plt.plot(np.sort(ood_errors), label="Mass + MOI OOD")
    plt.axhline(threshold, linestyle="--", label="checkpoint ID p99")
    plt.yscale("log")
    plt.xlabel("sorted sample index")
    plt.ylabel("reconstruction MSE (log scale)")
    plt.title("Sorted reconstruction errors")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_sorted_errors_log.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(curves["fpr"], curves["tpr"], label=f"AUROC = {curves['auroc']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("ROC: mass/MOI OOD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_roc_curve.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(curves["recall"], curves["precision"], label=f"AP = {curves['average_precision']:.4f}")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title("Precision-recall: mass/MOI OOD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_precision_recall_curve.png", dpi=200)
    plt.close()

    if id_latent.shape[1] >= 2:
        dims = [(0, 1)]
        if id_latent.shape[1] >= 3:
            dims += [(0, 2), (1, 2)]
        for i, j in dims:
            all_x = np.concatenate([id_latent[:, i], ood_latent[:, i]])
            all_y = np.concatenate([id_latent[:, j], ood_latent[:, j]])
            xpad = max(0.05 * np.ptp(all_x), 1e-6)
            ypad = max(0.05 * np.ptp(all_y), 1e-6)
            xlim = (all_x.min() - xpad, all_x.max() + xpad)
            ylim = (all_y.min() - ypad, all_y.max() + ypad)
            for tag, groups in (
                ("id", [(id_latent, "ID force only")]),
                ("ood", [(ood_latent, "Mass + MOI OOD")]),
                ("both", [(id_latent, "ID force only"), (ood_latent, "Mass + MOI OOD")]),
            ):
                plt.figure()
                for z, label in groups:
                    stride = max(len(z) // 50000, 1)
                    plt.scatter(z[::stride, i], z[::stride, j], s=3, alpha=0.3, label=label)
                plt.xlim(*xlim)
                plt.ylim(*ylim)
                plt.xlabel(f"z{i + 1}")
                plt.ylabel(f"z{j + 1}")
                plt.title(f"Latent z{i + 1} vs z{j + 1}: {tag.upper()}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(out_dir / f"{prefix}_latent_z{i+1}_z{j+1}_{tag}.png", dpi=200)
                plt.close()

            # Additional close-up based only on the central ID distribution.
            # Extreme parameter-OOD points no longer determine the plot limits.
            id_x_lo, id_x_hi = np.percentile(id_latent[:, i], [0.5, 99.5])
            id_y_lo, id_y_hi = np.percentile(id_latent[:, j], [0.5, 99.5])
            id_x_pad = max(0.15 * (id_x_hi - id_x_lo), 1e-6)
            id_y_pad = max(0.15 * (id_y_hi - id_y_lo), 1e-6)
            zoom_xlim = (id_x_lo - id_x_pad, id_x_hi + id_x_pad)
            zoom_ylim = (id_y_lo - id_y_pad, id_y_hi + id_y_pad)
            ood_in_zoom = (
                (ood_latent[:, i] >= zoom_xlim[0])
                & (ood_latent[:, i] <= zoom_xlim[1])
                & (ood_latent[:, j] >= zoom_ylim[0])
                & (ood_latent[:, j] <= zoom_ylim[1])
            )

            plt.figure()
            id_stride = max(len(id_latent) // 50000, 1)
            ood_stride = max(len(ood_latent) // 50000, 1)
            plt.scatter(
                id_latent[::id_stride, i], id_latent[::id_stride, j],
                s=3, alpha=0.25, label="ID force only",
            )
            plt.scatter(
                ood_latent[::ood_stride, i], ood_latent[::ood_stride, j],
                s=3, alpha=0.20, label="Mass + MOI OOD",
            )
            plt.xlim(*zoom_xlim)
            plt.ylim(*zoom_ylim)
            plt.xlabel(f"z{i + 1}")
            plt.ylabel(f"z{j + 1}")
            plt.title(
                f"Latent z{i + 1} vs z{j + 1}: ID-scale zoom\n"
                f"OOD points inside view: {100.0 * np.mean(ood_in_zoom):.3f}%"
            )
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                out_dir / f"{prefix}_latent_z{i+1}_z{j+1}_both_id_scale_zoom.png",
                dpi=200,
            )
            plt.close()

            # A symmetric-log view preserves the full range while enlarging the center.
            plt.figure()
            plt.scatter(
                id_latent[::id_stride, i], id_latent[::id_stride, j],
                s=3, alpha=0.25, label="ID force only",
            )
            plt.scatter(
                ood_latent[::ood_stride, i], ood_latent[::ood_stride, j],
                s=3, alpha=0.20, label="Mass + MOI OOD",
            )
            plt.xscale("symlog", linthresh=max(float(np.std(id_latent[:, i])), 1e-3))
            plt.yscale("symlog", linthresh=max(float(np.std(id_latent[:, j])), 1e-3))
            plt.xlabel(f"z{i + 1} (symlog)")
            plt.ylabel(f"z{j + 1} (symlog)")
            plt.title(f"Latent z{i + 1} vs z{j + 1}: full range, symlog zoom")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                out_dir / f"{prefix}_latent_z{i+1}_z{j+1}_both_symlog.png",
                dpi=200,
            )
            plt.close()

    severity = np.asarray([r["severity"] for r in episode_rows])
    mean_error = np.asarray([r["mean_error"] for r in episode_rows])
    mass_scale = np.asarray([r["mass_scale"] for r in episode_rows])
    moi_scale = np.asarray([
        np.mean([r["moi_scale_x"], r["moi_scale_y"], r["moi_scale_z"]])
        for r in episode_rows
    ])

    plt.figure()
    plt.scatter(severity, mean_error, s=18, alpha=0.65)
    plt.yscale("log")
    plt.xlabel("parameter severity = max relative deviation from 1")
    plt.ylabel("episode mean reconstruction MSE (log scale)")
    plt.title("Detection score versus parameter change")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_error_vs_parameter_severity.png", dpi=200)
    plt.close()

    plt.figure()
    plt.scatter(mass_scale, mean_error, s=25, alpha=0.75)
    plt.yscale("log")
    plt.xlabel("common mass/MOI multiplier")
    plt.ylabel("episode mean reconstruction MSE (log scale)")
    plt.title("Reconstruction error versus common physical-parameter scale")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_error_vs_common_scale.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a force-trained autoencoder on unseen mass and inertia variations."
    )
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--suffix", type=str, default="ae_disturbance.pt")
    parser.add_argument("--output_dir", type=str, default="runs/mass_moi_ood_evaluation")

    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tf", type=float, default=10.0)
    parser.add_argument("--history_len", type=int, default=10)
    parser.add_argument("--id_episodes", type=int, default=200)
    parser.add_argument("--ood_episodes", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=70000)

    parser.add_argument(
        "--scale_min", type=float, default=0.7,
        help="Minimum common multiplier applied to both mass and the full MOI matrix.",
    )
    parser.add_argument(
        "--scale_max", type=float, default=1.3,
        help="Maximum common multiplier applied to both mass and the full MOI matrix.",
    )
    args = parser.parse_args()

    if not (0 < args.scale_min <= args.scale_max):
        raise ValueError("Invalid common mass/MOI scale range")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = get_checkpoint_by_index(args.run_dir, args.index, args.suffix)
    print(f"Device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")

    model, mean, std, checkpoint_thresholds, metadata = AutoEncoder.load(
        checkpoint_path, map_location=device
    )
    model.to(device).eval()
    if mean is None or std is None:
        raise RuntimeError("Checkpoint does not contain normalization mean/std")
    if metadata.get("feature_version") != 2:
        raise RuntimeError("Checkpoint does not use corrected feature version 2")
    if int(metadata.get("history_len", -1)) != args.history_len:
        raise ValueError(
            f"history_len mismatch: checkpoint={metadata.get('history_len')}, requested={args.history_len}"
        )

    rng = np.random.default_rng(args.seed)
    max_episodes = max(args.id_episodes, args.ood_episodes)
    shared_forces = rng.uniform(-3.0, 3.0, size=(max_episodes, 3))

    id_mass_scales = np.ones(args.id_episodes)
    id_moi_scales = np.ones((args.id_episodes, 3))

    # One episode-constant multiplier is shared by mass and every principal MOI.
    # Thus m_true = scale * m_nominal and J_true = scale * J_nominal.
    common_scales = rng.uniform(
        args.scale_min, args.scale_max, size=args.ood_episodes
    )
    ood_mass_scales = common_scales.copy()
    ood_moi_scales = np.repeat(common_scales[:, None], 3, axis=1)

    episode_steps = int(round(args.tf / args.dt))
    print("\nCollecting ID reference: force in [-3,3] N, nominal mass and MOI")
    id_data = collect_parameter_dataset(
        forces=shared_forces[:args.id_episodes],
        mass_scales=id_mass_scales,
        moi_scales=id_moi_scales,
        episode_steps=episode_steps,
        dt=args.dt,
        history_len=args.history_len,
        seed=args.seed + 1000,
    )

    print("Collecting parameter OOD: same force range plus mass/MOI multipliers")
    ood_data = collect_parameter_dataset(
        forces=shared_forces[:args.ood_episodes],
        mass_scales=ood_mass_scales,
        moi_scales=ood_moi_scales,
        episode_steps=episode_steps,
        dt=args.dt,
        history_len=args.history_len,
        seed=args.seed + 2000,
    )

    mean = mean.cpu()
    std = std.cpu()
    id_x = apply_standardization(torch.from_numpy(id_data["x"]), mean, std)
    ood_x = apply_standardization(torch.from_numpy(ood_data["x"]), mean, std)

    id_errors, id_latent = evaluate_errors(model, id_x, args.batch_size, device)
    ood_errors, ood_latent = evaluate_errors(model, ood_x, args.batch_size, device)

    curves = binary_curves(id_errors, ood_errors)
    threshold_rows = threshold_metrics(id_errors, ood_errors, checkpoint_thresholds)
    id_episode_rows = per_episode_summary(
        id_errors, id_data["episode_id"], id_data["mass_scale"],
        id_data["moi_scale"], id_data["force"]
    )
    ood_episode_rows = per_episode_summary(
        ood_errors, ood_data["episode_id"], ood_data["mass_scale"],
        ood_data["moi_scale"], ood_data["force"]
    )
    for row in id_episode_rows:
        row["class"] = "id_force_only"
    for row in ood_episode_rows:
        row["class"] = "mass_moi_ood"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.index:03d}_mass_moi_ood"

    metrics = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata": metadata,
        "configuration": vars(args),
        "nominal_mass": NOMINAL_MASS,
        "nominal_J_diagonal": np.diag(NOMINAL_J).tolist(),
        "id_definition": "external force uniform in [-3,3] N; nominal mass and MOI",
        "ood_definition": (
            "same ID force range plus one episode-constant common multiplier s applied "
            "to both mass and the full MOI matrix: m_true=s*m_nominal, J_true=s*J_nominal"
        ),
        "note": (
            "Uniform [0.7,1.3] includes values arbitrarily close to 1.0, so mild parameter "
            "changes may be physically indistinguishable and need not produce large errors."
        ),
        "id_reconstruction_error": summarize(id_errors),
        "mass_moi_ood_reconstruction_error": summarize(ood_errors),
        "error_mean_ratio_ood_over_id": float(np.mean(ood_errors) / max(np.mean(id_errors), 1e-15)),
        "auroc": curves["auroc"],
        "average_precision": curves["average_precision"],
        "threshold_metrics": threshold_rows,
    }

    with (output_dir / f"{prefix}_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)
    write_csv(output_dir / f"{prefix}_threshold_metrics.csv", threshold_rows)
    write_csv(output_dir / f"{prefix}_episode_metrics.csv", id_episode_rows + ood_episode_rows)

    np.savez_compressed(
        output_dir / f"{prefix}_raw_results.npz",
        id_errors=id_errors,
        ood_errors=ood_errors,
        id_latent=id_latent,
        ood_latent=ood_latent,
        id_episode_id=id_data["episode_id"],
        ood_episode_id=ood_data["episode_id"],
        id_forces=id_data["force"],
        ood_forces=ood_data["force"],
        ood_common_scales=ood_data["mass_scale"],
        ood_mass_scales=ood_data["mass_scale"],
        ood_moi_scales=ood_data["moi_scale"],
        roc_fpr=curves["fpr"],
        roc_tpr=curves["tpr"],
        pr_precision=curves["precision"],
        pr_recall=curves["recall"],
    )

    save_plots(
        output_dir, prefix, id_errors, ood_errors, id_latent, ood_latent,
        threshold_rows, curves, ood_episode_rows
    )

    print("\nReconstruction-error summary")
    print(
        f"ID       mean={np.mean(id_errors):.6e}, p95={np.percentile(id_errors,95):.6e}, "
        f"p99={np.percentile(id_errors,99):.6e}, max={np.max(id_errors):.6e}"
    )
    print(
        f"Mass/MOI mean={np.mean(ood_errors):.6e}, p95={np.percentile(ood_errors,95):.6e}, "
        f"p99={np.percentile(ood_errors,99):.6e}, max={np.max(ood_errors):.6e}"
    )
    print(f"AUROC: {curves['auroc']:.6f}")
    print(f"Average precision: {curves['average_precision']:.6f}")

    print("\nThreshold metrics")
    for row in threshold_rows:
        print(
            f"{row['threshold_name']:18s} threshold={row['threshold']:.6e} | "
            f"ID false alarm={row['id_false_alarm_rate']:.3%} | "
            f"OOD detected={row['ood_detection_rate']:.3%}"
        )

    print(f"\nSaved all results to: {output_dir}")
    print("Upload the metrics JSON, threshold CSV, episode CSV, and plots for analysis.")


if __name__ == "__main__":
    main()
