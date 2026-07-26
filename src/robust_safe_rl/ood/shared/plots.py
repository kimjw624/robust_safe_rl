"""Shared plotting helpers for autoencoder OOD diagnostics.

These produce the reconstruction-error histogram, sorted-error curve, and 2-D /
3-D latent-space scatter plots used by the training, testing, and evaluation
scripts. All figures are written to disk with the Agg backend, so this is safe
to call in headless / batch runs.
"""

from pathlib import Path

import numpy as np


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

