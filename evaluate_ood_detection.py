import argparse
from pathlib import Path

import numpy as np
import torch

from autoencoder import AutoEncoder
from train_autoencoder import (
    collect_dataset,
    apply_standardization,
    evaluate_errors,
)


def get_checkpoint_by_index(run_dir, index, suffix):
    run_dir = Path(run_dir)
    path = run_dir / f"{index:03d}_{suffix}"

    if not path.exists():
        raise FileNotFoundError(f"Could not find checkpoint: {path}")

    return path


def compute_confusion_matrix(y_true, y_pred):
    """
    y_true:
        0 = ID
        1 = OOD

    y_pred:
        0 = predicted ID
        1 = predicted OOD
    """

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    return tn, fp, fn, tp


def print_detection_report(y_true, y_pred, threshold):
    tn, fp, fn, tp = compute_confusion_matrix(y_true, y_pred)

    total = tn + fp + fn + tp

    accuracy = (tp + tn) / max(total, 1)
    id_false_alarm_rate = fp / max(fp + tn, 1)
    ood_detection_rate = tp / max(tp + fn, 1)
    ood_miss_rate = fn / max(tp + fn, 1)

    print("\nOOD detection using threshold = max ID validation reconstruction error")
    print(f"threshold: {threshold:.6e}")

    print("\nConfusion matrix")
    print("Rows are true labels, columns are predicted labels.")
    print("")
    print("                 predicted ID     predicted OOD")
    print(f"true ID       {tn:12d}     {fp:13d}")
    print(f"true OOD      {fn:12d}     {tp:13d}")

    print("\nMetrics")
    print(f"accuracy:             {accuracy:.3%}")
    print(f"ID false alarm rate:  {id_false_alarm_rate:.3%}")
    print(f"OOD detection rate:   {ood_detection_rate:.3%}")
    print(f"OOD miss rate:        {ood_miss_rate:.3%}")


def save_eval_plots(
    plot_dir,
    index,
    id_errors,
    ood_errors,
    id_latent,
    ood_latent,
    threshold,
):
    import matplotlib
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 001 reconstruction error histogram
    plt.figure()
    plt.hist(id_errors, bins=100, alpha=0.6, label="ID")
    plt.hist(ood_errors, bins=100, alpha=0.6, label="Strict OOD")
    plt.axvline(threshold, linestyle="--", label="max ID threshold")
    plt.xlabel("reconstruction MSE")
    plt.ylabel("count")
    plt.title("Reconstruction error: ID vs strict OOD")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{index:03d}_001_reconstruction_hist_max_id.png", dpi=200)
    plt.close()

    # 002 sorted reconstruction errors
    plt.figure()
    plt.plot(np.sort(id_errors), label="ID")
    plt.plot(np.sort(ood_errors), label="Strict OOD")
    plt.axhline(threshold, linestyle="--", label="max ID threshold")
    plt.xlabel("sorted sample index")
    plt.ylabel("reconstruction MSE")
    plt.title("Sorted reconstruction errors")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{index:03d}_002_sorted_reconstruction_max_id.png", dpi=200)
    plt.close()

    # 003 latent z1-z2
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
        plt.savefig(plot_dir / f"{index:03d}_003_latent_z1_z2.png", dpi=200)
        plt.close()

    # 004 latent z1-z3
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
        plt.savefig(plot_dir / f"{index:03d}_004_latent_z1_z3.png", dpi=200)
        plt.close()

    # 005 latent z2-z3
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
        plt.savefig(plot_dir / f"{index:03d}_005_latent_z2_z3.png", dpi=200)
        plt.close()

    # 006 3D latent space
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
        ax.set_title("3D latent space: ID vs strict OOD")
        ax.legend()

        plt.tight_layout()
        plt.savefig(plot_dir / f"{index:03d}_006_latent_3d.png", dpi=200)
        plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--index", type=int, required=True)

    parser.add_argument("--run_dir", type=str, default="runs")
    parser.add_argument("--suffix", type=str, default="ae_disturbance.pt")
    parser.add_argument("--plot_dir", type=str, default="runs/eval_plots")

    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tf", type=float, default=10.0)
    parser.add_argument("--history_len", type=int, default=3)

    parser.add_argument("--id_episodes", type=int, default=50)
    parser.add_argument("--ood_episodes", type=int, default=50)

    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument("--force_sample_each_step", action="store_true")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = get_checkpoint_by_index(
        run_dir=args.run_dir,
        index=args.index,
        suffix=args.suffix,
    )

    print(f"Loading checkpoint: {ckpt_path}")

    model, mean, std, thresholds = AutoEncoder.load(
        ckpt_path,
        map_location=device,
    )

    model.to(device)
    model.eval()

    if mean is None or std is None:
        raise RuntimeError("Checkpoint does not contain mean/std normalization.")

    mean = mean.cpu()
    std = std.cpu()

    episode_steps = int(args.tf / args.dt)

    print("\nCollecting labeled ID test data")
    print("ID label = 0, force in [-3, 3] N")

    id_np, id_forces = collect_dataset(
        num_episodes=args.id_episodes,
        episode_steps=episode_steps,
        dt=args.dt,
        random_force=1,
        history_len=args.history_len,
        seed=args.seed,
        force_sample_each_step=args.force_sample_each_step,
    )

    print("\nCollecting labeled strict OOD test data")
    print("OOD label = 1, force in [-5, 5] N with at least one axis outside [-3, 3] N")

    ood_np, ood_forces = collect_dataset(
        num_episodes=args.ood_episodes,
        episode_steps=episode_steps,
        dt=args.dt,
        random_force=3,
        history_len=args.history_len,
        seed=args.seed + 1000,
        force_sample_each_step=args.force_sample_each_step,
    )

    id_x = torch.tensor(id_np, dtype=torch.float32)
    ood_x = torch.tensor(ood_np, dtype=torch.float32)

    id_x = apply_standardization(id_x, mean, std)
    ood_x = apply_standardization(ood_x, mean, std)

    id_errors, id_latent = evaluate_errors(
        model=model,
        x=id_x,
        batch_size=args.batch_size,
        device=device,
    )

    ood_errors, ood_latent = evaluate_errors(
        model=model,
        x=ood_x,
        batch_size=args.batch_size,
        device=device,
    )

    # Main threshold requested:
    # OOD if reconstruction error is greater than max ID reconstruction error.
    if "id_max" in thresholds:
        threshold = float(thresholds["id_max"])
        print("\nUsing checkpoint id_max threshold.")
    else:
        threshold = float(np.max(id_errors))
        print("\nCheckpoint did not contain id_max. Using max of this test ID set.")

    id_pred = (id_errors > threshold).astype(int)
    ood_pred = (ood_errors > threshold).astype(int)

    y_true = np.concatenate([
        np.zeros_like(id_pred),
        np.ones_like(ood_pred),
    ])

    y_pred = np.concatenate([
        id_pred,
        ood_pred,
    ])

    print("\nReconstruction-error summary")
    print(
        "ID  "
        f"mean={id_errors.mean():.6e}, "
        f"p95={np.percentile(id_errors, 95):.6e}, "
        f"p99={np.percentile(id_errors, 99):.6e}, "
        f"max={id_errors.max():.6e}"
    )

    print(
        "OOD "
        f"mean={ood_errors.mean():.6e}, "
        f"p95={np.percentile(ood_errors, 95):.6e}, "
        f"p99={np.percentile(ood_errors, 99):.6e}, "
        f"max={ood_errors.max():.6e}"
    )

    print_detection_report(
        y_true=y_true,
        y_pred=y_pred,
        threshold=threshold,
    )

    save_eval_plots(
        plot_dir=args.plot_dir,
        index=args.index,
        id_errors=id_errors,
        ood_errors=ood_errors,
        id_latent=id_latent,
        ood_latent=ood_latent,
        threshold=threshold,
    )

    print(f"\nSaved evaluation plots to: {args.plot_dir}")


if __name__ == "__main__":
    main()