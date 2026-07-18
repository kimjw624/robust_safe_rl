import argparse
from pathlib import Path

import numpy as np
import torch

from autoencoder import AutoEncoder
from train_autoencoder import (
    collect_dataset,
    apply_standardization,
    evaluate_errors,
    detection_report,
    save_training_plots,
)


def get_checkpoint_by_index(run_dir, index, suffix):
    run_dir = Path(run_dir)
    path = run_dir / f"{index:03d}_{suffix}"

    if not path.exists():
        raise FileNotFoundError(f"Could not find checkpoint: {path}")

    return path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--run_dir", type=str, default="runs")
    parser.add_argument("--plot_dir", type=str, default="runs/test_plots")
    parser.add_argument("--suffix", type=str, default="ae_disturbance.pt")

    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--tf", type=float, default=10.0)
    parser.add_argument("--id_episodes", type=int, default=50)
    parser.add_argument("--ood_episodes", type=int, default=50)
    parser.add_argument("--history_len", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=999)

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
        raise RuntimeError("Checkpoint does not contain input normalization mean/std.")

    mean = mean.to("cpu")
    std = std.to("cpu")

    episode_steps = int(args.tf / args.dt)

    print("Collecting test ID data: random_force=1, [-3, 3] N")

    id_np, id_forces = collect_dataset(
        num_episodes=args.id_episodes,
        episode_steps=episode_steps,
        dt=args.dt,
        random_force=1,
        history_len=args.history_len,
        seed=args.seed,
        force_sample_each_step=args.force_sample_each_step,
    )

    print("Collecting test strict OOD data: random_force=3")
    print("Strict OOD means at least one force axis exceeds 3 N.")

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

    fresh_thresholds, lines = detection_report(
        id_errors=id_errors,
        ood_errors=ood_errors,
    )

    print("\nLoaded checkpoint thresholds")

    for key, value in thresholds.items():
        print(f"{key:18s}: {value:.6e}")

    print("\nFresh test reconstruction-error summary")

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

    print("\nFresh test threshold report")
    print("threshold_name       threshold       ID false alarm       OOD detected")

    for name, th, fa, detected in lines:
        print(
            f"{name:18s} "
            f"{th:.6e} "
            f"{fa:18.3%} "
            f"{detected:18.3%}"
        )

    print("\nCheckpoint-threshold test report")
    print("threshold_name       threshold       ID false alarm       OOD detected")

    for name, th in thresholds.items():
        id_false_alarm = float(np.mean(id_errors > th))
        ood_detected = float(np.mean(ood_errors > th))

        print(
            f"{name:18s} "
            f"{th:.6e} "
            f"{id_false_alarm:18.3%} "
            f"{ood_detected:18.3%}"
        )

    save_training_plots(
        plot_dir=args.plot_dir,
        run_idx=args.index,
        id_errors=id_errors,
        ood_errors=ood_errors,
        id_latent=id_latent,
        ood_latent=ood_latent,
        thresholds=thresholds,
    )

    print(f"\nSaved test plots to: {args.plot_dir}")


if __name__ == "__main__":
    main()