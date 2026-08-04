"""Evaluate a trained residual-SAC checkpoint: roll out, plot, and log.

Runs the greedy policy across a set of disturbance factors k, records full
trajectories, and writes organised artifacts into <run_dir>/eval/<eval_tag>/:

    summary.json            per-k metrics + aggregate
    summary.csv             same, flat table
    traj_k<k>.npz           raw arrays for each k (for custom re-plotting)
    k<k>_3d.png             3D trajectory: desired / nominal twin / true
    k<k>_topview.png        top-down (North-East) view
    k<k>_position.png       North/East/Down vs time (desired vs true)
    k<k>_pos_error.png      true-vs-nominal position error over time
    k<k>_residual.png       learned residual (df, dMx, dMy, dMz) over time
    k<k>_control.png        total thrust and moments applied to the true plant
    baseline_vs_residual.png   bar chart: mean tracking error with/without residual

Usage:
    python -m robust_safe_rl.scripts.evaluate_residual \
        --checkpoint runs_residual/dob_baseline/trial_1/dob_baseline_trial_1.pt

Everything for one evaluated model stays in a single folder.
"""

import argparse
import csv
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from robust_safe_rl.rl.config import Config
from robust_safe_rl.rl.residual_env import ResidualTwinEnv
from robust_safe_rl.rl.sac import SAC


# --------------------------------------------------------------------------- io
def load_agent(checkpoint, cfg, device="cpu"):
    env = ResidualTwinEnv(cfg.env, seed=0)
    agent = SAC(env.obs_dim, env.action_dim, cfg.sac, cfg.net, device=device)
    sd = torch.load(checkpoint, map_location=device)
    agent.load_state_dict(sd)
    return agent


def rollout(agent, cfg, k, use_residual=True):
    """Run one greedy episode at disturbance k. Return a dict of trajectory arrays."""
    env = ResidualTwinEnv(cfg.env, seed=999)
    obs = env.reset(k=k)

    t, x_true, x_nom, x_des = [], [], [], []
    v_true, v_nom = [], []
    residual, u_total, pos_err = [], [], []

    step = 0
    while True:
        a = agent.act(obs, deterministic=True) if use_residual else np.zeros(env.action_dim)

        # snapshot desired at current time before stepping
        desired = env.traj.desired(env.t)
        st = env.dyn_true.state()
        sn = env.dyn_nom.state()

        t.append(env.t)
        x_true.append(st["x"].copy())
        x_nom.append(sn["x"].copy())
        x_des.append(desired["x"].copy())
        v_true.append(st["v"].copy())
        v_nom.append(sn["v"].copy())

        obs, r, term, trunc, info = env.step(a)

        # residual (physical units) and total control now come from step info
        residual.append(np.asarray(info["residual"]).copy())
        u_total.append(np.asarray(info["u_total"]).copy())
        pos_err.append(info["pos_err"])

        step += 1
        if term or trunc:
            break

    return {
        "k": k,
        "t": np.asarray(t),
        "x_true": np.asarray(x_true),
        "x_nom": np.asarray(x_nom),
        "x_des": np.asarray(x_des),
        "v_true": np.asarray(v_true),
        "v_nom": np.asarray(v_nom),
        "residual": np.asarray(residual),
        "u_total": np.asarray(u_total),
        "pos_err": np.asarray(pos_err),
        "terminated": bool(term),
        "steps": step,
    }


# ------------------------------------------------------------------- plotting
def _set_axes_equal(ax):
    xl, yl, zl = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xr, yr, zr = abs(xl[1] - xl[0]), abs(yl[1] - yl[0]), abs(zl[1] - zl[0])
    r = 0.5 * max(xr, yr, zr)
    xm, ym, zm = np.mean(xl), np.mean(yl), np.mean(zl)
    ax.set_xlim3d(xm - r, xm + r)
    ax.set_ylim3d(ym - r, ym + r)
    ax.set_zlim3d(zm - r, zm + r)


def _ktag(k):
    return f"{k:.2f}".replace(".", "p")


def plot_rollout(d, out_dir):
    k = d["k"]
    tag = _ktag(k)
    t = d["t"]

    # 3D trajectory: desired / nominal twin / true
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(d["x_des"][:, 0], d["x_des"][:, 1], d["x_des"][:, 2], "--", label="desired")
    ax.plot(d["x_nom"][:, 0], d["x_nom"][:, 1], d["x_nom"][:, 2], label="nominal twin")
    ax.plot(d["x_true"][:, 0], d["x_true"][:, 1], d["x_true"][:, 2], label="true (residual)")
    ax.set_xlabel("North x [m]"); ax.set_ylabel("East y [m]"); ax.set_zlabel("Down z [m]")
    ax.set_title(f"3D trajectory (k={k:.2f})"); ax.legend(); ax.grid(True)
    _set_axes_equal(ax)
    fig.savefig(os.path.join(out_dir, f"k{tag}_3d.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Top view
    fig = plt.figure()
    plt.plot(d["x_des"][:, 0], d["x_des"][:, 1], "--", label="desired")
    plt.plot(d["x_nom"][:, 0], d["x_nom"][:, 1], label="nominal twin")
    plt.plot(d["x_true"][:, 0], d["x_true"][:, 1], label="true (residual)")
    plt.axis("equal"); plt.xlabel("North x [m]"); plt.ylabel("East y [m]")
    plt.title(f"Top view (k={k:.2f})"); plt.legend(); plt.grid(True)
    fig.savefig(os.path.join(out_dir, f"k{tag}_topview.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Position vs time (desired vs true), 3 stacked axes
    fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
    for i, name in enumerate(["North x", "East y", "Down z"]):
        axs[i].plot(t, d["x_des"][:, i], "--", label="desired")
        axs[i].plot(t, d["x_true"][:, i], label="true")
        axs[i].set_ylabel(f"{name} [m]"); axs[i].grid(True); axs[i].legend(loc="upper right")
    axs[-1].set_xlabel("time [s]")
    fig.suptitle(f"Position tracking (k={k:.2f})")
    fig.savefig(os.path.join(out_dir, f"k{tag}_position.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Position error (true vs nominal) over time
    fig = plt.figure()
    plt.plot(t, d["pos_err"])
    plt.xlabel("time [s]"); plt.ylabel("‖x_nom - x_true‖ [m]")
    plt.title(f"True-vs-nominal position error (k={k:.2f})"); plt.grid(True)
    fig.savefig(os.path.join(out_dir, f"k{tag}_pos_error.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Residual over time
    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axs[0].plot(t, d["residual"][:, 0], label="df [N]")
    axs[0].set_ylabel("thrust residual [N]"); axs[0].grid(True); axs[0].legend()
    for i, lbl in enumerate(["dMx", "dMy", "dMz"], start=1):
        axs[1].plot(t, d["residual"][:, i], label=f"{lbl} [N·m]")
    axs[1].set_ylabel("moment residual [N·m]"); axs[1].set_xlabel("time [s]")
    axs[1].grid(True); axs[1].legend()
    fig.suptitle(f"Learned residual (k={k:.2f})")
    fig.savefig(os.path.join(out_dir, f"k{tag}_residual.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Total control applied to the true plant
    fig, axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axs[0].plot(t, d["u_total"][:, 0]); axs[0].set_ylabel("thrust f [N]"); axs[0].grid(True)
    for i, lbl in enumerate(["Mx", "My", "Mz"], start=1):
        axs[1].plot(t, d["u_total"][:, i], label=f"{lbl}")
    axs[1].set_ylabel("moment [N·m]"); axs[1].set_xlabel("time [s]"); axs[1].grid(True); axs[1].legend()
    fig.suptitle(f"Total control on true plant (k={k:.2f})")
    fig.savefig(os.path.join(out_dir, f"k{tag}_control.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_comparison(rows, out_dir):
    """Bar chart of mean tracking error with vs without the residual, per k."""
    ks = [r["k"] for r in rows]
    with_res = [r["mean_pos_err"] for r in rows]
    without = [r["baseline_mean_pos_err"] for r in rows]

    x = np.arange(len(ks)); w = 0.38
    fig = plt.figure(figsize=(8, 4.5))
    plt.bar(x - w / 2, without, w, label="base controller only")
    plt.bar(x + w / 2, with_res, w, label="base + residual")
    plt.xticks(x, [f"{k:.2f}" for k in ks])
    plt.xlabel("disturbance k"); plt.ylabel("mean position error [m]")
    plt.title("Tracking error: baseline vs residual")
    plt.legend(); plt.grid(True, axis="y")
    fig.savefig(os.path.join(out_dir, "baseline_vs_residual.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- driver
def evaluate(checkpoint, cfg, ks, eval_tag=None, device="cpu"):
    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint))
    ckpt_stem = os.path.splitext(os.path.basename(checkpoint))[0]
    eval_tag = eval_tag or f"eval_{ckpt_stem}"
    out_dir = os.path.join(ckpt_dir, "eval", eval_tag)
    os.makedirs(out_dir, exist_ok=True)

    agent = load_agent(checkpoint, cfg, device=device)

    rows = []
    for k in ks:
        d = rollout(agent, cfg, k, use_residual=True)
        base = rollout(agent, cfg, k, use_residual=False)

        plot_rollout(d, out_dir)
        np.savez(os.path.join(out_dir, f"traj_k{_ktag(k)}.npz"), **{
            kk: vv for kk, vv in d.items() if isinstance(vv, np.ndarray)
        })

        row = {
            "k": k,
            "steps": d["steps"],
            "terminated": d["terminated"],
            "mean_pos_err": float(np.mean(d["pos_err"])),
            "final_pos_err": float(d["pos_err"][-1]),
            "max_pos_err": float(np.max(d["pos_err"])),
            "mean_residual_thrust": float(np.mean(np.abs(d["residual"][:, 0]))),
            "mean_residual_moment": float(np.mean(np.abs(d["residual"][:, 1:4]))),
            "baseline_mean_pos_err": float(np.mean(base["pos_err"])),
            "baseline_terminated": base["terminated"],
            "baseline_steps": base["steps"],
        }
        rows.append(row)
        print(f"k={k:.2f}  mean_err {row['mean_pos_err']:.4f} "
              f"(base {row['baseline_mean_pos_err']:.4f})  steps {row['steps']}"
              f"{'  [TERMINATED]' if row['terminated'] else ''}")

    plot_baseline_comparison(rows, out_dir)

    aggregate = {
        "checkpoint": os.path.abspath(checkpoint),
        "ks": list(ks),
        "mean_pos_err_over_k": float(np.mean([r["mean_pos_err"] for r in rows])),
        "mean_baseline_pos_err_over_k": float(np.mean([r["baseline_mean_pos_err"] for r in rows])),
        "per_k": rows,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(aggregate, f, indent=2)

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\neval artifacts -> {out_dir}")
    return out_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to a .pt checkpoint")
    p.add_argument("--ks", type=float, nargs="+",
                   default=[0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5],
                   help="disturbance factors to evaluate")
    p.add_argument("--eval_tag", default=None,
                   help="subfolder name under eval/ (default: eval_<ckpt stem>)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    # Load the config that was dumped next to the checkpoint, if present, so the
    # network shapes match. Fall back to defaults otherwise.
    cfg = Config()
    cfg_json = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "config.json")
    if os.path.exists(cfg_json):
        _apply_config_json(cfg, cfg_json)

    evaluate(args.checkpoint, cfg, args.ks, eval_tag=args.eval_tag, device=args.device)


def _apply_config_json(cfg, path):
    """Overlay saved config values onto a fresh Config (best-effort, shallow)."""
    with open(path) as f:
        saved = json.load(f)
    for section in ("env", "sac", "net"):
        if section in saved and hasattr(cfg, section):
            obj = getattr(cfg, section)
            for k, v in saved[section].items():
                if hasattr(obj, k):
                    # tuples get JSON-dumped as lists; restore tuple where the default is one
                    cur = getattr(obj, k)
                    if isinstance(cur, tuple) and isinstance(v, list):
                        v = tuple(v)
                    setattr(obj, k, v)
    for k in ("seed", "runs_root", "run_name", "trial"):
        if k in saved:
            setattr(cfg, k, saved[k])


if __name__ == "__main__":
    main()