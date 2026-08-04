"""Run the integrated detector+residual evaluation.

For each sampled scenario (k, force vector, force-onset time), runs FOUR
controller cases on the identical scenario:

  1. base          : base controller only
  2. base+res      : base + residual policy (always on)
  3. base+res+ae   : base + residual, gated by the autoencoder OOD detector
  4. base+res+dyn  : base + residual, gated by the residual-dynamics OOD detector

The detector-gated cases latch to base-only after `confirm_window` consecutive
OOD detections. Everything needed for a cumulative comparison report is logged
per step and per episode, with force-onset and detection times recorded so plots
can mark them.

Output layout (mirrors the other run folders):
  <runs_root>/<run_name>/trial_<N>/
    config.json
    episodes.json         per-episode summary for all 4 cases (k, |F|, onset,
                          detection step, mean/var error, terminated, ...)
    per_step/ep<E>.npz    per-step traces for every case (for detailed plots)
    plots/ep<E>_error.png tracking error vs time, force-onset & detection marked
    plots/ep<E>_detect.png detector scores vs time with thresholds
    summary.json          aggregate metrics across episodes, per case
"""

import argparse
import json
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robust_safe_rl.rl.config import Config
from robust_safe_rl.rl.run_utils import resolve_run_dir_args
from robust_safe_rl.eval_integrated.env import IntegratedEvalEnv, sample_scenario
from robust_safe_rl.eval_integrated.controllers import ResidualPolicy, LatchedDetectorGate
from robust_safe_rl.eval_integrated.detectors import AEDetector, DynamicsDetector


CASES = ["base", "base+res", "base+res+ae", "base+res+dyn"]
EPISODE_STEPS = 1000  # set from config in main()


def load_policy_cfg(policy_ckpt):
    """Read the residual policy's config.json (next to its checkpoint)."""
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(policy_ckpt)), "config.json")
    with open(cfg_path) as f:
        return json.load(f)


def run_case(case, env, scenario, policy, detectors, confirm_window):
    """Run one controller case on one scenario. Returns per-step trace dict."""
    obs = env.reset(scenario)
    gate = LatchedDetectorGate(confirm_window)
    det = None
    if case == "base+res+ae":
        det = detectors["ae"]; det.reset()
    elif case == "base+res+dyn":
        det = detectors["dyn"]; det.reset()

    T = env.cfg.episode_steps
    trace = {k: [] for k in ("t", "pos_err_des", "pos_err_nom", "force_on",
                             "det_score", "ood_flag", "fired", "res_norm")}

    for step in range(T):
        # decide residual
        if case == "base":
            residual = np.zeros(4)
        else:
            residual = policy.residual(obs)
            if gate.fired:                       # latched -> fall back to base
                residual = np.zeros(4)

        info = env.step(residual)
        obs = info["obs"]

        # detector scoring (uses the step's twin states / a_res)
        score = np.nan
        ood_now = False
        if det is not None:
            if det.name == "autoencoder":
                score = det.score(info["nom_next"], info["true_next"], info["u_total"])
            else:  # dynamics
                score = det.score(info["st_for_accel"], info["u_total"], info["a_res"])
            if det.warmed_up():
                ood_now = det.is_ood(score)
        fired = gate.update(ood_now, step) if det is not None else False

        trace["t"].append(info["step"] * env.dt)
        trace["pos_err_des"].append(info["pos_err_des"])
        trace["pos_err_nom"].append(info["pos_err_nom"])
        trace["force_on"].append(info["force_on"])
        trace["det_score"].append(score)
        trace["ood_flag"].append(bool(ood_now))
        trace["fired"].append(bool(fired))
        trace["res_norm"].append(float(np.linalg.norm(info["u_res"])))

        if info["terminated"] or info["truncated"]:
            break

    for k in trace:
        trace[k] = np.asarray(trace[k])
    trace["fired_step"] = gate.fired_step
    trace["threshold"] = det.threshold if det is not None else None
    return trace


def episode_metrics(case, trace, scenario, onset_step, dt):
    err = trace["pos_err_des"]
    # split error before/after force onset for a fair comparison
    onset_t = onset_step * dt
    pre = err[trace["t"] < onset_t]
    post = err[trace["t"] >= onset_t]
    return {
        "case": case,
        "k": scenario.k,
        "force_mag": scenario.force_mag,
        "onset_step": onset_step,
        "onset_time": onset_t,
        "steps": int(len(err)),
        "err_mean": float(err.mean()),
        "err_var": float(err.var()),
        "err_mean_pre": float(pre.mean()) if pre.size else None,
        "err_mean_post": float(post.mean()) if post.size else None,
        "err_max": float(err.max()),
        "detected": trace["fired_step"] is not None,
        "detect_step": trace["fired_step"],
        "detect_delay_steps": (trace["fired_step"] - onset_step
                               if trace["fired_step"] is not None else None),
        "terminated": bool(len(err) < dt_episode_steps(trace)),
    }


def dt_episode_steps(trace):
    # a full-length episode ran all steps; a terminated one stopped early.
    # We infer the intended length from the trace's own recorded t vs the
    # configured episode length passed via the module-level EPISODE_STEPS.
    return EPISODE_STEPS


def plot_episode(ep_idx, traces, scenario, onset_step, dt, out_dir):
    onset_t = onset_step * dt
    # --- tracking error, all cases ---
    fig = plt.figure(figsize=(9, 5))
    for case in CASES:
        tr = traces[case]
        plt.plot(tr["t"], tr["pos_err_des"], label=case, lw=1.3)
    plt.axvline(onset_t, color="k", ls="--", lw=1.2, label=f"force onset ({onset_t:.2f}s)")
    # mark detection times for the gated cases
    for case, c in [("base+res+ae", "tab:green"), ("base+res+dyn", "tab:red")]:
        fs = traces[case]["fired_step"]
        if fs is not None:
            plt.axvline(fs * dt, color=c, ls=":", lw=1.4,
                        label=f"{case} detect ({fs*dt:.2f}s)")
    plt.xlabel("time [s]"); plt.ylabel("position error vs desired [m]")
    plt.title(f"Episode {ep_idx}: k={scenario.k:.2f}, |F|={scenario.force_mag:.2f} N")
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, f"ep{ep_idx}_error.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- detector scores ---
    fig = plt.figure(figsize=(9, 5))
    for case, c in [("base+res+ae", "tab:green"), ("base+res+dyn", "tab:red")]:
        tr = traces[case]
        plt.plot(tr["t"], tr["det_score"], label=f"{case} score", color=c, lw=1.1)
        if tr["threshold"] is not None:
            plt.axhline(tr["threshold"], color=c, ls="--", lw=1.0, alpha=0.7,
                        label=f"{case} threshold")
    plt.axvline(onset_t, color="k", ls="--", lw=1.2, label="force onset")
    plt.xlabel("time [s]"); plt.ylabel("detector score")
    plt.title(f"Episode {ep_idx}: detector scores (log scale)")
    plt.yscale("log"); plt.legend(fontsize=8); plt.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, f"ep{ep_idx}_detect.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(args):
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    cfg = Config()
    global EPISODE_STEPS
    EPISODE_STEPS = cfg.env.episode_steps

    run_dir, trial = resolve_run_dir_args(args.runs_root, args.run_name, args.trial)
    plots_dir = os.path.join(run_dir, "plots")
    steps_dir = os.path.join(run_dir, "per_step")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(steps_dir, exist_ok=True)
    print(f"run: {args.run_name}  trial: {trial}\nartifacts -> {run_dir}")

    # --- load residual policy (its obs_mode drives the env observation) ---
    pcfg = load_policy_cfg(args.policy)
    # Older policies (trained before obs_mode existed) implicitly used the
    # "history" observation; default to it when the field is absent.
    obs_mode = pcfg["env"].get("obs_mode", "history")
    hidden = pcfg["net"]["hidden"]
    action_scale = pcfg["env"]["action_scale"]
    env = IntegratedEvalEnv(cfg.env, obs_mode=obs_mode, device=device)
    policy = ResidualPolicy(args.policy, env.obs_dim, 4, hidden, action_scale, device=device)

    # --- load detectors ---
    detectors = {
        "ae": AEDetector(args.ae_ckpt, device=device),
        "dyn": DynamicsDetector(args.dyn_ckpt, args.dyn_threshold, device=device),
    }
    print(f"AE threshold {detectors['ae'].threshold:.5e} (hist {detectors['ae'].history_len})")
    print(f"DYN threshold {detectors['dyn'].threshold:.5e} (hist {detectors['dyn'].history_len})")

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"policy": os.path.abspath(args.policy), "obs_mode": obs_mode,
                   "ae_ckpt": os.path.abspath(args.ae_ckpt),
                   "dyn_ckpt": os.path.abspath(args.dyn_ckpt),
                   "dyn_threshold": args.dyn_threshold,
                   "ae_threshold": detectors["ae"].threshold,
                   "confirm_window": args.confirm_window,
                   "episodes": args.episodes, "force_mode": args.force_mode,
                   "onset_range_s": [args.onset_min, args.onset_max],
                   "run_name": args.run_name, "trial": trial}, f, indent=2)

    rng = np.random.default_rng(args.seed)
    all_metrics = []

    for ep in range(args.episodes):
        scenario = sample_scenario(cfg.env, rng, force_mode=args.force_mode,
                                   onset_range_s=(args.onset_min, args.onset_max))
        traces = {}
        for case in CASES:
            traces[case] = run_case(case, env, scenario, policy, detectors,
                                    args.confirm_window)
            m = episode_metrics(case, traces[case], scenario, scenario.onset_step, env.dt)
            all_metrics.append({"episode": ep, **m})

        # save per-step traces + plots
        np.savez(os.path.join(steps_dir, f"ep{ep}.npz"),
                 **{f"{case}__{k}": v for case in CASES for k, v in traces[case].items()
                    if isinstance(v, np.ndarray)},
                 k=scenario.k, force_mag=scenario.force_mag,
                 force_vec=scenario.force_vec, onset_step=scenario.onset_step)
        plot_episode(ep, traces, scenario, scenario.onset_step, env.dt, plots_dir)

        if (ep + 1) % 5 == 0:
            print(f"  episode {ep+1}/{args.episodes}  k={scenario.k:.2f} "
                  f"|F|={scenario.force_mag:.2f} onset={scenario.onset_step*env.dt:.2f}s")

    with open(os.path.join(run_dir, "episodes.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    # --- aggregate summary per case ---
    summary = {}
    for case in CASES:
        rows = [m for m in all_metrics if m["case"] == case]
        errs = np.array([r["err_mean"] for r in rows])
        errs_post = np.array([r["err_mean_post"] for r in rows if r["err_mean_post"] is not None])
        det_rows = [r for r in rows if r["detected"]]
        delays = np.array([r["detect_delay_steps"] for r in det_rows
                           if r["detect_delay_steps"] is not None])
        summary[case] = {
            "err_mean_overall": float(errs.mean()),
            "err_mean_post_onset": float(errs_post.mean()) if errs_post.size else None,
            "err_var_overall": float(errs.var()),
            "n_terminated": int(sum(r["terminated"] for r in rows)),
            "n_detected": len(det_rows),
            "detect_rate": len(det_rows) / len(rows) if rows else None,
            "mean_detect_delay_steps": float(delays.mean()) if delays.size else None,
        }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== summary (mean position error vs desired) ===")
    for case in CASES:
        s = summary[case]
        det = f"  detect {s['detect_rate']:.2f} delay {s['mean_detect_delay_steps']}" \
            if s["detect_rate"] is not None and "res" in case and "+" in case else ""
        print(f"  {case:14s}: overall {s['err_mean_overall']:.4f}  "
              f"post-onset {s['err_mean_post_onset']}{det}")
    print(f"\nartifacts -> {run_dir}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="trained SAC residual policy checkpoint")
    p.add_argument("--ae_ckpt", required=True, help="trained autoencoder checkpoint")
    p.add_argument("--dyn_ckpt", required=True, help="trained residual-dynamics checkpoint")
    p.add_argument("--dyn_threshold", type=float, required=True,
                   help="residual-dynamics OOD threshold (e.g. its saved ID p99)")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--confirm_window", type=int, default=10)
    p.add_argument("--force_mode", type=int, default=3, choices=[1, 2, 3])
    p.add_argument("--onset_min", type=float, default=3.0)
    p.add_argument("--onset_max", type=float, default=5.0)
    p.add_argument("--runs_root", default="runs_detector_residual_mid_eval")
    p.add_argument("--run_name", default="integrated")
    p.add_argument("--trial", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())