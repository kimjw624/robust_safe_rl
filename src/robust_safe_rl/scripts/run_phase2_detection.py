"""Phase 2: detection + 4-controller comparison, per disturbance type.

Central questions, per untrained OOD disturbance appearing mid-episode:
  (i)  can each detector detect it?
  (ii) how do the four controllers differ?

Four controllers on IDENTICAL scenarios (same k, severity, onset):
  base, base+res, base+res+ae, base+res+dyn.

Per disturbance type {force, thrust_factor, arm_length}:
  * mass/MOI present as the realistic backdrop (random k, replayed across cases),
  * one OOD disturbance at a FIXED representative severity,
  * switched on at a FIXED onset (default 4.0 s),
  * detector-gated cases latch to base-only after `confirm_window` consecutive
    OOD flags.

Logs per (type, controller): pre/post-onset tracking error; for detector cases
also detection rate, delay, and pre-onset false-positive rate. Plots mark the
onset and detection times.

Output under <runs_root>/<run_name>/trial_N/<dist_type>/:
  episodes.json, summary.json, plots/ep*_error.png, plots/ep*_detect.png
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
from robust_safe_rl.eval_integrated.multi_env import MultiDisturbanceEvalEnv, sample_scenario
from robust_safe_rl.eval_integrated.controllers import ResidualPolicy, LatchedDetectorGate
from robust_safe_rl.eval_integrated.detectors import AEDetector, DynamicsDetector


BASE_CASES = ["base", "base+res", "base+res+dyn"]
AE_CASE = "base+res+ae"
# NOTE: the autoencoder detector is currently DISABLED by default. It has a known
# feature-mismatch bug -- it was trained on base-only actions/state-divergence
# (no residual), but the eval feeds it residual-laden u_total and residual-driven
# state divergence, so it sees out-of-distribution inputs from step 1 and fires
# immediately every episode regardless of any real disturbance. The fix is to
# feed it base-only "shadow" signals matching its training (like the dynamics
# detector's controller-independent a_res). Until then, run without --ae_ckpt.
REPR_SEVERITY = {"force": None, "thrust_factor": 0.8, "arm_length": 0.8}  # force uses strict band


def load_policy(policy_ckpt, env, device):
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(policy_ckpt)), "config.json")
    with open(cfg_path) as f:
        pcfg = json.load(f)
    obs_mode = pcfg["env"].get("obs_mode", "history")
    return (ResidualPolicy(policy_ckpt, env.obs_dim, 4, pcfg["net"]["hidden"],
                           pcfg["env"]["action_scale"], device=device), obs_mode)


def run_case(case, env, scenario, policy, detectors, confirm_window):
    obs = env.reset(scenario)
    gate = LatchedDetectorGate(confirm_window)
    det = detectors.get("ae") if case == "base+res+ae" else (
          detectors["dyn"] if case == "base+res+dyn" else None)
    if det is not None:
        det.reset()

    tr = {k: [] for k in ("t", "err", "det_score", "ood_flag", "fired")}
    for step in range(env.cfg.episode_steps):
        if case == "base":
            residual = np.zeros(4)
        else:
            residual = policy.residual(obs)
            if gate.fired:
                residual = np.zeros(4)
        info = env.step(residual)
        obs = info["obs"]

        score, ood_now = np.nan, False
        if det is not None:
            if det.name == "autoencoder":
                score = det.score(info["nom_next"], info["true_next"], info["u_total"])
            else:
                score = det.score(info["st_for_accel"], info["u_total"], info["a_res"])
            if det.warmed_up():
                ood_now = det.is_ood(score)
        fired = gate.update(ood_now, step) if det is not None else False

        tr["t"].append(info["step"] * env.dt)
        tr["err"].append(info["pos_err_des"])
        tr["det_score"].append(score)
        tr["ood_flag"].append(bool(ood_now))
        tr["fired"].append(bool(fired))
        if info["terminated"] or info["truncated"]:
            break
    for k in tr:
        tr[k] = np.asarray(tr[k])
    tr["fired_step"] = gate.fired_step
    tr["threshold"] = det.threshold if det is not None else None
    return tr


def episode_row(ep, case, tr, scenario, dt):
    onset_t = scenario.onset_step * dt
    pre = tr["err"][tr["t"] < onset_t]
    post = tr["err"][tr["t"] >= onset_t]
    return {
        "episode": ep, "case": case, "k": scenario.k,
        "severity": scenario.severity_scalar, "onset_step": scenario.onset_step,
        "err_rmse": float(np.sqrt(np.mean(tr["err"] ** 2))),
        "err_mean_pre": float(pre.mean()) if pre.size else None,
        "err_mean_post": float(post.mean()) if post.size else None,
        "detected": tr["fired_step"] is not None,
        "detect_step": tr["fired_step"],
        "detect_delay_steps": (tr["fired_step"] - scenario.onset_step
                               if tr["fired_step"] is not None else None),
    }


def plot_episode(ep, dtype, traces, scenario, dt, out_dir, cases):
    onset_t = scenario.onset_step * dt
    fig = plt.figure(figsize=(9, 5))
    for case in cases:
        plt.plot(traces[case]["t"], traces[case]["err"], label=case, lw=1.3)
    plt.axvline(onset_t, color="k", ls="--", lw=1.2, label=f"onset ({onset_t:.1f}s)")
    for case, c in [("base+res+ae", "tab:green"), ("base+res+dyn", "tab:red")]:
        if case in traces and traces[case]["fired_step"] is not None:
            plt.axvline(traces[case]["fired_step"] * dt, color=c, ls=":", lw=1.4,
                        label=f"{case} detect")
    plt.xlabel("time [s]"); plt.ylabel("position error [m]")
    plt.title(f"{dtype} ep{ep}: k={scenario.k:.2f}, sev={scenario.severity_scalar:.2f}")
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, f"ep{ep}_error.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(args):
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    cfg = Config()
    run_dir, trial = resolve_run_dir_args(args.runs_root, args.run_name, args.trial)
    print(f"run: {args.run_name}  trial: {trial}\nartifacts -> {run_dir}")

    tmp_env = MultiDisturbanceEvalEnv(cfg.env, obs_mode="history")
    policy, obs_mode = load_policy(args.policy, tmp_env, device)
    env = MultiDisturbanceEvalEnv(cfg.env, obs_mode=obs_mode)
    policy, _ = load_policy(args.policy, env, device)

    detectors = {"dyn": DynamicsDetector(args.dyn_ckpt, args.dyn_threshold, device=device)}
    CASES = list(BASE_CASES)
    if args.ae_ckpt:
        detectors["ae"] = AEDetector(args.ae_ckpt, device=device,
                                     threshold_key=args.ae_threshold_key)
        CASES = ["base", "base+res", AE_CASE, "base+res+dyn"]
        print(f"AE threshold {detectors['ae'].threshold:.4e}  ", end="")
    else:
        print("AE detector DISABLED (no --ae_ckpt; known feature-mismatch bug). ", end="")
    print(f"DYN threshold {detectors['dyn'].threshold:.4e}")

    onset_step = int(args.onset_s / cfg.env.dt)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"policy": os.path.abspath(args.policy), "obs_mode": obs_mode,
                   "ae_ckpt": os.path.abspath(args.ae_ckpt) if args.ae_ckpt else None,
                   "dyn_ckpt": os.path.abspath(args.dyn_ckpt),
                   "dyn_threshold": args.dyn_threshold,
                   "ae_threshold": detectors["ae"].threshold if "ae" in detectors else None,
                   "onset_s": args.onset_s, "confirm_window": args.confirm_window,
                   "episodes": args.episodes, "dist_types": args.dist_types,
                   "run_name": args.run_name, "trial": trial}, f, indent=2)

    grand = {}
    for dtype in args.dist_types:
        dtype_dir = os.path.join(run_dir, dtype)
        plots_dir = os.path.join(dtype_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        rng = np.random.default_rng(args.seed)   # SAME seed per type -> same k draws
        rows = []
        for ep in range(args.episodes):
            sc = sample_scenario(cfg.env, rng, dist_type=dtype, onset_step=onset_step,
                                 force_mode=3, param_min=REPR_SEVERITY.get(dtype) or 0.8,
                                 param_max=REPR_SEVERITY.get(dtype) or 0.8)
            traces = {case: run_case(case, env, sc, policy, detectors, args.confirm_window)
                      for case in CASES}
            for case in CASES:
                rows.append(episode_row(ep, case, traces[case], sc, env.dt))
            if ep < args.plot_episodes:
                plot_episode(ep, dtype, traces, sc, env.dt, plots_dir, CASES)
        with open(os.path.join(dtype_dir, "episodes.json"), "w") as f:
            json.dump(rows, f, indent=2)

        # aggregate per case
        agg = {}
        for case in CASES:
            cr = [r for r in rows if r["case"] == case]
            post = np.array([r["err_mean_post"] for r in cr if r["err_mean_post"] is not None])
            det_rows = [r for r in cr if r["detected"]]
            delays = np.array([r["detect_delay_steps"] for r in det_rows
                               if r["detect_delay_steps"] is not None])
            # false positive = detection fired before onset
            fp = [r for r in det_rows if r["detect_step"] is not None
                  and r["detect_step"] < r["onset_step"]]
            agg[case] = {
                "rmse_mean": float(np.mean([r["err_rmse"] for r in cr])),
                "post_onset_mean": float(post.mean()) if post.size else None,
                "detect_rate": (len(det_rows) / len(cr)) if ("res" in case and "+" in case
                                and case != "base+res") else None,
                "mean_delay_steps": float(delays.mean()) if delays.size else None,
                "false_positive_rate": (len(fp) / len(cr)) if det_rows else None,
            }
        grand[dtype] = agg
        with open(os.path.join(dtype_dir, "summary.json"), "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\n[{dtype}]")
        for case in CASES:
            a = agg[case]
            extra = ""
            if a["detect_rate"] is not None:
                extra = f"  detect {a['detect_rate']:.2f} delay {a['mean_delay_steps']} fp {a['false_positive_rate']}"
            print(f"  {case:14s}: post-onset {a['post_onset_mean']}{extra}")

    with open(os.path.join(run_dir, "grand_summary.json"), "w") as f:
        json.dump(grand, f, indent=2)
    print(f"\nartifacts -> {run_dir}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--ae_ckpt", default=None,
                   help="autoencoder checkpoint. OMIT to disable the AE case "
                        "(recommended for now: the AE has a known feature-mismatch bug).")
    p.add_argument("--ae_threshold_key", default="id_max",
                   choices=["id_p95", "id_p99", "id_p995", "id_max", "id_mean_plus_3std"],
                   help="which saved AE threshold to use; id_max is most robust when "
                        "the ID error distribution is tightly peaked (the mass/MOI case)")
    p.add_argument("--dyn_ckpt", required=True)
    p.add_argument("--dyn_threshold", type=float, required=True)
    p.add_argument("--dist_types", nargs="+", default=["force", "thrust_factor", "arm_length"])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--onset_s", type=float, default=4.0)
    p.add_argument("--confirm_window", type=int, default=10)
    p.add_argument("--plot_episodes", type=int, default=3)
    p.add_argument("--runs_root", default="runs_phase2")
    p.add_argument("--run_name", default="phase2")
    p.add_argument("--trial", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())