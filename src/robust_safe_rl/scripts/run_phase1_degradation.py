"""Phase 1: residual degradation vs severity, per disturbance type.

Central question: how do base-only and base+residual perform under each
uncertainty, as a function of severity?

Four conditions, each comparing 2 controllers (base, base+res):
  * nominal       : no disturbance (k=1) -- confirms residual does not hurt.
  * massmoi       : mass/MOI multiplier k, swept over [k_min, k_max].
  * thrust_factor : motor thrust-coeff scale (mixer), swept; mass/MOI OFF (k=1).
  * arm_length    : rotor arm scale (mixer), swept; mass/MOI OFF (k=1).

The OOD conditions run at k=1 to isolate each uncertainty. The disturbance is
CONSTANT from t=0 (this is a steady-state characterization, not an onset study).

Fairness: for each condition the same set of severity values is applied to BOTH
controllers (pre-sampled, replayed), so any difference is the controller.

Outputs (per condition) under <runs_root>/<run_name>/trial_N/:
  sweep_<cond>.json      per-severity RMSE for base and base+res
  curve_<cond>.png       tracking RMSE vs severity, base vs base+res
  repr_<cond>.png        tracking error vs time at the representative severity
  summary.json           aggregate RMSE-over-sweep per condition and controller
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
from robust_safe_rl.eval_integrated.multi_env import MultiDisturbanceEvalEnv, DisturbanceScenario
from robust_safe_rl.eval_integrated.controllers import ResidualPolicy


# representative severities for the time-series plots
REPR = {"nominal": None, "massmoi": 1.3, "thrust_factor": 0.8, "arm_length": 0.8}


def load_policy(policy_ckpt, env, device):
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(policy_ckpt)), "config.json")
    with open(cfg_path) as f:
        pcfg = json.load(f)
    hidden = pcfg["net"]["hidden"]
    action_scale = pcfg["env"]["action_scale"]
    obs_mode = pcfg["env"].get("obs_mode", "history")
    return ResidualPolicy(policy_ckpt, env.obs_dim, 4, hidden, action_scale, device=device), obs_mode


def make_scenario(cond, severity, k_for_cond):
    """Build a constant-from-t=0 scenario for a condition + severity."""
    if cond == "nominal":
        return DisturbanceScenario(k=1.0, dist_type="none", severity=0.0, onset_step=0)
    if cond == "massmoi":
        return DisturbanceScenario(k=severity, dist_type="none", severity=0.0, onset_step=0)
    # thrust_factor / arm_length at k=1
    return DisturbanceScenario(k=1.0, dist_type=cond, severity=severity, onset_step=0)


def run_episode(env, scenario, policy, use_residual):
    obs = env.reset(scenario)
    errs = []
    ts = []
    for _ in range(env.cfg.episode_steps):
        residual = policy.residual(obs) if use_residual else np.zeros(4)
        info = env.step(residual)
        obs = info["obs"]
        errs.append(info["pos_err_des"])
        ts.append(info["step"] * env.dt)
        if info["terminated"] or info["truncated"]:
            break
    errs = np.asarray(errs)
    rmse = float(np.sqrt(np.mean(errs ** 2)))
    return rmse, np.asarray(ts), errs, len(errs) < env.cfg.episode_steps


def severity_grid(cond, args):
    if cond == "nominal":
        return [0.0]
    if cond == "massmoi":
        return list(np.round(np.linspace(args.k_min, args.k_max, args.sweep_points), 3))
    # params
    return list(np.round(np.linspace(args.param_min, args.param_max, args.sweep_points), 3))


def main(args):
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    cfg = Config()
    run_dir, trial = resolve_run_dir_args(args.runs_root, args.run_name, args.trial)
    print(f"run: {args.run_name}  trial: {trial}\nartifacts -> {run_dir}")

    env = MultiDisturbanceEvalEnv(cfg.env, obs_mode="history")
    policy, obs_mode = load_policy(args.policy, env, device)
    if obs_mode != "history":
        env = MultiDisturbanceEvalEnv(cfg.env, obs_mode=obs_mode)  # rebuild to match policy

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({"policy": os.path.abspath(args.policy), "obs_mode": obs_mode,
                   "conditions": args.conditions, "sweep_points": args.sweep_points,
                   "repeats": args.repeats, "run_name": args.run_name, "trial": trial}, f, indent=2)

    rng = np.random.default_rng(args.seed)
    summary = {}

    for cond in args.conditions:
        grid = severity_grid(cond, args)
        base_rmse, res_rmse = [], []
        for sev in grid:
            # repeats for averaging (nominal/param are deterministic given sev, but
            # keep >=1; massmoi severity IS the k so repeats add nothing -- use 1)
            b_list, r_list = [], []
            for rep in range(args.repeats):
                sc = make_scenario(cond, sev, k_for_cond=sev)
                b, _, _, _ = run_episode(env, sc, policy, use_residual=False)
                r, _, _, _ = run_episode(env, sc, policy, use_residual=True)
                b_list.append(b); r_list.append(r)
            base_rmse.append(float(np.mean(b_list)))
            res_rmse.append(float(np.mean(r_list)))

        # save sweep
        with open(os.path.join(run_dir, f"sweep_{cond}.json"), "w") as f:
            json.dump({"severity": [float(s) for s in grid],
                       "base_rmse": base_rmse, "res_rmse": res_rmse}, f, indent=2)

        # curve plot (skip for nominal single point)
        if cond != "nominal":
            fig = plt.figure(figsize=(7, 5))
            plt.plot(grid, base_rmse, "o-", label="base", color="gray")
            plt.plot(grid, res_rmse, "s-", label="base+res", color="tab:blue")
            xlabel = "k (mass/MOI)" if cond == "massmoi" else f"{cond} multiplier"
            plt.xlabel(xlabel); plt.ylabel("tracking RMSE [m]")
            plt.title(f"Phase 1: residual vs {cond} (k=1 isolated)" if cond != "massmoi"
                      else "Phase 1: residual vs mass/MOI")
            plt.legend(); plt.grid(True, alpha=0.3)
            fig.savefig(os.path.join(run_dir, f"curve_{cond}.png"), dpi=140, bbox_inches="tight")
            plt.close(fig)

        # representative time-series plot
        repr_sev = REPR[cond]
        if repr_sev is not None or cond == "nominal":
            sc = make_scenario(cond, repr_sev if repr_sev is not None else 0.0, repr_sev)
            _, tb, eb, _ = run_episode(env, sc, policy, use_residual=False)
            _, tr, er, _ = run_episode(env, sc, policy, use_residual=True)
            fig = plt.figure(figsize=(8, 4.5))
            plt.plot(tb, eb, label="base", color="gray")
            plt.plot(tr, er, label="base+res", color="tab:blue")
            plt.xlabel("time [s]"); plt.ylabel("position error [m]")
            ttl = "nominal" if cond == "nominal" else f"{cond} = {repr_sev}"
            plt.title(f"Phase 1 representative: {ttl}")
            plt.legend(); plt.grid(True, alpha=0.3)
            fig.savefig(os.path.join(run_dir, f"repr_{cond}.png"), dpi=140, bbox_inches="tight")
            plt.close(fig)

        # aggregate RMSE over the sweep
        summary[cond] = {
            "rmse_over_sweep_base": float(np.mean(base_rmse)),
            "rmse_over_sweep_res": float(np.mean(res_rmse)),
            "residual_helps": float(np.mean(base_rmse) - np.mean(res_rmse)),
            "n_severities": len(grid),
        }
        print(f"  {cond:14s}: base RMSE {summary[cond]['rmse_over_sweep_base']:.4f}  "
              f"res RMSE {summary[cond]['rmse_over_sweep_res']:.4f}  "
              f"(residual {'helps' if summary[cond]['residual_helps']>0 else 'HURTS'} "
              f"by {abs(summary[cond]['residual_helps']):.4f})")

    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nartifacts -> {run_dir}")


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--conditions", nargs="+",
                   default=["nominal", "massmoi", "thrust_factor", "arm_length"])
    p.add_argument("--sweep_points", type=int, default=11)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--k_min", type=float, default=0.5)
    p.add_argument("--k_max", type=float, default=1.5)
    p.add_argument("--param_min", type=float, default=0.7)
    p.add_argument("--param_max", type=float, default=1.3)
    p.add_argument("--runs_root", default="runs_phase1")
    p.add_argument("--run_name", default="phase1")
    p.add_argument("--trial", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())