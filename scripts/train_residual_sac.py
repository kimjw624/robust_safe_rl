"""Train the residual SAC policy with optional fixed-step curriculum.

This script is intentionally report-oriented: every run gets a zero-padded run
folder, a frozen resolved config/curriculum, scalar CSV logs, deterministic
seeded evaluations, and best/last/interrupted checkpoints.
"""

import argparse
import csv
import json
import os
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import torch

from robust_safe_rl.rl.config import Config
from robust_safe_rl.rl.curriculum import env_config_for_stage, load_curriculum
from robust_safe_rl.rl.residual_env import ResidualTwinEnv
from robust_safe_rl.rl.sac import SAC, ReplayBuffer
from robust_safe_rl.rl.run_utils import resolve_run_dir, checkpoint_name, dump_config


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _append_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _checkpoint_payload(agent, step, episode, scheduled_stage_idx, episode_stage_idx,
                        best_tracking_score, best_true_des_rmse, curriculum_rng):
    # Keep agent keys at the root so existing evaluation loaders that expect
    # checkpoint["actor"] continue to work.
    payload = agent.state_dict()
    payload["_training"] = {
        "step": int(step),
        "episode": int(episode),
        "scheduled_stage_idx": int(scheduled_stage_idx),
        "episode_stage_idx": int(episode_stage_idx),
        "best_tracking_score": float(best_tracking_score),
        "best_true_des_pos_rmse": float(best_true_des_rmse),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "curriculum_rng_state": curriculum_rng.bit_generator.state,
    }
    if torch.cuda.is_available():
        payload["_training"]["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def evaluate_env_config(agent, env_cfg, n_episodes=5, seed=12345):
    """Deterministic fixed-seed evaluation for one disturbance distribution.

    ``tracking_score`` is the fixed-horizon average of the state/model-tracking
    reward. If an episode terminates early, the unvisited remainder of the
    nominal horizon contributes zero, so early failure is automatically penalized.
    """
    episode_metrics = []
    env = ResidualTwinEnv(env_cfg, seed=seed)
    horizon = int(env_cfg.episode_steps)

    for _ in range(int(n_episodes)):
        obs = env.reset()
        ep_return = 0.0
        state_reward_sum = 0.0
        n = 0
        true_des_sse = 0.0
        nom_des_sse = 0.0
        true_nom_sse = 0.0
        terminated = False
        sat_steps = 0
        residual_norm_sum = 0.0

        while True:
            action = agent.act(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_return += float(reward)
            state_reward_sum += float(info.get("reward_state", 0.0))
            sat_steps += int(bool(info.get("actuator_saturated", False)))
            residual_norm_sum += float(np.linalg.norm(np.asarray(info.get("residual", 0.0))))

            desired = env.traj.desired(env.t)
            st = env.dyn_true.state()
            sn = env.dyn_nom.state()
            e_true_des = st["x"] - desired["x"]
            e_nom_des = sn["x"] - desired["x"]
            e_true_nom = st["x"] - sn["x"]
            true_des_sse += float(np.dot(e_true_des, e_true_des))
            nom_des_sse += float(np.dot(e_nom_des, e_nom_des))
            true_nom_sse += float(np.dot(e_true_nom, e_true_nom))
            n += 1

            if term or trunc:
                terminated = bool(term)
                break

        denom = max(n, 1)
        episode_metrics.append({
            "return": float(ep_return),
            "tracking_score": float(state_reward_sum / max(horizon, 1)),
            "true_des_pos_rmse": float(np.sqrt(true_des_sse / denom)),
            "nom_des_pos_rmse": float(np.sqrt(nom_des_sse / denom)),
            "true_nom_pos_rmse": float(np.sqrt(true_nom_sse / denom)),
            "terminated": float(terminated),
            "episode_length": int(n),
            "actuator_sat_fraction": float(sat_steps / denom),
            "mean_residual_norm": float(residual_norm_sum / denom),
        })

    keys = (
        "return", "tracking_score", "true_des_pos_rmse", "nom_des_pos_rmse",
        "true_nom_pos_rmse", "terminated", "episode_length",
        "actuator_sat_fraction", "mean_residual_norm",
    )
    out = {k: float(np.mean([m[k] for m in episode_metrics])) for k in keys}
    out["termination_rate"] = out.pop("terminated")
    out["episodes"] = int(n_episodes)
    return out


def evaluate_curriculum(agent, stage_env_cfgs, stage_names, seen_through,
                        episodes_per_stage, seed=12345):
    """Evaluate every curriculum stage seen so far using fixed random cases."""
    by_stage = {}
    for i in range(int(seen_through) + 1):
        by_stage[stage_names[i]] = evaluate_env_config(
            agent,
            stage_env_cfgs[i],
            n_episodes=episodes_per_stage,
            seed=seed + 1000 * i,
        )

    metric_names = (
        "return", "tracking_score", "true_des_pos_rmse", "nom_des_pos_rmse",
        "true_nom_pos_rmse", "termination_rate", "episode_length",
        "actuator_sat_fraction", "mean_residual_norm",
    )
    aggregate = {
        key: float(np.mean([m[key] for m in by_stage.values()]))
        for key in metric_names
    }
    aggregate["stages_evaluated"] = len(by_stage)
    return aggregate, by_stage


def main(cfg=None):
    cfg = cfg or Config()
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    cfg.device = device
    cfg.env.device = device
    set_seed(cfg.seed)

    curriculum = None
    stage_env_cfgs = None
    stage_envs = None
    stage_names = None
    total_timesteps = cfg.sac.total_timesteps

    if cfg.use_curriculum:
        curriculum = load_curriculum(cfg.curriculum_path)
        stage_env_cfgs = [env_config_for_stage(cfg.env, s) for s in curriculum.stages]
        stage_names = [s.name for s in curriculum.stages]
        stage_envs = [
            ResidualTwinEnv(stage_cfg, seed=cfg.seed + 1000 * i)
            for i, stage_cfg in enumerate(stage_env_cfgs)
        ]
        total_timesteps = curriculum.total_timesteps
        obs_dims = {env.obs_dim for env in stage_envs}
        action_dims = {env.action_dim for env in stage_envs}
        if len(obs_dims) != 1 or len(action_dims) != 1:
            raise ValueError("all curriculum stages must use the same observation/action dimensions")
    else:
        stage_env_cfgs = [cfg.env]
        stage_names = ["fixed_distribution"]
        stage_envs = [ResidualTwinEnv(cfg.env, seed=cfg.seed)]

    run_dir, trial = resolve_run_dir(cfg, create=True)
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    logs_dir = os.path.join(run_dir, "logs")
    eval_dir = os.path.join(run_dir, "evaluation")
    dump_config(cfg, run_dir)
    if curriculum is not None:
        shutil.copyfile(cfg.curriculum_path, os.path.join(run_dir, "curriculum.toml"))

    obs_dim = stage_envs[0].obs_dim
    action_dim = stage_envs[0].action_dim
    agent = SAC(obs_dim, action_dim, cfg.sac, cfg.net, device=device)
    buffer = ReplayBuffer(cfg.sac.buffer_size, obs_dim, action_dim, torch.device(device))
    curriculum_rng = np.random.default_rng(cfg.seed + 424242)

    run_info = {
        "run_name": cfg.run_name,
        "trial": trial,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "residual_interface": str(cfg.env.residual_interface),
        "wrench_thrust_filter_beta": float(cfg.env.wrench_thrust_filter_beta),
        "force_vector_limit_N": float(cfg.env.force_vector_limit_N),
        "total_timesteps": int(total_timesteps),
        "status": "running",
        "best_metric": "mean tracking_score; true->desired position RMSE tie-break",
    }
    _write_json(os.path.join(run_dir, "run_info.json"), run_info)

    print(f"run: {cfg.run_name}  trial: {trial:03d}")
    print(f"artifacts -> {run_dir}")
    if cfg.env.residual_interface == "force_vector":
        print(
            f"residual interface: force_vector  action_dim={action_dim}  "
            f"delta_A limit=+/-{cfg.env.force_vector_limit_N:g} N/axis  "
            f"filter_beta={cfg.env.force_vector_filter_beta:g}  obs_dim={obs_dim}"
        )
    else:
        print(
            f"residual interface: wrench  action_dim={action_dim}  "
            f"thrust_filter_beta={cfg.env.wrench_thrust_filter_beta:g}  obs_dim={obs_dim}"
        )
    if curriculum is not None:
        print(f"curriculum: {cfg.curriculum_path}")
        print(f"curriculum total steps: {total_timesteps}")
        print(f"rehearsal probability: {curriculum.rehearsal_probability:.2f}")
    else:
        print(f"training disturbances: {', '.join(cfg.env.disturbances)}")

    scheduled_stage_idx = 0
    episode_stage_idx = 0
    env = stage_envs[0]
    obs = env.reset()
    ep_count = 0
    t0 = time.time()
    last_announced_stage = -1
    stats = None
    best_tracking_score = -np.inf
    best_true_des_rmse = np.inf
    last_step = 0

    # Per-episode accumulators.
    ep_return = 0.0
    ep_state_reward_sum = 0.0
    ep_len = 0
    ep_true_des_sse = 0.0
    ep_nom_des_sse = 0.0
    ep_true_nom_sse = 0.0
    ep_sat_steps = 0
    ep_residual_norm_sum = 0.0

    try:
        for step in range(1, total_timesteps + 1):
            last_step = step
            if curriculum is not None:
                scheduled_stage_idx = curriculum.stage_index(step)
                if scheduled_stage_idx != last_announced_stage:
                    print(
                        f"\n[curriculum @ step {step}] scheduled stage "
                        f"{scheduled_stage_idx + 1}/{len(curriculum.stages)}: "
                        f"{stage_names[scheduled_stage_idx]}"
                    )
                    if ep_len > 0:
                        print("  current episode stays intact; new stage applies at next reset")
                    last_announced_stage = scheduled_stage_idx

            if step < cfg.sac.learning_starts:
                action = np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
            else:
                action = agent.act(obs, deterministic=False).astype(np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_state_reward_sum += float(info.get("reward_state", 0.0))
            ep_len += 1
            ep_sat_steps += int(bool(info.get("actuator_saturated", False)))
            ep_residual_norm_sum += float(np.linalg.norm(np.asarray(info.get("residual", 0.0))))

            desired = env.traj.desired(env.t)
            st = env.dyn_true.state()
            sn = env.dyn_nom.state()
            ep_true_des_sse += float(np.dot(st["x"] - desired["x"], st["x"] - desired["x"]))
            ep_nom_des_sse += float(np.dot(sn["x"] - desired["x"], sn["x"] - desired["x"]))
            ep_true_nom_sse += float(np.dot(st["x"] - sn["x"], st["x"] - sn["x"]))

            done_for_bootstrap = float(terminated)
            buffer.add(obs, action, reward, next_obs, done_for_bootstrap)
            obs = next_obs

            if step >= cfg.sac.learning_starts:
                stats = agent.update(buffer)
                if step % cfg.train_log_every == 0:
                    _append_csv(os.path.join(logs_dir, "updates.csv"), {
                        "step": step,
                        "scheduled_stage": stage_names[scheduled_stage_idx],
                        **stats,
                    })

            if terminated or truncated:
                ep_count += 1
                denom = max(ep_len, 1)
                _append_csv(os.path.join(logs_dir, "episodes.csv"), {
                    "step": step,
                    "episode": ep_count,
                    "train_stage": stage_names[episode_stage_idx],
                    "return": ep_return,
                    "tracking_score": ep_state_reward_sum / max(int(env.cfg.episode_steps), 1),
                    "length": ep_len,
                    "terminated": int(bool(terminated)),
                    "truncated": int(bool(truncated)),
                    "true_des_pos_rmse": np.sqrt(ep_true_des_sse / denom),
                    "nom_des_pos_rmse": np.sqrt(ep_nom_des_sse / denom),
                    "true_nom_pos_rmse": np.sqrt(ep_true_nom_sse / denom),
                    "actuator_sat_fraction": ep_sat_steps / denom,
                    "mean_residual_norm": ep_residual_norm_sum / denom,
                    "k": float(info.get("k", 1.0)),
                })

                if ep_count % 20 == 0:
                    sps = int(step / max(time.time() - t0, 1e-9))
                    print(
                        f"step {step:>8}  ep {ep_count:>5}  return {ep_return:7.2f}  "
                        f"len {ep_len:>4}  train_stage {stage_names[episode_stage_idx]}  "
                        f"k {info['k']:.2f}  alpha {agent.alpha:.3f}  {sps} sps"
                    )

                ep_return = 0.0
                ep_state_reward_sum = 0.0
                ep_len = 0
                ep_true_des_sse = 0.0
                ep_nom_des_sse = 0.0
                ep_true_nom_sse = 0.0
                ep_sat_steps = 0
                ep_residual_norm_sum = 0.0

                if curriculum is not None:
                    scheduled_for_next = curriculum.stage_index(min(step + 1, total_timesteps))
                    episode_stage_idx = curriculum.sample_episode_stage(
                        scheduled_for_next, curriculum_rng
                    )
                else:
                    episode_stage_idx = 0
                env = stage_envs[episode_stage_idx]
                obs = env.reset()

            if step % cfg.eval_every == 0 and step >= cfg.sac.learning_starts:
                if curriculum is not None:
                    seen_through = curriculum.stage_index(step)
                    eval_metrics, eval_by_stage = evaluate_curriculum(
                        agent,
                        stage_env_cfgs,
                        stage_names,
                        seen_through=seen_through,
                        episodes_per_stage=curriculum.evaluation_episodes_per_stage,
                    )
                else:
                    eval_metrics = evaluate_env_config(agent, cfg.env, n_episodes=6)
                    eval_by_stage = {stage_names[0]: eval_metrics}

                print(
                    f"  [eval @ {step}] tracking {eval_metrics['tracking_score']:.4f}  "
                    f"true->desired {eval_metrics['true_des_pos_rmse']:.4f} m  "
                    f"true->nominal {eval_metrics['true_nom_pos_rmse']:.4f} m  "
                    f"term {100.0 * eval_metrics['termination_rate']:.1f}%"
                )

                eval_row = {
                    "step": step,
                    "scheduled_stage": stage_names[scheduled_stage_idx],
                    **eval_metrics,
                }
                _append_csv(os.path.join(logs_dir, "evaluation.csv"), eval_row)
                _append_jsonl(os.path.join(logs_dir, "evaluation_by_stage.jsonl"), {
                    "step": step,
                    "scheduled_stage": stage_names[scheduled_stage_idx],
                    "aggregate": eval_metrics,
                    "by_stage": eval_by_stage,
                })

                score = float(eval_metrics["tracking_score"])
                rmse = float(eval_metrics["true_des_pos_rmse"])
                improved = (score > best_tracking_score + 1e-12) or (
                    abs(score - best_tracking_score) <= 1e-12 and rmse < best_true_des_rmse
                )
                if improved:
                    best_tracking_score = score
                    best_true_des_rmse = rmse
                    torch.save(
                        _checkpoint_payload(
                            agent, step, ep_count, scheduled_stage_idx, episode_stage_idx,
                            best_tracking_score, best_true_des_rmse, curriculum_rng,
                        ),
                        os.path.join(checkpoint_dir, checkpoint_name("best")),
                    )
                    print(f"    new best -> tracking {score:.4f}, pos RMSE {rmse:.4f} m")

            if step % cfg.checkpoint_every == 0 and step >= cfg.sac.learning_starts:
                torch.save(
                    _checkpoint_payload(
                        agent, step, ep_count, scheduled_stage_idx, episode_stage_idx,
                        best_tracking_score, best_true_des_rmse, curriculum_rng,
                    ),
                    os.path.join(checkpoint_dir, checkpoint_name("step", step)),
                )

    except KeyboardInterrupt:
        interrupted_path = os.path.join(checkpoint_dir, checkpoint_name("interrupted"))
        torch.save(
            _checkpoint_payload(
                agent, last_step, ep_count, scheduled_stage_idx, episode_stage_idx,
                best_tracking_score, best_true_des_rmse, curriculum_rng,
            ),
            interrupted_path,
        )
        run_info.update({
            "status": "interrupted",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "last_step": int(last_step),
            "episodes": int(ep_count),
            "best_tracking_score": float(best_tracking_score),
            "best_true_des_pos_rmse": float(best_true_des_rmse),
        })
        _write_json(os.path.join(run_dir, "run_info.json"), run_info)
        print(f"\ninterrupted. checkpoint -> {interrupted_path}")
        return

    last_path = os.path.join(checkpoint_dir, checkpoint_name("last"))
    torch.save(
        _checkpoint_payload(
            agent, total_timesteps, ep_count, scheduled_stage_idx, episode_stage_idx,
            best_tracking_score, best_true_des_rmse, curriculum_rng,
        ),
        last_path,
    )

    # Evaluate the best checkpoint automatically. Plot/report generation remains
    # a later layer; this gives a frozen final numeric summary immediately.
    best_path = os.path.join(checkpoint_dir, checkpoint_name("best"))
    if os.path.exists(best_path):
        best_sd = torch.load(best_path, map_location=device, weights_only=False)
        agent.load_state_dict(best_sd)
        if curriculum is not None:
            final_metrics, final_by_stage = evaluate_curriculum(
                agent,
                stage_env_cfgs,
                stage_names,
                seen_through=len(stage_names) - 1,
                episodes_per_stage=curriculum.evaluation_episodes_per_stage,
            )
        else:
            final_metrics = evaluate_env_config(agent, cfg.env, n_episodes=6)
            final_by_stage = {stage_names[0]: final_metrics}
        _write_json(os.path.join(eval_dir, "best_model_metrics.json"), {
            "aggregate": final_metrics,
            "by_stage": final_by_stage,
        })

    run_info.update({
        "status": "completed",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "last_step": int(total_timesteps),
        "episodes": int(ep_count),
        "best_tracking_score": float(best_tracking_score),
        "best_true_des_pos_rmse": float(best_true_des_rmse),
    })
    _write_json(os.path.join(run_dir, "run_info.json"), run_info)
    print(f"done. last checkpoint -> {last_path}")
    if os.path.exists(best_path):
        print(f"best checkpoint -> {best_path}")


def build_config_from_args(args):
    cfg = Config()

    if args.run_name is not None:      cfg.run_name = args.run_name
    if args.trial is not None:         cfg.trial = args.trial
    if args.seed is not None:          cfg.seed = args.seed
    if args.device is not None:        cfg.device = args.device
    if args.runs_root is not None:     cfg.runs_root = args.runs_root
    if args.curriculum is not None:
        cfg.curriculum_path = args.curriculum
        cfg.use_curriculum = True
    if args.no_curriculum:             cfg.use_curriculum = False

    if args.total_timesteps is not None:  cfg.sac.total_timesteps = args.total_timesteps
    if args.lr_actor is not None:         cfg.sac.lr_actor = args.lr_actor
    if args.lr_critic is not None:        cfg.sac.lr_critic = args.lr_critic
    if args.gamma is not None:            cfg.sac.gamma = args.gamma
    if args.batch_size is not None:       cfg.sac.batch_size = args.batch_size
    if args.target_entropy_scale is not None:
        cfg.sac.target_entropy_scale = args.target_entropy_scale

    if args.tau_pos is not None:          cfg.env.tau_pos = args.tau_pos
    if args.reward_norm is not None:      cfg.env.reward_norm = args.reward_norm
    if args.obs_mode is not None:         cfg.env.obs_mode = args.obs_mode
    if args.dyn_ckpt_path is not None:    cfg.env.dyn_ckpt_path = args.dyn_ckpt_path
    if args.ares_hist is not None:        cfg.env.ares_hist = args.ares_hist
    if args.ares_include_action:          cfg.env.ares_include_action = True
    if args.device is not None:           cfg.env.device = args.device
    if args.pid_integral_leak is not None: cfg.env.pid_integral_leak = args.pid_integral_leak
    if args.residual_interface is not None: cfg.env.residual_interface = args.residual_interface
    if args.wrench_thrust_filter_beta is not None:
        cfg.env.wrench_thrust_filter_beta = args.wrench_thrust_filter_beta
    if args.force_vector_limit_N is not None: cfg.env.force_vector_limit_N = args.force_vector_limit_N

    if args.disturbances is not None:      cfg.env.disturbances = tuple(args.disturbances)
    if args.k_min is not None:             cfg.env.k_min = args.k_min
    if args.k_max is not None:             cfg.env.k_max = args.k_max
    if args.force_max is not None:         cfg.env.external_force_max = args.force_max
    if args.motor_coeff_min is not None:   cfg.env.motor_coeff_min = args.motor_coeff_min
    if args.motor_coeff_max is not None:   cfg.env.motor_coeff_max = args.motor_coeff_max
    if args.moment_coeff_min is not None:  cfg.env.moment_coeff_min = args.moment_coeff_min
    if args.moment_coeff_max is not None:  cfg.env.moment_coeff_max = args.moment_coeff_max
    if args.arm_length_min is not None:    cfg.env.arm_length_min = args.arm_length_min
    if args.arm_length_max is not None:    cfg.env.arm_length_max = args.arm_length_max
    if args.per_motor_params:              cfg.env.per_motor_params = True

    if args.hidden is not None:            cfg.net.hidden = tuple(args.hidden)
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="Train residual SAC (all args optional).")
    p.add_argument("--run_name", default=None)
    p.add_argument("--trial", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None, help="cpu or cuda")
    p.add_argument("--runs_root", default=None)
    p.add_argument("--curriculum", default=None,
                   help="path to curriculum TOML; default comes from config.py")
    p.add_argument("--no_curriculum", action="store_true")

    p.add_argument("--total_timesteps", type=int, default=None,
                   help="single-distribution training only; curriculum uses sum of stage timesteps")
    p.add_argument("--lr_actor", type=float, default=None)
    p.add_argument("--lr_critic", type=float, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--target_entropy_scale", type=float, default=None)
    p.add_argument("--tau_pos", type=float, default=None)
    p.add_argument("--reward_norm", type=float, default=None)
    p.add_argument("--obs_mode", default=None, choices=["history", "pid", "pid_hist", "pred_ares"])
    p.add_argument("--dyn_ckpt_path", default=None)
    p.add_argument("--ares_hist", type=int, default=None)
    p.add_argument("--ares_include_action", action="store_true")
    p.add_argument("--pid_integral_leak", type=float, default=None)
    p.add_argument("--hidden", type=int, nargs="+", default=None)
    p.add_argument(
        "--residual_interface", default=None, choices=["wrench", "force_vector"],
        help="wrench: legacy [df,dM]; force_vector: 3-D delta_A before geometric attitude construction",
    )
    p.add_argument(
        "--wrench_thrust_filter_beta", type=float, default=None,
        help=(
            "direct-wrench only: first-order LPF beta on residual collective thrust; "
            "beta=1 disables filtering, beta=0.2 is the ID-validated retraining candidate"
        ),
    )
    p.add_argument(
        "--force_vector_limit_N", type=float, default=None,
        help="per-axis physical limit for force_vector residual; default 4 N",
    )

    p.add_argument("--disturbances", nargs="+", default=None,
                   choices=["none", "massmoi", "force", "motor_coeff", "moment_coeff", "arm_length"])
    p.add_argument("--k_min", type=float, default=None)
    p.add_argument("--k_max", type=float, default=None)
    p.add_argument("--force_max", type=float, default=None)
    p.add_argument("--motor_coeff_min", type=float, default=None)
    p.add_argument("--motor_coeff_max", type=float, default=None)
    p.add_argument("--moment_coeff_min", type=float, default=None)
    p.add_argument("--moment_coeff_max", type=float, default=None)
    p.add_argument("--arm_length_min", type=float, default=None)
    p.add_argument("--arm_length_max", type=float, default=None)
    p.add_argument("--per_motor_params", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(build_config_from_args(parse_args()))
