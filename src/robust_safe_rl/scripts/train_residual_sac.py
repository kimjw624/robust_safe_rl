"""Train a residual SAC policy on the twin-plant DOB task.

Usage (from the package root, with robust_safe_rl importable):

    python -m robust_safe_rl.scripts.train_residual_sac

Reference: CleanRL sac_continuous_action. Single environment, off-policy.
"""

import json
import os
import time

import numpy as np
import torch

from robust_safe_rl.rl.config import Config
from robust_safe_rl.rl.residual_env import ResidualTwinEnv
from robust_safe_rl.rl.sac import SAC, ReplayBuffer
from robust_safe_rl.rl.run_utils import resolve_run_dir, checkpoint_name, dump_config


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(agent, cfg, n_episodes=5, ks=None):
    """Greedy evaluation. If ks given, evaluate those fixed disturbances."""
    env = ResidualTwinEnv(cfg.env, seed=12345)
    returns, errs = [], []
    for i in range(n_episodes):
        k = None if ks is None else ks[i % len(ks)]
        obs = env.reset(k=k)
        ep_r, last_err = 0.0, 0.0
        while True:
            a = agent.act(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            ep_r += r
            last_err = info["pos_err"]
            if term or trunc:
                break
        returns.append(ep_r)
        errs.append(last_err)
    return float(np.mean(returns)), float(np.mean(errs))


def main(cfg=None):
    cfg = cfg or Config()
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    set_seed(cfg.seed)

    run_dir, trial = resolve_run_dir(cfg, create=True)
    dump_config(cfg, run_dir)
    print(f"run: {cfg.run_name}  trial: {trial}")
    print(f"artifacts -> {run_dir}")

    env = ResidualTwinEnv(cfg.env, seed=cfg.seed)
    obs_dim, action_dim = env.obs_dim, env.action_dim

    agent = SAC(obs_dim, action_dim, cfg.sac, cfg.net, device=device)
    buffer = ReplayBuffer(cfg.sac.buffer_size, obs_dim, action_dim, torch.device(device))

    obs = env.reset()
    ep_return, ep_len, ep_count = 0.0, 0, 0
    t0 = time.time()
    log = []

    for step in range(1, cfg.sac.total_timesteps + 1):
        # Action: random during warmup, else from the policy.
        if step < cfg.sac.learning_starts:
            action = np.random.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)
        else:
            action = agent.act(obs, deterministic=False).astype(np.float32)

        next_obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        ep_len += 1

        # Bootstrap on truncation (time limit), not on true termination (divergence).
        done_for_bootstrap = float(terminated)
        buffer.add(obs, action, reward, next_obs, done_for_bootstrap)
        obs = next_obs

        if terminated or truncated:
            ep_count += 1
            obs = env.reset()
            if ep_count % 20 == 0:
                sps = int(step / (time.time() - t0))
                print(f"step {step:>8}  ep {ep_count:>5}  return {ep_return:7.2f}  "
                      f"len {ep_len:>4}  k {info['k']:.2f}  alpha {agent.alpha:.3f}  {sps} sps")
            ep_return, ep_len = 0.0, 0

        # Learn.
        if step >= cfg.sac.learning_starts:
            stats = agent.update(buffer)

        if step % cfg.eval_every == 0 and step >= cfg.sac.learning_starts:
            ev_ret, ev_err = evaluate(agent, cfg, n_episodes=6,
                                      ks=[0.5, 0.7, 0.9, 1.1, 1.3, 1.5])
            print(f"  [eval @ {step}] mean_return {ev_ret:.2f}  mean_final_pos_err {ev_err:.4f}")
            log.append({"step": step, "eval_return": ev_ret, "eval_pos_err": ev_err,
                        **{k: stats[k] for k in ("critic_loss", "actor_loss", "alpha")}})
            with open(os.path.join(run_dir, "log.json"), "w") as f:
                json.dump(log, f, indent=2)

        if step % cfg.checkpoint_every == 0 and step >= cfg.sac.learning_starts:
            path = os.path.join(run_dir, checkpoint_name(cfg.run_name, trial, step))
            torch.save(agent.state_dict(), path)

    final_path = os.path.join(run_dir, checkpoint_name(cfg.run_name, trial))
    torch.save(agent.state_dict(), final_path)
    print(f"done. final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()