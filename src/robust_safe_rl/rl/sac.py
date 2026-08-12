"""SAC core: replay buffer + agent (twin-Q, auto-alpha, delayed actor updates).

Follows the CleanRL sac_continuous_action structure: critic updated every step,
actor and target networks updated every ``policy_frequency`` steps (with the
delayed actor update compensated). Entropy temperature alpha is auto-tuned to a
target entropy of ``-target_entropy_scale * action_dim``. Critic loss is Huber;
gradients are norm-clipped.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .networks import Actor, TwinCritic


class ReplayBuffer:
    """Fixed-size FIFO buffer of flat transitions, sampled uniformly."""

    def __init__(self, capacity, obs_dim, action_dim, device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.dones[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        to = lambda a: torch.as_tensor(a[idx], device=self.device)
        return to(self.obs), to(self.actions), to(self.rewards), to(self.next_obs), to(self.dones)


class SAC:
    """Soft Actor-Critic agent."""

    def __init__(self, obs_dim, action_dim, sac_cfg, net_cfg, device="cpu"):
        self.cfg = sac_cfg
        self.device = torch.device(device)
        self.action_dim = action_dim

        hidden = tuple(net_cfg.hidden)

        self.actor = Actor(
            obs_dim, action_dim, hidden=hidden,
            log_std_min=sac_cfg.log_std_min, log_std_max=sac_cfg.log_std_max,
            layernorm=net_cfg.actor_layernorm,
            zero_init_mean=net_cfg.zero_init_actor_mean,
            initial_log_std=net_cfg.initial_log_std,
        ).to(self.device)

        self.critic = TwinCritic(obs_dim, action_dim, hidden=hidden,
                                 layernorm=net_cfg.critic_layernorm).to(self.device)
        self.critic_target = TwinCritic(obs_dim, action_dim, hidden=hidden,
                                        layernorm=net_cfg.critic_layernorm).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=sac_cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=sac_cfg.lr_critic)

        # Auto-tuned entropy temperature.
        self.target_entropy = -sac_cfg.target_entropy_scale * action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=sac_cfg.lr_alpha)

        self._updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        obs = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        if deterministic:
            # Do not sample and then discard the sample: deterministic evaluation
            # must not consume the training RNG stream.
            mean, _ = self.actor(obs)
            action = torch.tanh(mean)
        else:
            action, _, _ = self.actor.sample(obs)
        return action.squeeze(0).cpu().numpy()

    def update(self, buffer):
        cfg = self.cfg
        obs, action, reward, next_obs, done = buffer.sample(cfg.batch_size)

        # ---- critic update (every step) ----
        with torch.no_grad():
            next_a, next_logp, _ = self.actor.sample(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_a)
            min_q_t = torch.min(q1_t, q2_t) - self.log_alpha.exp() * next_logp
            target_q = reward + (1.0 - done) * cfg.gamma * min_q_t

        q1, q2 = self.critic(obs, action)
        critic_loss = (
            F.huber_loss(q1, target_q, delta=cfg.huber_delta)
            + F.huber_loss(q2, target_q, delta=cfg.huber_delta)
        )

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.grad_clip)
        self.critic_opt.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        alpha_loss = torch.tensor(0.0, device=self.device)
        logp_mean = float("nan")
        action_abs_mean = float("nan")

        # ---- delayed actor + alpha update ----
        if self._updates % cfg.policy_frequency == 0:
            # Compensate for the delay by doing policy_frequency actor steps.
            for _ in range(cfg.policy_frequency):
                a_pi, logp, _ = self.actor.sample(obs)
                logp_mean = float(logp.detach().mean().item())
                action_abs_mean = float(a_pi.detach().abs().mean().item())
                q1_pi, q2_pi = self.critic(obs, a_pi)
                min_q_pi = torch.min(q1_pi, q2_pi)
                actor_loss = (self.log_alpha.exp().detach() * logp - min_q_pi).mean()

                self.actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.grad_clip)
                self.actor_opt.step()

                # Recompute log pi after the actor update so the temperature is
                # adapted to the CURRENT policy rather than the pre-update policy.
                with torch.no_grad():
                    _, logp_alpha, _ = self.actor.sample(obs)
                alpha_loss = -(
                    self.log_alpha.exp() * (logp_alpha + self.target_entropy)
                ).mean()
                self.alpha_opt.zero_grad()
                alpha_loss.backward()
                self.alpha_opt.step()

        # ---- soft target update ----
        if self._updates % cfg.target_frequency == 0:
            with torch.no_grad():
                for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                    pt.mul_(1.0 - cfg.tau).add_(cfg.tau * p)

        self._updates += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": self.alpha,
            "alpha_loss": float(alpha_loss.item()),
            "q1_mean": float(q1.detach().mean().item()),
            "q2_mean": float(q2.detach().mean().item()),
            "target_q_mean": float(target_q.detach().mean().item()),
            "logp_mean": logp_mean,
            "policy_action_abs_mean": action_abs_mean,
        }

    def state_dict(self):
        """Full trainable SAC state (the replay buffer is saved separately if desired)."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "updates": int(self._updates),
            "target_entropy": float(self.target_entropy),
        }

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))
        if "actor_opt" in sd:
            self.actor_opt.load_state_dict(sd["actor_opt"])
        if "critic_opt" in sd:
            self.critic_opt.load_state_dict(sd["critic_opt"])
        if "alpha_opt" in sd:
            self.alpha_opt.load_state_dict(sd["alpha_opt"])
        self._updates = int(sd.get("updates", 0))
