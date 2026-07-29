"""Configuration for residual-SAC training on the twin-plant DOB task.

Every value here was fixed during the planning phase. Grouped by concern:
environment / task, observation, action bounds, reward, SAC, and networks.
Change values here rather than hard-coding them elsewhere.
"""

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    # --- integration / episode ---
    dt: float = 0.01
    episode_steps: int = 1000          # one lap of the circular trajectory

    # --- nominal (reference) plant parameters; the base controller uses these ---
    mass_nom: float = 2.0
    J_nom: tuple = (0.022, 0.022, 0.04)
    gravity: float = 9.807

    # --- trajectory ---
    traj_radius: float = 0.79
    traj_speed: float = 0.5
    traj_z0: float = -1.0

    # --- disturbance: single per-episode multiplier k applied to BOTH m and J ---
    #     m_true = k * mass_nom,  J_true = k * J_nom,  k ~ U[k_min, k_max]
    k_min: float = 0.5
    k_max: float = 1.5

    # --- history stacking ---
    history: int = 10                  # H: number of stacked frames

    # --- termination (no penalty yet; just ends the episode) ---
    term_pos_error: float = 2.0        # metres
    term_tilt_deg: float = 90.0        # degrees from upright

    # --- reward weights and exponential scales (tau) ---
    w_pos: float = 1.0
    w_vel: float = 0.2
    w_att: float = 0.2
    w_omega: float = 0.2
    tau_pos: float = 0.2        # DEFAULT = 0.3
    tau_vel: float = 0.5
    tau_att: float = 0.5
    tau_omega: float = 1.0
    reward_norm: float = 1.6           # divide reward by sum of weights -> [0, 1]

    # --- residual action bounds (20% of x500 physical envelope) ---
    #     [df (N), dMx, dMy, dMz (N*m)]
    action_scale: tuple = (6.84, 0.72, 0.44, 0.055)


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005                 # target smoothing (Polyak)
    buffer_size: int = 1_000_000
    batch_size: int = 256
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    learning_starts: int = 5000        # random-action warmup
    policy_frequency: int = 2          # actor update every N critic updates
    target_frequency: int = 1          # target net update cadence
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    target_entropy_scale: float = 0.5  # target_entropy = -scale * action_dim  (-> -2 for 4D)
    huber_delta: float = 1.0
    grad_clip: float = 1.0
    total_timesteps: int = 1_000_000


@dataclass
class NetConfig:
    hidden: tuple = (512, 512)
    critic_layernorm: bool = True
    actor_layernorm: bool = False
    # zero-init the actor mean head so the residual starts at ~0 (pure base controller)
    zero_init_actor_mean: bool = True


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    net: NetConfig = field(default_factory=NetConfig)
    seed: int = 0
    device: str = "cuda"                # "cuda" if available
    eval_every: int = 20_000
    checkpoint_every: int = 50_000

    # --- run organisation ---
    # Artifacts go to:  <runs_root>/<run_name>/trial_<N>/
    #   checkpoints:    <run_name>_trial_<N>.pt   (+ periodic _step<S>.pt)
    #   training log:   log.json
    #   config dump:    config.json
    # trial_<N> auto-increments so re-running the same run_name never overwrites.
    runs_root: str = "runs_residual"
    run_name: str = "residual_sac_tau0.2"
    trial: int = None                  # None -> auto-pick next free index