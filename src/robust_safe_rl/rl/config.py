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
    episode_steps: int = 1000          # one 10 s period of the default figure-eight

    # --- nominal (reference) plant parameters; the base controller uses these ---
    mass_nom: float = 2.0
    J_nom: tuple = (0.022, 0.022, 0.04)
    gravity: float = 9.807

    # --- trajectory ---
    traj_radius: float = 0.79
    traj_speed: float = 0.5
    traj_z0: float = 0.0

    # --- per-episode training disturbances ---
    # Select any subset of:
    #   "massmoi"      : common scale on true mass and inertia
    #   "force"        : constant world/NED external force
    #   "motor_coeff"  : rotor thrust coefficient (motorConstant / k_f) scale
    #   "moment_coeff" : rotor drag momentConstant scale
    #   "arm_length"   : rotor position / moment-arm scale
    # Disturbances listed together are sampled independently and applied
    # simultaneously for the entire episode.  Use ("none",) for a nominal plant.
    disturbances: tuple = ("massmoi",)

    # mass/MOI: m_true = k*m_nom, J_true = k*J_nom
    k_min: float = 0.7
    k_max: float = 1.3

    # external force: each NED/world axis is sampled independently from
    # U[-external_force_max, +external_force_max] N and held for the episode.
    external_force_max: float = 3.0

    # Actuator/geometry disturbances are multiplicative scales around the x500
    # nominal values used by the controller/allocation model.
    motor_coeff_min: float = 0.7
    motor_coeff_max: float = 1.3
    moment_coeff_min: float = 0.7
    moment_coeff_max: float = 1.3
    arm_length_min: float = 0.7
    arm_length_max: float = 1.3

    # False -> one common scale for all four rotors. True -> sample one scale per
    # rotor for motor_coeff, moment_coeff, and arm_length.
    per_motor_params: bool = False

    # --- history stacking ---
    history: int = 10                  # H: number of stacked frames (used by obs_mode="history" and the tail of "pid_hist")

    # --- observation mode ---
    #   "history"  : causal [e, a, ..., e, a, e] history; H errors and H-1 past
    #                normalized SAC actions (no current action in the observation)
    #   "pid"      : PID errors on the twin discrepancy + u_base           (mode A, minimal)
    #   "pid_hist" : PID errors + u_base + a short raw-discrepancy tail    (mode B, ablation)
    obs_mode: str = "history"
    pid_integral_leak: float = 0.99    # lambda for the leaky integral: I <- lambda*I + e*dt
    pid_hist_frames: int = 3           # length of the short discrepancy tail in "pid_hist" mode

    # --- pred_ares mode (policy observes predicted residual acceleration) ---
    dyn_ckpt_path: str = ""            # path to the frozen supervised f_res checkpoint
    ares_hist: int = 1                 # number of predicted-a_res frames in the observation
    ares_include_action: bool = False # also include the residual action in each frame
    device: str = "cpu"               # device for the frozen predictor

    # --- termination (no penalty yet; just ends the episode) ---
    term_pos_error: float = 2.0        # metres
    term_tilt_deg: float = 90.0        # degrees from upright

    # --- fixed history-observation scaling ---
    # The twin discrepancy is centered physically at zero, so history mode uses
    # fixed zero-centered physical scales rather than a running mean/std. These
    # values were selected from disturbed baseline rollouts (k in [0.7, 1.3],
    # per-episode F_ext in [-3, 3] N/axis) so upper-tail errors are O(1).
    obs_pos_scale: float = 0.25        # m
    obs_vel_scale: float = 0.15        # m/s
    obs_att_scale: float = 0.25        # rad (SO(3) error-vector magnitude scale)
    obs_omega_scale: float = 0.20      # rad/s

    # --- reward weights and dimensionless exponential scales (tau) ---
    # Reward uses the same normalized twin-discrepancy groups as history mode:
    #   r_i = w_i * exp(-||e_i / obs_i_scale||^2 / tau_i^2)
    # Position is deliberately strict; angular-rate matching is more tolerant.
    w_pos: float = 1.0
    w_vel: float = 0.5
    w_att: float = 0.5
    w_omega: float = 0.2
    tau_pos: float = 0.5
    tau_vel: float = 1.0
    tau_att: float = 1.0
    tau_omega: float = 1.5
    reward_norm: float = 2.2           # sum of weights -> max state reward = 1

    # --- residual-action shaping ---
    # Penalties are applied to the normalized SAC request a in [-1, 1]^d, after
    # the state reward has been normalized to a maximum of 1.0. For the filtered
    # direct-wrench thrust experiment this deliberately remains the raw request,
    # so high-frequency policy chatter cannot hide behind the command filter.
    # Using normalized
    # actions keeps these weights independent of the physical residual units and
    # authority scale used by either interface.
    w_action_effort: float = 0.01      # -w * ||a_t||_2
    w_action_smooth: float = 0.01      # -w * ||a_t - a_{t-1}||_2

    # --- residual action interface / authority ---
    # "wrench" keeps the legacy direct residual [df, dMx, dMy, dMz].
    # "force_vector" uses a 3-D NED/world force correction delta_A inserted
    # upstream of the geometric attitude construction: A_cmd = A_base + delta_A.
    residual_interface: str = "wrench"

    # Force-vector interface: each normalized SAC action component in [-1, 1]
    # maps independently to [-force_vector_limit_N, +force_vector_limit_N] N.
    # The first A-residual experiment uses 4 N per axis.
    force_vector_limit_N: float = 4.0

    # Force-vector interface only: first-order smoothing on the normalized SAC
    # command before physical scaling. beta=1 disables filtering; beta=0.01
    # gives a ~1 s command time scale at dt=0.01 s.
    force_vector_filter_beta: float = 0.01

    # Direct-wrench interface only: first-order smoothing on the normalized
    # residual collective-thrust channel. Moments remain unfiltered. beta=1
    # exactly recovers the legacy direct-wrench interface. beta=0.2 is the
    # intervention selected by the paired ID rule-out experiments. The filter
    # state is initialized from the first requested action of each episode, so
    # there is no artificial startup step from zero.
    wrench_thrust_filter_beta: float = 1.0

    # --- legacy direct-wrench residual authority ---
    # Fraction rho of the nominal x500 wrench envelope available to the residual
    # policy. Physical residual limits are derived from mixer.py as
    # rho * [F_MAX, MX_MAX, MY_MAX, MZ_MAX], so changing this single value
    # changes the residual authority without duplicating actuator constants.
    # Named ``residual_authority`` rather than alpha to avoid confusion with
    # SAC's entropy-temperature alpha.
    residual_authority: float = 0.20


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
    target_entropy_scale: float = 0.5  # target_entropy = -scale * action_dim (-2 for 4D, -1.5 for 3D)
    huber_delta: float = 1.0
    grad_clip: float = 1.0
    total_timesteps: int = 1_000_000


@dataclass
class NetConfig:
    hidden: tuple = (256, 256)
    critic_layernorm: bool = True
    # History observations already normalize only the physical error components
    # before they reach any network; past SAC actions remain in [-1, 1].
    # Keep actor hidden-layer LayerNorm off initially so we do not add a second,
    # different normalization on top of the fixed physical input scaling.
    actor_layernorm: bool = False
    # Mean starts near zero, while the stochastic policy is intentionally broad.
    zero_init_actor_mean: bool = True
    initial_log_std: float = 0.0        # std=1 before tanh -> broad initial exploration


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    sac: SACConfig = field(default_factory=SACConfig)
    net: NetConfig = field(default_factory=NetConfig)
    seed: int = 0
    device: str = "cpu"                # "cuda" if available
    eval_every: int = 20_000
    checkpoint_every: int = 50_000
    train_log_every: int = 1_000       # optimizer diagnostics cadence

    # --- fixed-step curriculum ---
    # The human-editable TOML file controls stage durations/disturbance ranges.
    # Set use_curriculum=False (or pass --no_curriculum) for a single fixed
    # disturbance distribution using EnvConfig directly.
    use_curriculum: bool = True
    curriculum_path: str = "configs/residual_sac_curriculum.toml"

    # --- run organisation ---
    # Artifacts go to:  <runs_root>/<run_name>/trial_XXX/
    #   checkpoints:    <run_name>_trial_<N>.pt   (+ periodic _step<S>.pt)
    #   training log:   log.json
    #   config dump:    config.json
    # trial_<N> auto-increments so re-running the same run_name never overwrites.
    runs_root: str = "runs_residual"
    run_name: str = "residual_sac_curriculum"
    trial: int = None                  # None -> auto-pick next free index

