"""Collect (state-history, residual-acceleration) data for residual dynamics learning.

Rolls out the BASE controller only (no residual policy) across episodes with a
per-episode mass/inertia multiplier k ~ U[k_min, k_max], and logs, at each step:

  input frame : [R(9), v(3), omega(3), u_total(4)]  = 19 dims  (NO position)
  target      : a_res = [v_dot_true - v_dot_nom, omega_dot_true - omega_dot_nom] = 6 dims

The target is the residual acceleration: the difference between the true
(disturbed) dynamics and the nominal-model prediction, evaluated at the SAME
state and control. Both accelerations come from Dynamics.accel_at -- the nominal
one uses nominal m,J on the true plant's state (a counterfactual "what the
nominal model would predict here"), NOT the drifted nominal twin's state. This
isolates purely the effect of the mass/MOI disturbance on the acceleration.

Position is deliberately excluded from the input: it does not enter the
rigid-body residual dynamics, and including it would make the model (and any
downstream OOD detector) spuriously sensitive to unfamiliar positions.

Output: an .npz with arrays
  X       (N, H_dyn, 19)   history windows of input frames (zero-padded at start)
  Y       (N, 6)           residual-acceleration targets (aligned to last frame)
  k       (N,)             the disturbance factor for each sample (for analysis)
  meta    dict             config used

Usage:
  python -m robust_safe_rl.scripts.collect_dynamics_data \
      --episodes 200 --history 10 --out data/dyn_massmoi.npz
"""

import argparse
import os

import numpy as np

from robust_safe_rl.core import Controller, DesiredTrajectory, Dynamics
from robust_safe_rl.rl.mixer import Mixer, NOMINAL_ARM
from robust_safe_rl.rl.config import Config


FRAME_DIM = 19   # R(9) + v(3) + omega(3) + u_total(4)
TARGET_DIM = 6   # residual accel: translational(3) + angular(3)


def collect(cfg, episodes, history, seed=0, k_min=None, k_max=None,
            disturbance="massmoi", force_mode=1, param_min=0.7, param_max=1.3,
            per_axis=False):
    """Collect residual-dynamics data under a chosen disturbance type.

    per_axis: for thrust_factor / arm_length, if True perturb each motor/arm
    independently (length-4 draws); if False use a single global multiplier.
    """
    """Collect residual-dynamics data under a chosen disturbance type.

    disturbance:
      "massmoi" : per-episode mass/inertia scaling k ~ U[k_min,k_max] (in-dist).
      "force"   : per-episode constant external force via Dynamics.random_force
                  mode (1=ID band, 2=loose OOD, 3=strict OOD). Mass/inertia stay
                  nominal. This is the OOD disturbance TYPE for the detector test.

    The residual-acceleration target is always a_true - a_nominal at the SAME
    state: for "massmoi" the gap comes from the (m,J) mismatch; for "force" it
    comes from the external force (true includes it, nominal does not).
    """
    env_cfg = cfg.env
    dt = env_cfg.dt
    k_min = env_cfg.k_min if k_min is None else k_min
    k_max = env_cfg.k_max if k_max is None else k_max

    J_nom = np.diag(np.asarray(env_cfg.J_nom, dtype=float))
    mass_nom = env_cfg.mass_nom

    # For "force" we let Dynamics sample the external force per episode via its
    # random_force mode; for "massmoi" force stays off (mode 0).
    rf = force_mode if disturbance == "force" else 0
    dyn = Dynamics(dt=dt, mass=mass_nom, J=J_nom, gravity=env_cfg.gravity, random_force=rf)
    mixer = Mixer()   # nominal == true until perturbed for thrust_factor / arm_length
    ctrl = Controller(dt=dt, mass=mass_nom, J=J_nom, gravity=env_cfg.gravity)
    traj = DesiredTrajectory(radius=env_cfg.traj_radius, speed=env_cfg.traj_speed,
                             z0=env_cfg.traj_z0)
    rng = np.random.default_rng(seed)

    X_list, Y_list, k_list = [], [], []

    for ep in range(episodes):
        # Per-episode disturbance parameters. Defaults = nominal (no disturbance).
        k = 1.0                       # mass/MOI multiplier
        ep_force = np.zeros(3)        # external force
        mixer.reset_true()            # thrust/arm disturbances go through the mixer

        if disturbance == "massmoi":
            k = float(rng.uniform(k_min, k_max))
            dyn.set_inertial_scale(k)
            ep_tag = k
        elif disturbance == "force":
            dyn.set_inertial_scale(1.0)
            ep_force = dyn.sample_external_force()
            ep_tag = float(np.linalg.norm(ep_force))
        elif disturbance == "thrust_factor":
            # true motors have scaled k_f (strength); affects f AND M via allocation
            dyn.set_inertial_scale(1.0)
            if per_axis:
                kf_scale = rng.uniform(param_min, param_max, size=4)
                ep_tag = float(np.mean(kf_scale))
            else:
                kf_scale = float(rng.uniform(param_min, param_max))
                ep_tag = float(kf_scale)
            mixer.set_true(kf_scale=kf_scale)
        elif disturbance == "arm_length":
            # true rotor arms scaled; affects moments (not thrust) via allocation
            dyn.set_inertial_scale(1.0)
            if per_axis:
                arm = NOMINAL_ARM * rng.uniform(param_min, param_max, size=4)
                ep_tag = float(np.mean(arm) / NOMINAL_ARM)
            else:
                s = float(rng.uniform(param_min, param_max))
                arm = NOMINAL_ARM * s
                ep_tag = s
            mixer.set_true(arm_x=arm)
        else:
            raise ValueError(f"unknown disturbance: {disturbance!r}")

        d0 = traj.desired(0.0)
        dyn.reset(x=d0["x"], v=d0["v"], external_force=ep_force)
        ctrl.reset()

        # per-episode ring buffer of input frames, zero-padded
        frames = np.zeros((history, FRAME_DIM), dtype=np.float64)
        t = 0.0

        for step in range(env_cfg.episode_steps):
            st = dyn.state()

            # Divergence guard: the base-only light-drone (k<1) case can blow up.
            # Stop BEFORE the moments explode -- use a tighter angular-rate/tilt
            # bound so we don't log the pathological last few steps that would
            # otherwise dominate the regression with huge outlier targets.
            if (not np.all(np.isfinite(st["x"]))
                    or np.linalg.norm(st["v"]) > 10.0
                    or np.linalg.norm(st["omega"]) > 10.0
                    or st["R"][2, 2] < 0.5):   # >60 deg tilt
                break

            desired = traj.desired(t)

            # base control on the true state (controller uses nominal m,J)
            f, M, _ = ctrl.compute_control(st, desired)

            # Applied wrench to the TRUE plant, with parameter disturbances:
            #   thrust_factor scales f and M together; arm_length scales M only.
            # Applied wrench to the TRUE plant. For thrust_factor / arm_length the
            # mixer round-trip (allocate with nominal, reconstruct with true)
            # produces the actual wrench; for massmoi / force it is the identity.
            f_applied, M_applied = mixer.apply(f, M)
            u_total = np.array([f_applied, M_applied[0], M_applied[1], M_applied[2]])

            # ---- residual-acceleration target: a_true - a_nominal at same state ----
            # a_true : the true plant's acceleration under its disturbance and the
            #          ACTUAL applied wrench.
            # a_nom  : what the nominal model predicts for the COMMANDED wrench with
            #          nominal params and no force -- i.e. what the controller thinks
            #          it is producing. The residual is the gap the disturbance opens.
            vd_true, wd_true = dyn.accel_at(
                st["R"], st["v"], st["omega"], f_applied, M_applied,
                mass=k * mass_nom, J=k * J_nom, external_force=ep_force)
            vd_nom, wd_nom = dyn.accel_at(
                st["R"], st["v"], st["omega"], f, M,
                mass=mass_nom, J=J_nom, external_force=np.zeros(3))
            a_res = np.concatenate([vd_true - vd_nom, wd_true - wd_nom])

            # ---- input frame (no position) ----
            frame = np.concatenate([st["R"].reshape(9), st["v"], st["omega"], u_total])
            frames = np.roll(frames, -1, axis=0)
            frames[-1] = frame

            X_list.append(frames.copy())
            Y_list.append(a_res)
            k_list.append(ep_tag)

            # advance the true plant with the ACTUALLY applied (gained) wrench
            dyn.step(f_applied, M_applied)
            t = (step + 1) * dt

        if (ep + 1) % 20 == 0:
            label = {"massmoi": "k", "force": "|F|",
                     "thrust_factor": "tf", "arm_length": "arm"}.get(disturbance, "d")
            print(f"  episode {ep + 1}/{episodes}  ({label}={ep_tag:.3f})  "
                  f"samples so far: {len(X_list)}")

    X = np.asarray(X_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.float32)
    k_arr = np.asarray(k_list, dtype=np.float32)
    return X, Y, k_arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--history", type=int, default=10,
                   help="H_dyn: input history length. Use 1 for the MEMORYLESS "
                        "model (single frame, 19-dim input); use >1 to stack that "
                        "many past frames so the net can identify k from the state "
                        "evolution. Ablate 1 vs 10 to test whether history helps.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--disturbance", default="massmoi",
                   choices=["massmoi", "force", "thrust_factor", "arm_length"],
                   help="disturbance type: massmoi/force (as before), or the OOD "
                        "parameter cases thrust_factor / arm_length")
    p.add_argument("--force_mode", type=int, default=1, choices=[1, 2, 3],
                   help="force band when --disturbance force: 1=ID[-3,3], 2=loose[-5,5], 3=strict OOD")
    p.add_argument("--param_min", type=float, default=0.7,
                   help="min multiplier for thrust_factor / arm_length disturbances")
    p.add_argument("--param_max", type=float, default=1.3,
                   help="max multiplier for thrust_factor / arm_length disturbances")
    p.add_argument("--per_axis", action="store_true",
                   help="perturb each motor/arm independently (default: global multiplier)")
    p.add_argument("--k_min", type=float, default=None)
    p.add_argument("--k_max", type=float, default=None)
    p.add_argument("--out", default="data/dyn_massmoi.npz")
    args = p.parse_args()

    cfg = Config()
    X, Y, k = collect(cfg, args.episodes, args.history, seed=args.seed,
                      k_min=args.k_min, k_max=args.k_max,
                      disturbance=args.disturbance, force_mode=args.force_mode,
                      param_min=args.param_min, param_max=args.param_max,
                      per_axis=args.per_axis)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, X=X, Y=Y, k=k,
             history=args.history, frame_dim=FRAME_DIM, target_dim=TARGET_DIM)
    print(f"\nsaved {X.shape[0]} samples -> {args.out}")
    print(f"  X shape {X.shape}  Y shape {Y.shape}")
    print(f"  target stats: mean |a_res| = {np.mean(np.abs(Y), axis=0)}")
    print(f"  k range: [{k.min():.3f}, {k.max():.3f}]")


if __name__ == "__main__":
    main()