#!/usr/bin/env python3
"""
Record a sim2sim tracking run as an MP4 video (headless, no GUI).

Runs both the MuJoCo simulator and the BFM-Zero policy in a single process,
bypassing ZMQ entirely. Renders offscreen via mujoco.Renderer.

Usage:
    # Single file mode
    python scripts/record_tracking.py \
        --task config/exp/tracking/walking.yaml \
        --output output.mp4

    # Full options
    python scripts/record_tracking.py \
        --robot_config config/robot/g1.yaml \
        --policy_config config/policy/motivo_newG1.yaml \
        --scene_config config/scene/g1_29dof.yaml \
        --model_path ./model/exported/FBcprAuxModel.onnx \
        --task config/exp/tracking/walking.yaml \
        --output output.mp4 \
        --width 1920 --height 1080 --fps 50 \
        --duration 10.0 \
        --camera tracking \
        --init_duration 2.0

    # Batch mode: record videos for all .pkl files in a directory
    python scripts/record_tracking.py \
        --input_dir model/tracking_inference/ \
        --output_dir video/tracking/
"""

import os

# Force EGL backend for headless offscreen rendering (must be set before importing mujoco)
os.environ.setdefault("MUJOCO_GL", "egl")

import sys
import argparse
import subprocess
import numpy as np
import yaml
import joblib
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from utils.strings import resolve_matching_names_values, unitree_joint_names
from utils.math import quat_mul, quat_conjugate, yaw_quat
from rl_policy.observations import Observation, ObsGroup


# ---------------------------------------------------------------------------
# Lightweight shims that replace ZMQ-based StateProcessor / CommandSender
# ---------------------------------------------------------------------------

class DirectStateProcessor:
    """Drop-in for StateProcessor that reads MuJoCo state directly."""

    def __init__(self, dest_joint_names, mj_model, mj_data, sim_joint_indices, root_qpos_adr, root_qvel_adr):
        self.num_dof = len(dest_joint_names)
        self.joint_names = dest_joint_names

        # MuJoCo handles
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.sim_joint_indices = sim_joint_indices  # (unitree_idx, qpos_adr, qvel_adr) tuples
        self.root_qpos_adr = root_qpos_adr
        self.root_qvel_adr = root_qvel_adr

        # Mapping from unitree ordering to isaac (dest) ordering
        self.joint_indices_in_source = [
            unitree_joint_names.index(name) for name in dest_joint_names
        ]

        # State arrays (same layout as real StateProcessor)
        self.qpos = np.zeros(3 + 4 + self.num_dof)
        self.qvel = np.zeros(3 + 3 + self.num_dof)
        self.root_pos_w = self.qpos[0:3]
        self.root_lin_vel_w = self.qvel[0:3]
        self.root_quat_b = self.qpos[3:7]
        self.root_ang_vel_b = self.qvel[3:6]
        self.joint_pos = self.qpos[7:]
        self.joint_vel = self.qvel[6:]

    def _prepare_low_state(self):
        """Read state directly from mj_data (replaces ZMQ receive)."""
        mj_data = self.mj_data

        # Build full unitree-order joint arrays from MuJoCo data
        joint_pos_full = np.zeros(len(unitree_joint_names), dtype=np.float32)
        joint_vel_full = np.zeros(len(unitree_joint_names), dtype=np.float32)
        for unitree_idx, qpos_adr, qvel_adr in self.sim_joint_indices:
            joint_pos_full[unitree_idx] = mj_data.qpos[qpos_adr]
            joint_vel_full[unitree_idx] = mj_data.qvel[qvel_adr]

        # Root orientation: extract body-frame quaternion (same as SimulationBridge)
        root_quat_w = mj_data.qpos[self.root_qpos_adr + 3:self.root_qpos_adr + 7]
        root_quat_yaw_w = yaw_quat(root_quat_w)
        root_quat_b = quat_mul(quat_conjugate(root_quat_yaw_w), root_quat_w)

        # Root angular velocity in body frame
        root_ang_vel_b = mj_data.qvel[self.root_qvel_adr + 3:self.root_qvel_adr + 6]

        # Write into state arrays
        self.root_quat_b[:] = root_quat_b
        self.root_ang_vel_b[:] = root_ang_vel_b
        for dst_idx, src_idx in enumerate(self.joint_indices_in_source):
            self.joint_pos[dst_idx] = joint_pos_full[src_idx]
            self.joint_vel[dst_idx] = joint_vel_full[src_idx]

        return True


class DirectCommandSender:
    """Drop-in for CommandSender that writes torques to mj_data directly."""

    def __init__(self, policy_config, mj_model, mj_data, sim_joint_indices):
        self.mj_model = mj_model
        self.mj_data = mj_data

        # Resolve KP/KD in unitree ordering
        joint_kp_dict = policy_config["joint_kp"]
        indices, _, kp_vals = resolve_matching_names_values(
            joint_kp_dict, unitree_joint_names, preserve_order=True, strict=False
        )
        self.joint_kp = np.zeros(len(unitree_joint_names))
        self.joint_kp[indices] = kp_vals

        joint_kd_dict = policy_config["joint_kd"]
        indices, _, kd_vals = resolve_matching_names_values(
            joint_kd_dict, unitree_joint_names, preserve_order=True, strict=False
        )
        self.joint_kd = np.zeros(len(unitree_joint_names))
        self.joint_kd[indices] = kd_vals

        # Default joint positions in unitree ordering
        default_pos_dict = policy_config["default_joint_pos"]
        indices, _, default_vals = resolve_matching_names_values(
            default_pos_dict, unitree_joint_names, preserve_order=True, strict=False
        )
        self.default_joint_pos_unitree = np.zeros(len(unitree_joint_names))
        self.default_joint_pos_unitree[indices] = default_vals

        # Isaac to unitree mapping
        isaac_names = policy_config["isaac_joint_names"]
        self.joint_indices_unitree = [
            unitree_joint_names.index(name) for name in isaac_names
        ]

        # Sim joint mapping: (unitree_idx, qpos_adr, qvel_adr, act_adr)
        self.sim_joint_indices = sim_joint_indices

        # Current command in unitree ordering
        self.cmd_q = self.default_joint_pos_unitree.copy()
        self.cmd_dq = np.zeros(len(unitree_joint_names))
        self.cmd_tau = np.zeros(len(unitree_joint_names))

        # Effort limits from MuJoCo model
        joint_names_mujoco = [mj_model.joint(i).name for i in range(mj_model.njnt)]
        actuator_names_mujoco = [
            mj_model.actuator(i).name + "_joint" for i in range(mj_model.nu)
        ]
        from utils.strings import resolve_matching_names_values as rmn
        # We'll compute effort limits from robot config below
        self.torques = np.zeros(mj_model.nu)

        # kp_level for compatibility
        self._kp_level = 1.0
        self.has_command = False

    @property
    def kp_level(self):
        return self._kp_level

    @kp_level.setter
    def kp_level(self, value):
        self._kp_level = value

    def send_command(self, cmd_q, cmd_dq, cmd_tau):
        """Store command targets (isaac ordering -> unitree ordering).
        Actual PD torques are computed per sim step via apply_torques()."""
        self.cmd_q[self.joint_indices_unitree] = cmd_q
        self.cmd_dq[self.joint_indices_unitree] = cmd_dq
        self.cmd_tau[self.joint_indices_unitree] = cmd_tau
        self.has_command = True

    def apply_torques(self):
        """Recompute PD torques from current mj_data state and write to ctrl.
        Must be called every sim step (200Hz) for stable PD control."""
        if not self.has_command:
            return
        self.torques[:] = 0.0
        for unitree_idx, qpos_adr, qvel_adr, act_adr in self.sim_joint_indices:
            q_des = self.cmd_q[unitree_idx]
            dq_des = self.cmd_dq[unitree_idx]
            tau_ff = self.cmd_tau[unitree_idx]
            kp = self.joint_kp[unitree_idx] * self._kp_level
            kd = self.joint_kd[unitree_idx]
            self.torques[act_adr] = (
                tau_ff
                + kp * (q_des - self.mj_data.qpos[qpos_adr])
                + kd * (dq_des - self.mj_data.qvel[qvel_adr])
            )
        self.mj_data.ctrl[:] = self.torques


# ---------------------------------------------------------------------------
# Joint index mapping (same logic as SimulationBridge.init_joint_indices)
# ---------------------------------------------------------------------------

def build_sim_joint_indices(mj_model):
    """Returns list of (unitree_idx, qpos_adr, qvel_adr, act_adr) and root addresses."""
    joint_names_mujoco = [mj_model.joint(i).name for i in range(mj_model.njnt)]
    actuator_names_mujoco = [
        mj_model.actuator(i).name + "_joint" for i in range(mj_model.nu)
    ]

    sim_joint_indices = []
    shared = set(joint_names_mujoco) & set(unitree_joint_names)
    for name in shared:
        unitree_idx = unitree_joint_names.index(name)
        joint_idx = joint_names_mujoco.index(name)
        qpos_adr = mj_model.jnt_qposadr[joint_idx]
        qvel_adr = mj_model.jnt_dofadr[joint_idx]
        act_adr = actuator_names_mujoco.index(name)
        sim_joint_indices.append((unitree_idx, qpos_adr, qvel_adr, act_adr))

    if "floating_base_joint" in joint_names_mujoco:
        root_idx = joint_names_mujoco.index("floating_base_joint")
    elif "pelvis_root" in joint_names_mujoco:
        root_idx = joint_names_mujoco.index("pelvis_root")
    else:
        raise ValueError("No root joint found")
    root_qpos_adr = mj_model.jnt_qposadr[root_idx]
    root_qvel_adr = mj_model.jnt_dofadr[root_idx]

    return sim_joint_indices, root_qpos_adr, root_qvel_adr


# ---------------------------------------------------------------------------
# Build policy without ZMQ (monkey-patch approach)
# ---------------------------------------------------------------------------

def build_policy_headless(robot_config, policy_config, exp_config, model_path,
                          mj_model, mj_data, sim_joint_indices, root_qpos_adr, root_qvel_adr):
    """Construct BFMZeroPolicy with direct state/command instead of ZMQ."""
    from rl_policy.bfm_zero import BFMZeroPolicy

    # Temporarily override robot type to avoid ZMQ/keyboard init
    patched_robot_config = dict(robot_config)
    patched_robot_config["ROBOT_TYPE"] = "g1_29dof"

    # StateProcessor.__init__ creates ZMQ sockets. We'll replace the whole object
    # after construction. To avoid ZMQ socket creation during __init__, we
    # monkey-patch StateProcessor temporarily.
    from rl_policy.utils import state_processor as sp_module
    from rl_policy.utils import command_sender as cs_module

    orig_sp_init = sp_module.StateProcessor.__init__
    orig_cs_init = cs_module.CommandSender.__init__

    isaac_names = policy_config["isaac_joint_names"]

    # Lightweight state indices for DirectStateProcessor
    state_indices_for_sp = [
        (u_idx, qp, qv) for u_idx, qp, qv, _ in sim_joint_indices
    ]

    direct_sp = DirectStateProcessor(
        isaac_names, mj_model, mj_data,
        state_indices_for_sp, root_qpos_adr, root_qvel_adr,
    )
    direct_cs = DirectCommandSender(
        policy_config, mj_model, mj_data, sim_joint_indices,
    )

    # Patch __init__ to copy attributes from our direct implementations
    # so that setup_observations() works during BFMZeroPolicy.__init__
    def patched_sp_init(self, *a, **kw):
        self.__dict__.update(direct_sp.__dict__)

    def patched_cs_init(self, *a, **kw):
        self.__dict__.update(direct_cs.__dict__)

    sp_module.StateProcessor.__init__ = patched_sp_init
    cs_module.CommandSender.__init__ = patched_cs_init

    # Prevent keyboard listener from starting
    import threading
    orig_thread_start = threading.Thread.start

    def noop_start(self):
        pass

    threading.Thread.start = noop_start

    try:
        policy = BFMZeroPolicy(
            robot_config=patched_robot_config,
            policy_config=policy_config,
            exp_config=exp_config,
            model_path=model_path,
            rl_rate=50,
        )
    finally:
        sp_module.StateProcessor.__init__ = orig_sp_init
        cs_module.CommandSender.__init__ = orig_cs_init
        threading.Thread.start = orig_thread_start

    # Ensure the direct implementations are used (they share the same __dict__)
    policy.state_processor = direct_sp
    policy.command_sender = direct_cs

    return policy


# ---------------------------------------------------------------------------
# Camera setup
# ---------------------------------------------------------------------------

def setup_camera(mj_model, camera_mode, distance=3.0, elevation=-20.0, azimuth=90.0):
    """Create a mujoco camera for rendering."""
    cam = mujoco.MjvCamera()
    if camera_mode == "tracking":
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mj_model.body("pelvis").id
        cam.distance = distance
        cam.elevation = elevation
        cam.azimuth = azimuth
    elif camera_mode == "fixed":
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = 0  # first camera in XML
    else:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = distance
        cam.elevation = elevation
        cam.azimuth = azimuth
    return cam


# ---------------------------------------------------------------------------
# Stabilize + Record
# ---------------------------------------------------------------------------

def stabilize_and_record(policy, mj_model, mj_data, renderer, cam,
                         sim_joint_indices, root_qpos_adr, root_qvel_adr,
                         output_path, total_rl_steps, sim_steps_per_rl,
                         init_rl_steps, fps, width, height):
    """Reset sim, stabilize standing pose, then record policy rollout to MP4."""

    # Reset MuJoCo to initial state
    mujoco.mj_resetData(mj_model, mj_data)

    # Set joint qpos to policy default angles
    for i, name in enumerate(policy.isaac_joint_names):
        unitree_idx = unitree_joint_names.index(name)
        for u_idx, qpos_adr, qvel_adr, act_adr in sim_joint_indices:
            if u_idx == unitree_idx:
                mj_data.qpos[qpos_adr] = policy.default_dof_angles[i]
                break

    init_root_qpos = mj_data.qpos[root_qpos_adr:root_qpos_adr + 7].copy()
    mujoco.mj_forward(mj_model, mj_data)

    # Stabilize: PD hold at default pose with root pinned
    policy.use_policy_action = False
    policy.get_ready_state = False
    policy.last_action = np.zeros(policy.num_actions)
    policy.state_dict = {"action": np.zeros(policy.num_actions)}
    policy.command_sender.has_command = False
    policy.command_sender.send_command(
        policy.default_dof_angles.copy(), np.zeros(policy.num_dofs), np.zeros(policy.num_dofs))

    for _ in range(init_rl_steps):
        for _ in range(sim_steps_per_rl):
            policy.command_sender.apply_torques()
            mujoco.mj_step(mj_model, mj_data)
            mj_data.qpos[root_qpos_adr:root_qpos_adr + 7] = init_root_qpos
            mj_data.qvel[root_qvel_adr:root_qvel_adr + 6] = 0.0

    # Activate policy
    policy.get_ready_state = False
    policy.use_policy_action = True
    policy.start_motion = True
    policy.t = policy.t_start
    policy.reset()

    # Start ffmpeg
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "rgb24",
        "-r", str(fps), "-i", "-",
        "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
        output_path,
    ]
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    for step in range(total_rl_steps):
        policy.state_processor._prepare_low_state()

        try:
            obs_dict, observations = policy.prepare_obs_for_rl()
            action = policy.policy(observations)
            action = action.clip(-1, 1)
            action_scaled = policy.action_rescale * action
            policy.last_action = action_scaled

            policy_action = np.zeros(policy.num_dofs)
            policy_action[policy.controlled_joint_indices] = action_scaled
            policy_action = policy_action * policy.action_scale
            q_target = policy_action + policy.default_dof_angles
            q_target = np.clip(q_target, policy.joint_pos_lower_limit, policy.joint_pos_upper_limit)
            policy.command_sender.send_command(q_target, np.zeros(policy.num_dofs), np.zeros(policy.num_dofs))
        except Exception as e:
            print(f"  Policy error at step {step}: {e}")
            policy.state_dict["action"] = np.zeros(policy.num_actions)

        for _ in range(sim_steps_per_rl):
            policy.command_sender.apply_torques()
            mujoco.mj_step(mj_model, mj_data)

        renderer.update_scene(mj_data, camera=cam)
        frame = renderer.render()
        ffmpeg_proc.stdin.write(frame.tobytes())

        if (step + 1) % 200 == 0:
            print(f"  {step + 1}/{total_rl_steps} frames")

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    if ffmpeg_proc.returncode != 0:
        stderr = ffmpeg_proc.stderr.read().decode()
        print(f"  ffmpeg error: {stderr}")
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Record sim2sim tracking as MP4 (headless)")
    parser.add_argument("--robot_config", default="config/robot/g1.yaml")
    parser.add_argument("--policy_config", default="config/policy/motivo_newG1.yaml")
    parser.add_argument("--scene_config", default="config/scene/g1_29dof.yaml")
    parser.add_argument("--model_path", default="./model/exported/FBcprAuxModel.onnx")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=50, help="Video FPS (default: 50, matches policy rate)")
    parser.add_argument("--camera", default="tracking", choices=["tracking", "fixed", "free"])
    parser.add_argument("--distance", type=float, default=3.0, help="Camera distance")
    parser.add_argument("--elevation", type=float, default=-20.0, help="Camera elevation (degrees)")
    parser.add_argument("--azimuth", type=float, default=90.0, help="Camera azimuth (degrees)")
    parser.add_argument("--init_duration", type=float, default=2.0,
                        help="Seconds to run init pose before tracking starts (default: 2.0)")
    # Single mode
    parser.add_argument("--task", default=None,
                        help="Experiment config YAML (e.g. config/exp/tracking/walking.yaml)")
    parser.add_argument("--output", "-o", default=None, help="Output MP4 path")
    parser.add_argument("--duration", type=float, default=None,
                        help="Recording duration in seconds (default: auto from z frames)")
    # Batch mode
    parser.add_argument("--input_dir", default=None,
                        help="Input directory containing .pkl z-vector files (batch mode)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for MP4 videos (batch mode)")
    args = parser.parse_args()

    # Validate mode
    batch_mode = args.input_dir is not None or args.output_dir is not None
    single_mode = args.task is not None or args.output is not None
    if batch_mode and single_mode:
        parser.error("Cannot use --task/--output together with --input_dir/--output_dir")
    if batch_mode and (args.input_dir is None or args.output_dir is None):
        parser.error("Batch mode requires both --input_dir and --output_dir")
    if not batch_mode:
        if args.task is None:
            parser.error("Single mode requires --task")
        if args.output is None:
            args.output = "output.mp4"

    # Load configs
    with open(args.robot_config) as f:
        robot_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.policy_config) as f:
        policy_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.scene_config) as f:
        scene_config = yaml.load(f, Loader=yaml.FullLoader)

    sim_dt = scene_config["SIMULATE_DT"]
    rl_rate = args.fps
    sim_steps_per_rl = max(1, int(round((1.0 / rl_rate) / sim_dt)))
    init_rl_steps = int(args.init_duration * rl_rate)

    # Initialize MuJoCo
    mj_model = mujoco.MjModel.from_xml_path(scene_config["ROBOT_SCENE"])
    mj_data = mujoco.MjData(mj_model)
    mj_model.opt.timestep = sim_dt
    mj_model.vis.global_.offwidth = max(mj_model.vis.global_.offwidth, args.width)
    mj_model.vis.global_.offheight = max(mj_model.vis.global_.offheight, args.height)

    sim_joint_indices, root_qpos_adr, root_qvel_adr = build_sim_joint_indices(mj_model)

    # --- Build exp_config for policy construction ---
    if batch_mode:
        # Collect all .pkl files
        input_dir = os.path.abspath(args.input_dir)
        pkl_files = []
        for root, _, files in os.walk(input_dir):
            for f in sorted(files):
                if f.endswith(".pkl"):
                    pkl_files.append(os.path.join(root, f))
        if not pkl_files:
            print(f"No .pkl files found in {input_dir}")
            return

        # Build a temporary exp_config using the first pkl so the policy can init
        first_ctx = joblib.load(pkl_files[0])
        n_frames = first_ctx.shape[0]
        exp_config = {
            "type": "tracking",
            "start": 0,
            "end": n_frames,
            "stop": 0,
            "ctx_path": os.path.relpath(pkl_files[0], os.path.dirname(args.model_path)),
            "gamma": 0.9,
            "window_size": 1,
        }
    else:
        with open(args.task) as f:
            exp_config = yaml.load(f, Loader=yaml.FullLoader)

    # Build policy once (headless, no ZMQ)
    print("Building policy (headless) ...")
    policy = build_policy_headless(
        robot_config, policy_config, exp_config, args.model_path,
        mj_model, mj_data, sim_joint_indices, root_qpos_adr, root_qvel_adr,
    )

    renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)
    cam = setup_camera(mj_model, args.camera, args.distance, args.elevation, args.azimuth)

    if batch_mode:
        output_dir = os.path.abspath(args.output_dir)
        print(f"\nBatch mode: {len(pkl_files)} .pkl files in {input_dir}\n")

        for idx, pkl_path in enumerate(pkl_files, 1):
            rel_path = os.path.relpath(pkl_path, input_dir)
            stem = os.path.splitext(rel_path)[0]
            out_path = os.path.join(output_dir, stem + ".mp4")

            # Load z-vectors and determine duration
            ctx = joblib.load(pkl_path)
            n_frames = ctx.shape[0]
            total_rl_steps = n_frames if args.duration is None else int(args.duration * rl_rate)

            print(f"[{idx}/{len(pkl_files)}] {rel_path}  "
                  f"({n_frames} frames, {total_rl_steps / rl_rate:.1f}s)")

            # Swap context into policy
            policy.ctx = ctx
            policy.t_start = 0
            policy.t_end = n_frames
            policy.t_stop = 0

            try:
                ok = stabilize_and_record(
                    policy, mj_model, mj_data, renderer, cam,
                    sim_joint_indices, root_qpos_adr, root_qvel_adr,
                    out_path, total_rl_steps, sim_steps_per_rl,
                    init_rl_steps, rl_rate, args.width, args.height,
                )
                if ok:
                    print(f"  -> {out_path}")
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        print(f"\nBatch complete. Videos in {output_dir}")
    else:
        # Single mode: determine duration
        if args.duration is not None:
            total_rl_steps = int(args.duration * rl_rate)
        else:
            total_rl_steps = policy.ctx.shape[0]

        print(f"Recording: {total_rl_steps} frames ({total_rl_steps / rl_rate:.1f}s)")

        ok = stabilize_and_record(
            policy, mj_model, mj_data, renderer, cam,
            sim_joint_indices, root_qpos_adr, root_qvel_adr,
            args.output, total_rl_steps, sim_steps_per_rl,
            init_rl_steps, rl_rate, args.width, args.height,
        )
        if ok:
            print(f"Done! Video saved to {args.output}")


if __name__ == "__main__":
    main()
