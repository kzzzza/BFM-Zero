#!/usr/bin/env python3
"""
Convert motion data (.npz) to tracking z vectors (.pkl)
compatible with BFM-Zero_deploy/model/tracking_inference/.

Supports two input formats:
  1. data/ format (amass/lafan): joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, fps
  2. example_motion/ format:     dof_positions, body_positions, body_rotations, fps [, dof_names]

Usage:
    # data/ format (velocities included)
    python scripts/motion_to_tracking_z.py \
        --input ../BFM-Zero_inf/data/amass/Form_1_stageii.npz \
        --output model/tracking_inference/zs_form1.pkl

    # example_motion/ format (velocities auto-computed via finite difference)
    python scripts/motion_to_tracking_z.py \
        --input ../BFM-Zero_inf/example_motion/dance1_subject2_50_jpos.npz \
        --output model/tracking_inference/zs_dance1.pkl

    # Then update config/exp/tracking/ yaml to point to the new .pkl file.

    # Batch conversion: convert all .npz files in a directory
    python scripts/motion_to_tracking_z.py \
        --input_dir ../BFM-Zero_inf/data/amass/ \
        --output_dir model/tracking_inference/amass/
    # Preserves directory structure, appends _z to filenames:
    #   amass/sub1/walk.npz -> amass/sub1/walk_z.pkl

注意，需要在bfm0inf环境下运行，
确保安装了BFM-Zero_inf依赖，并且模型checkpoint存在于指定路径（默认 ../BFM-Zero_inf/model/checkpoint/model）。
"""

import sys
import os
import argparse
import numpy as np
import torch
import joblib

# Add inference repo to path
INF_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "BFM-Zero_inf")
sys.path.insert(0, INF_REPO)

from bfm_zero_inference_code.fb_cpr_aux.model import FBcprAuxModel
from env import MuJoCoBFMZeroEnv, calc_angular_velocity
from common import ACTION_SCALES, KP_GAINS, KD_GAINS, DEFAULT_JOINT_POS, ACTION_RESCALE

# data/ format (from GMR) uses BMimic joint ordering, which differs from the Isaac ordering
# that example_motion/ and the model expect. G1_JOINT_MAPPING[i] gives the Isaac index for
# BMimic position i, i.e.  bmimic_joints[i] = isaac_joints[G1_JOINT_MAPPING[i]].
# To convert data/ → Isaac: isaac_joints = bmimic_joints[BMIMIC_TO_ISAAC_JOINT].
G1_JOINT_MAPPING = [
    0, 6, 12,
    1, 7, 13,
    2, 8, 14,
    3,  9, 15, 22,
    4, 10, 16, 23,
    5, 11, 17, 24,
          18, 25,
          19, 26,
          20, 27,
          21, 28,
]
# Inverse: for each Isaac index, which BMimic index holds the value?
BMIMIC_TO_ISAAC_JOINT = [0] * 29
for _bmimic_idx, _isaac_idx in enumerate(G1_JOINT_MAPPING):
    BMIMIC_TO_ISAAC_JOINT[_isaac_idx] = _bmimic_idx

# BMimic body ordering also differs, but build_observations() only reads root (index 0 = pelvis
# in both orderings), so body reordering is not needed for tracking z generation.


def _detect_format(data):
    """Detect npz format: 'data' (amass/lafan) or 'example_motion'."""
    keys = set(data.files)
    if "joint_pos" in keys and "joint_vel" in keys:
        return "data"
    if "dof_positions" in keys and "body_positions" in keys:
        return "example_motion"
    raise ValueError(
        f"Unrecognized npz format. Keys: {sorted(keys)}\n"
        "Expected either data/ format (joint_pos, joint_vel, body_pos_w, ...) "
        "or example_motion/ format (dof_positions, body_positions, body_rotations, ...)"
    )


def _compute_velocities_finite_diff(joint_pos, body_pos, body_quat, fps):
    """Compute velocities via finite difference for example_motion/ format.

    For frame 0, velocities are set to zero.
    """
    n_frames = joint_pos.shape[0]
    dt = 1.0 / fps

    joint_vel = np.zeros_like(joint_pos, dtype=np.float32)
    body_lin_vel = np.zeros_like(body_pos, dtype=np.float32)
    body_ang_vel = np.zeros_like(body_pos, dtype=np.float32)

    # Frames 1..N-1: finite difference
    joint_vel[1:] = (joint_pos[1:] - joint_pos[:-1]) / dt
    body_lin_vel[1:] = (body_pos[1:] - body_pos[:-1]) / dt

    for i in range(1, n_frames):
        body_ang_vel[i] = calc_angular_velocity(body_quat[i], body_quat[i - 1], dt)

    return joint_vel, body_lin_vel, body_ang_vel


def load_motion_data(npz_path, start=0, end=-1):
    """Load motion data from .npz file, auto-detecting format.

    Supports:
      - data/ format: joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, fps
      - example_motion/ format: dof_positions, body_positions, body_rotations, fps [, dof_names]
    """
    data = np.load(npz_path, allow_pickle=True)
    fmt = _detect_format(data)

    fps_raw = data["fps"]
    fps = int(fps_raw.item() if fps_raw.ndim > 0 else fps_raw)

    if fmt == "data":
        # data/ format uses BMimic joint ordering — reorder to Isaac ordering
        joint_pos = data["joint_pos"][:, BMIMIC_TO_ISAAC_JOINT].astype(np.float32)
        joint_vel = data["joint_vel"][:, BMIMIC_TO_ISAAC_JOINT].astype(np.float32)
        body_pos_w = data["body_pos_w"].astype(np.float32)
        body_quat_w = data["body_quat_w"].astype(np.float32)
        body_lin_vel_w = data["body_lin_vel_w"].astype(np.float32)
        body_ang_vel_w = data["body_ang_vel_w"].astype(np.float32)
        print(f"  Detected data/ format (velocities included, joints reordered BMimic→Isaac)")
    else:
        joint_pos = data["dof_positions"].astype(np.float32)
        body_pos_w = data["body_positions"].astype(np.float32)
        body_quat_w = data["body_rotations"].astype(np.float32)
        print(f"  Detected example_motion/ format (computing velocities via finite difference)")
        joint_vel, body_lin_vel_w, body_ang_vel_w = _compute_velocities_finite_diff(
            joint_pos, body_pos_w, body_quat_w, fps
        )

    n_frames = joint_pos.shape[0]
    if end <= 0:
        end = n_frames
    end = min(end, n_frames)
    start = max(0, start)

    return {
        "fps": fps,
        "joint_pos": joint_pos[start:end],
        "joint_vel": joint_vel[start:end],
        "body_pos_w": body_pos_w[start:end],
        "body_quat_w": body_quat_w[start:end],
        "body_lin_vel_w": body_lin_vel_w[start:end],
        "body_ang_vel_w": body_ang_vel_w[start:end],
    }


def build_observations(motion, env):
    """Iterate over motion frames, set MuJoCo state, and collect backward observations."""
    joint_pos = motion["joint_pos"]
    joint_vel = motion["joint_vel"]
    body_pos_w = motion["body_pos_w"]
    body_quat_w = motion["body_quat_w"]
    body_lin_vel_w = motion["body_lin_vel_w"]
    body_ang_vel_w = motion["body_ang_vel_w"]

    n_frames = joint_pos.shape[0]
    obs_list = []

    env.reset()

    for i in range(n_frames):
        root_pos = body_pos_w[i, 0, :]
        root_quat = body_quat_w[i, 0, :]        # [w, x, y, z]
        root_vel = body_lin_vel_w[i, 0, :]
        root_ang_vel = body_ang_vel_w[i, 0, :]

        env.set_state(
            dof_positions=joint_pos[i],
            dof_velocities=joint_vel[i],
            root_quat=root_quat,
            root_pos=root_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
        )
        obs = env._create_observation_backward()
        obs_list.append(obs)

        if (i + 1) % 200 == 0:
            print(f"  Built observations: {i + 1}/{n_frames}")

    # Stack observations: each key (N,) -> (N, dim)
    next_obs = {}
    for k in obs_list[0].keys():
        next_obs[k] = torch.cat([obs_list[i][k] for i in range(len(obs_list))], dim=0)

    print(f"  Observation shapes: { {k: v.shape for k, v in next_obs.items()} }")
    return next_obs


def convert_single(input_path, output_path, model, env, start=0, end=-1):
    """Convert a single .npz motion file to tracking z .pkl."""
    # Load motion data
    print(f"Loading motion data from {input_path} ...")
    motion = load_motion_data(input_path, start, end)
    n_frames = motion["joint_pos"].shape[0]
    print(f"  {n_frames} frames at {motion['fps']} fps ({n_frames / motion['fps']:.1f}s)")

    # Build observations via MuJoCo env
    print("Building observations via MuJoCo env ...")
    next_obs = build_observations(motion, env)

    # Run tracking inference (backward_map + sliding window averaging + projection)
    print("Running tracking inference ...")
    with torch.no_grad():
        z = model.tracking_inference(next_obs)

    z_np = z.cpu().numpy().astype(np.float32)
    print(f"  z shape: {z_np.shape}, norm: {np.linalg.norm(z_np[0]):.2f}")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    joblib.dump(z_np, output_path)
    print(f"Saved tracking z to {output_path}")
    print(f"  Shape: {z_np.shape} ({z_np.shape[0]} frames x {z_np.shape[1]}D)")


def main():
    parser = argparse.ArgumentParser(description="Convert motion .npz to tracking z .pkl")
    parser.add_argument("--input", "-i", default=None, help="Input .npz motion file")
    parser.add_argument("--output", "-o", default=None, help="Output .pkl file for tracking z vectors")
    parser.add_argument("--input_dir", default=None, help="Input directory for batch conversion (all .npz files)")
    parser.add_argument("--output_dir", default=None, help="Output directory for batch conversion")
    parser.add_argument("--model_path", default=None,
                        help="Path to BFM-Zero_inf model checkpoint (default: ../BFM-Zero_inf/model/checkpoint/model)")
    parser.add_argument("--device", default="cpu", help="Device for inference (default: cpu)")
    parser.add_argument("--start", type=int, default=0, help="Start frame index (default: 0)")
    parser.add_argument("--end", type=int, default=-1, help="End frame index (default: -1 = all)")
    parser.add_argument("--seq_length", type=int, default=None,
                        help="Sliding window size for tracking averaging (default: use model config, typically 8)")
    args = parser.parse_args()

    # Validate arguments
    batch_mode = args.input_dir is not None or args.output_dir is not None
    single_mode = args.input is not None or args.output is not None
    if batch_mode and single_mode:
        parser.error("Cannot use --input/--output together with --input_dir/--output_dir")
    if batch_mode and (args.input_dir is None or args.output_dir is None):
        parser.error("Batch mode requires both --input_dir and --output_dir")
    if not batch_mode and (args.input is None or args.output is None):
        parser.error("Single mode requires both --input and --output")

    # Resolve model path
    if args.model_path is None:
        args.model_path = os.path.join(INF_REPO, "model", "checkpoint", "model")

    robot_xml = os.path.join(INF_REPO, "bfm_zero_inference_code", "g1_for_reward_inference.xml")

    # Load model
    print(f"Loading model from {args.model_path} ...")
    model = FBcprAuxModel.load(args.model_path, device=args.device)
    model.eval()
    z_dim = model.cfg.archi.z_dim
    seq_length = args.seq_length if args.seq_length is not None else model.cfg.seq_length
    print(f"  z_dim={z_dim}, seq_length={seq_length}")

    # Create MuJoCo env (shared across all conversions)
    env = MuJoCoBFMZeroEnv(
        robot_xml=robot_xml,
        kp_gains=KP_GAINS,
        kd_gains=KD_GAINS,
        default_joint_pos=DEFAULT_JOINT_POS,
        action_scales=ACTION_SCALES,
        action_rescale=ACTION_RESCALE,
    )

    if batch_mode:
        # Collect all .npz files preserving directory structure
        input_dir = os.path.abspath(args.input_dir)
        output_dir = os.path.abspath(args.output_dir)
        npz_files = []
        for root, _, files in os.walk(input_dir):
            for f in sorted(files):
                if f.endswith(".npz"):
                    npz_files.append(os.path.join(root, f))

        if not npz_files:
            print(f"No .npz files found in {input_dir}")
            return

        print(f"\nBatch mode: found {len(npz_files)} .npz files in {input_dir}\n")

        for idx, npz_path in enumerate(npz_files, 1):
            rel_path = os.path.relpath(npz_path, input_dir)
            stem = os.path.splitext(rel_path)[0]
            out_path = os.path.join(output_dir, stem + "_z.pkl")

            print(f"\n{'='*60}")
            print(f"[{idx}/{len(npz_files)}] {rel_path}")
            print(f"{'='*60}")

            try:
                convert_single(npz_path, out_path, model, env, args.start, args.end)
            except Exception as e:
                print(f"  ERROR: Failed to convert {rel_path}: {e}")
                continue

        print(f"\nBatch conversion complete. Output directory: {output_dir}")
    else:
        convert_single(args.input, args.output, model, env, args.start, args.end)
        print(f"\nTo use: update config/exp/tracking/*.yaml with ctx_path pointing to this file.")


if __name__ == "__main__":
    main()
