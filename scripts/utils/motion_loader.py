"""Load motion .npz files (data/ or example_motion/ format) into a unified dict.

Replicates the loading half of ``scripts/motion_to_tracking_z.py`` without
pulling in the BFM-Zero_inf PyTorch dependency. Output is a plain dict of
numpy arrays ready to be fed frame-by-frame into ``BackwardObsBuilder``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from utils.math import calc_angular_velocity


# data/ format (GMR retarget) uses BMimic joint order; map each Isaac index
# to the BMimic source index it lives at.
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
BMIMIC_TO_ISAAC_JOINT = [0] * 29
for _bmimic_idx, _isaac_idx in enumerate(G1_JOINT_MAPPING):
    BMIMIC_TO_ISAAC_JOINT[_isaac_idx] = _bmimic_idx


def _detect_format(data) -> str:
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
    """Frame 0 velocities are zero; the rest is plain forward differences."""
    n_frames = joint_pos.shape[0]
    dt = 1.0 / fps

    joint_vel = np.zeros_like(joint_pos, dtype=np.float32)
    body_lin_vel = np.zeros_like(body_pos, dtype=np.float32)
    body_ang_vel = np.zeros_like(body_pos, dtype=np.float32)

    joint_vel[1:] = (joint_pos[1:] - joint_pos[:-1]) / dt
    body_lin_vel[1:] = (body_pos[1:] - body_pos[:-1]) / dt

    for i in range(1, n_frames):
        body_ang_vel[i] = calc_angular_velocity(body_quat[i], body_quat[i - 1], dt)

    return joint_vel, body_lin_vel, body_ang_vel


def load_motion(npz_path: str | Path, start: int = 0, end: int = -1) -> Dict[str, np.ndarray]:
    """Load a motion .npz, auto-detecting format. Returns dict in Isaac ordering.

    Output keys: fps (int), joint_pos, joint_vel, body_pos_w, body_quat_w,
    body_lin_vel_w, body_ang_vel_w. All arrays are float32, sliced to [start:end].
    """
    data = np.load(npz_path, allow_pickle=True)
    fmt = _detect_format(data)

    fps_raw = data["fps"]
    fps = int(fps_raw.item() if fps_raw.ndim > 0 else fps_raw)

    if fmt == "data":
        joint_pos = data["joint_pos"][:, BMIMIC_TO_ISAAC_JOINT].astype(np.float32)
        joint_vel = data["joint_vel"][:, BMIMIC_TO_ISAAC_JOINT].astype(np.float32)
        body_pos_w = data["body_pos_w"].astype(np.float32)
        body_quat_w = data["body_quat_w"].astype(np.float32)
        body_lin_vel_w = data["body_lin_vel_w"].astype(np.float32)
        body_ang_vel_w = data["body_ang_vel_w"].astype(np.float32)
    else:
        joint_pos = data["dof_positions"].astype(np.float32)
        body_pos_w = data["body_positions"].astype(np.float32)
        body_quat_w = data["body_rotations"].astype(np.float32)
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
        "format": fmt,
        "joint_pos": joint_pos[start:end],
        "joint_vel": joint_vel[start:end],
        "body_pos_w": body_pos_w[start:end],
        "body_quat_w": body_quat_w[start:end],
        "body_lin_vel_w": body_lin_vel_w[start:end],
        "body_ang_vel_w": body_ang_vel_w[start:end],
    }
