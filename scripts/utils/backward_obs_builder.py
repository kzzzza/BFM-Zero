"""Per-frame backward observation builder for the BFM-Zero ONNX B-network.

Replicates ``BFM-Zero_inf/env.py:MuJoCoBFMZeroEnv`` for the minimal subset
needed to build the 64-D ``state`` and 463-D ``privileged_state`` that the
exported backward ONNX expects as ``b_obs = concat([state, privileged_state])``.

No PyTorch dependency, all numpy + MuJoCo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from utils.math import (
    calc_angular_velocity,
    calc_heading_quat_inv,
    quat_mul,
    quat_rotate_numpy,
    quat_to_tan_norm,
)


NUM_DOF = 29
STATE_DIM = 64
PRIV_DIM = 463


class BackwardObsBuilder:
    """Mirrors ``MuJoCoBFMZeroEnv._create_observation_backward`` over a MuJoCo model.

    Body filter — matches the inference repo verbatim:
      - exclude bodies starting with ``dummy``, ending with ``hand``, starting with ``world``
      - move ``head_link`` to the end of the list
    The reference XML (``g1_for_backward_obs.xml``) yields 31 bodies after filter.
    """

    def __init__(
        self,
        xml_path: str,
        default_joint_pos: np.ndarray,
        dt: float = 0.02,
    ):
        self.xml_path = str(xml_path)
        self.dt = float(dt)
        self.default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32).copy()
        if self.default_joint_pos.shape != (NUM_DOF,):
            raise ValueError(
                f"default_joint_pos must have shape ({NUM_DOF},), got {self.default_joint_pos.shape}"
            )

        self.mjm = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mjd = mujoco.MjData(self.mjm)

        self.valid_body_indices, self.body_names = self._collect_valid_bodies()
        self.num_bodies = len(self.valid_body_indices)

        expected_priv_dim = 1 + (self.num_bodies - 1) * 3 + self.num_bodies * 6 + self.num_bodies * 3 + self.num_bodies * 3
        if expected_priv_dim != PRIV_DIM:
            raise ValueError(
                f"XML {self.xml_path} yields {self.num_bodies} valid bodies → priv_dim={expected_priv_dim}, "
                f"expected {PRIV_DIM}. Check XML body list against BFM-Zero_inf."
            )

        self.body_pos_prev: np.ndarray | None = None
        self.body_quat_prev: np.ndarray | None = None

    def _collect_valid_bodies(self) -> Tuple[np.ndarray, list[str]]:
        valid_indices: list[int] = []
        names: list[str] = []
        head_idx: int | None = None

        for i in range(self.mjm.nbody):
            name = mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_BODY, i)
            if name is None:
                valid_indices.append(i)
                names.append(f"body_{i}")
                continue
            if name.startswith("dummy") or name.endswith("hand") or name.startswith("world"):
                continue
            if name == "head_link":
                head_idx = i
                continue
            valid_indices.append(i)
            names.append(name)

        if head_idx is not None:
            valid_indices.append(head_idx)
            names.append("head_link")

        return np.asarray(valid_indices, dtype=np.int64), names

    def reset(self) -> None:
        """Clear the previous-frame cache used for body velocity finite difference."""
        self.body_pos_prev = None
        self.body_quat_prev = None

    def set_state(
        self,
        dof_positions: np.ndarray,
        dof_velocities: np.ndarray | None,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        root_lin_vel_w: np.ndarray,
        root_ang_vel_w: np.ndarray,
    ) -> None:
        """Write a single frame's state into MuJoCo and run forward kinematics.

        Args:
            dof_positions:  (29,) joint angles in Isaac order.
            dof_velocities: (29,) or None.
            root_pos:       (3,) world position.
            root_quat:      (4,) world quaternion [w, x, y, z].
            root_lin_vel_w: (3,) world-frame root linear velocity.
            root_ang_vel_w: (3,) world-frame root angular velocity.
        """
        rot = Rotation.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
        local_root_ang_vel = rot.inv().apply(root_ang_vel_w)

        self.mjd.qpos[0:3] = root_pos
        self.mjd.qpos[3:7] = root_quat
        self.mjd.qpos[7 : 7 + NUM_DOF] = dof_positions
        self.mjd.qvel[0:3] = root_lin_vel_w
        self.mjd.qvel[3:6] = local_root_ang_vel
        if dof_velocities is None:
            self.mjd.qvel[6 : 6 + NUM_DOF] = 0.0
        else:
            self.mjd.qvel[6 : 6 + NUM_DOF] = dof_velocities

        mujoco.mj_forward(self.mjm, self.mjd)

    def get_state_obs(self) -> np.ndarray:
        """Return the 64-D ``state`` vector.

        Layout matches ``_create_observation_backward``:
            [dof_pos - default(29), dof_vel(29), projected_gravity(3), ang_vel(3)]
        ``ang_vel`` is ``R · qvel[3:6]`` (i.e. the world-frame root angular
        velocity reconstructed from the body-frame qvel slot).
        """
        dof_pos = self.mjd.qpos[7 : 7 + NUM_DOF].copy() - self.default_joint_pos
        dof_vel = self.mjd.qvel[6 : 6 + NUM_DOF].copy()

        root_quat = self.mjd.qpos[3:7]
        rot = Rotation.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
        projected_gravity = rot.inv().apply(np.array([0.0, 0.0, -1.0]))
        ang_vel = rot.apply(self.mjd.qvel[3:6].copy())

        state = np.concatenate([dof_pos, dof_vel, projected_gravity, ang_vel]).astype(np.float32)
        assert state.shape == (STATE_DIM,), f"state shape {state.shape} != ({STATE_DIM},)"
        return state

    def get_privileged_state(self) -> np.ndarray:
        """Return the 463-D ``privileged_state`` vector.

        Mirrors ``get_privileged_state`` in the inference repo:
            [root_height(1), local_body_pos[1:](90), local_body_rot_6d(186),
             local_body_lin_vel(93), local_body_ang_vel(93)]  = 463

        Local frame = heading-rotated root frame (yaw of root inverted).
        Body velocities come from a finite difference with the previous call;
        the first call after ``reset()`` returns zeros for body velocities.
        """
        body_pos = self.mjd.xpos[self.valid_body_indices, :].copy()    # (N, 3)
        body_quat = self.mjd.xquat[self.valid_body_indices, :].copy()  # (N, 4) [w, x, y, z]

        if self.body_pos_prev is None:
            body_lin_vel = np.zeros_like(body_pos)
            body_ang_vel = np.zeros_like(body_pos)
        else:
            body_lin_vel = (body_pos - self.body_pos_prev) / self.dt
            body_ang_vel = calc_angular_velocity(body_quat, self.body_quat_prev, self.dt)
        self.body_pos_prev = body_pos
        self.body_quat_prev = body_quat

        root_pos = body_pos[0:1, :]   # (1, 3)
        root_quat = body_quat[0:1, :] # (1, 4)

        heading_rot_inv = calc_heading_quat_inv(root_quat)              # (1, 4)
        heading_rot_inv_n = np.broadcast_to(heading_rot_inv, body_quat.shape).copy()  # (N, 4)

        local_body_pos = quat_rotate_numpy(heading_rot_inv_n, body_pos - root_pos)              # (N, 3)
        local_body_rot = quat_mul(heading_rot_inv_n, body_quat)                                  # (N, 4)
        local_body_rot_6d = quat_to_tan_norm(local_body_rot)                                     # (N, 6)
        local_body_lin_vel = quat_rotate_numpy(heading_rot_inv_n, body_lin_vel.astype(np.float32))  # (N, 3)
        local_body_ang_vel = quat_rotate_numpy(heading_rot_inv_n, body_ang_vel.astype(np.float32))  # (N, 3)

        root_h = root_pos[:, 2:3].reshape(-1)  # (1,)
        priv = np.concatenate([
            root_h,
            local_body_pos[1:, :].reshape(-1),   # drop root (it's zero anyway)
            local_body_rot_6d.reshape(-1),
            local_body_lin_vel.reshape(-1),
            local_body_ang_vel.reshape(-1),
        ]).astype(np.float32)

        assert priv.shape == (PRIV_DIM,), f"priv shape {priv.shape} != ({PRIV_DIM},)"
        return priv

    def build_b_obs(self) -> np.ndarray:
        """Convenience: returns the 527-D concatenated b_obs (state + privileged_state)."""
        return np.concatenate([self.get_state_obs(), self.get_privileged_state()]).astype(np.float32)
