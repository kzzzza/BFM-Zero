"""Online B-network z provider for ``tracking_online`` task type.

Wraps ``BackwardObsBuilder`` + the exported backward ONNX session and maintains
a frame-indexed cache of raw z vectors. Each policy step computes z only for
the new frontier frame and returns the (forward-looking) sliding-window mean
projected back to ``||z|| = sqrt(256)`` — numerically equivalent to the offline
pipeline in ``scripts/motion_to_z_onnx.py``.

Finite-diff invariant
---------------------
``BackwardObsBuilder`` derives body lin/ang velocities from the previous
``set_state`` call. ``push_frame(k)`` therefore REQUIRES ``k == last_frame + 1``
(or no prior call after ``reset()``). Out-of-order or skipped frames trigger an
automatic ``reset()`` + warning to keep the state coherent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import onnxruntime as ort
from loguru import logger

from scripts.utils.backward_obs_builder import BackwardObsBuilder
from scripts.motion_to_z_onnx import DEFAULT_JOINT_POS, _project_z


Z_DIM = 256


class OnlineZProvider:
    def __init__(
        self,
        scene_xml: str,
        onnx_path: str,
        seq_length: int,
        dt: float,
        default_joint_pos: np.ndarray = DEFAULT_JOINT_POS,
        providers: list[str] | None = None,
    ):
        if seq_length < 1:
            raise ValueError(f"seq_length must be >= 1, got {seq_length}")
        if not Path(scene_xml).exists():
            raise FileNotFoundError(f"scene_xml not found: {scene_xml}")
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"backward onnx not found: {onnx_path}")

        self.seq_length = int(seq_length)
        self.dt = float(dt)

        self.builder = BackwardObsBuilder(
            scene_xml,
            default_joint_pos=default_joint_pos,
            dt=self.dt,
        )

        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=providers or ["CPUExecutionProvider"],
        )
        self.in_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name

        self.raw_z_cache: Dict[int, np.ndarray] = {}
        self.builder_last_frame: Optional[int] = None

    def reset(self) -> None:
        self.builder.reset()
        self.raw_z_cache.clear()
        self.builder_last_frame = None

    def next_frame_to_seed(self, default_start: int = 0) -> int:
        """Returns the next frame index the builder is ready to ingest.

        After ``reset()`` this is ``default_start``; otherwise it is
        ``builder_last_frame + 1``.
        """
        if self.builder_last_frame is None:
            return default_start
        return self.builder_last_frame + 1

    def push_frame(
        self,
        frame_idx: int,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        root_pos: np.ndarray,
        root_quat: np.ndarray,
        root_lin_vel_w: np.ndarray,
        root_ang_vel_w: np.ndarray,
    ) -> np.ndarray:
        """Advance the builder by exactly one frame and cache its raw z.

        If ``frame_idx`` is not contiguous with ``builder_last_frame``, the
        builder is reset and this frame becomes the new starting point (its
        body velocities will be zero — matching offline behavior on
        ``env.reset()``).
        """
        if self.builder_last_frame is not None and frame_idx != self.builder_last_frame + 1:
            logger.warning(
                f"OnlineZProvider: discontinuity (last={self.builder_last_frame}, "
                f"new={frame_idx}); auto-resetting builder."
            )
            self.reset()

        self.builder.set_state(
            dof_positions=joint_pos,
            dof_velocities=joint_vel,
            root_pos=root_pos,
            root_quat=root_quat,
            root_lin_vel_w=root_lin_vel_w,
            root_ang_vel_w=root_ang_vel_w,
        )
        b_obs = self.builder.build_b_obs().reshape(1, -1).astype(np.float32)
        z_raw = self.session.run([self.out_name], {self.in_name: b_obs})[0].reshape(-1).astype(np.float32)

        self.raw_z_cache[frame_idx] = z_raw
        self.builder_last_frame = frame_idx
        return z_raw

    def push_frame_from_motion(self, motion: Dict[str, np.ndarray], frame_idx: int) -> np.ndarray:
        """Convenience for file-source mode: pull frame ``frame_idx`` from the
        loaded motion dict and call ``push_frame``. The motion dict's index 0
        corresponds to the slice start passed to ``load_motion``.
        """
        return self.push_frame(
            frame_idx=frame_idx,
            joint_pos=motion["joint_pos"][frame_idx],
            joint_vel=motion["joint_vel"][frame_idx],
            root_pos=motion["body_pos_w"][frame_idx, 0, :],
            root_quat=motion["body_quat_w"][frame_idx, 0, :],
            root_lin_vel_w=motion["body_lin_vel_w"][frame_idx, 0, :],
            root_ang_vel_w=motion["body_ang_vel_w"][frame_idx, 0, :],
        )

    def get_z(self, t: int, motion_length: Optional[int] = None) -> np.ndarray:
        """Return projected sliding-window-mean z for frame ``t``.

        Window is ``[t, min(t + seq_length, motion_length or cached_upper))``.
        Caller must have already pushed every frame in that range.
        """
        if motion_length is not None:
            window_end = min(t + self.seq_length, motion_length)
        else:
            cached_upper = (self.builder_last_frame + 1) if self.builder_last_frame is not None else t
            window_end = min(t + self.seq_length, cached_upper)

        if window_end <= t:
            raise RuntimeError(
                f"OnlineZProvider.get_z: empty window at t={t} "
                f"(motion_length={motion_length}, last_frame={self.builder_last_frame})"
            )

        zs = []
        for k in range(t, window_end):
            if k not in self.raw_z_cache:
                raise RuntimeError(
                    f"OnlineZProvider.get_z: missing cached z for frame {k}; "
                    f"call push_frame for the full window [{t}, {window_end}) first."
                )
            zs.append(self.raw_z_cache[k])

        z_mean = np.mean(np.stack(zs, axis=0), axis=0)
        return _project_z(z_mean).astype(np.float32)
