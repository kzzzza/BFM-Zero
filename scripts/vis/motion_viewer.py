#!/usr/bin/env python3
"""Visualize raw motion data (.npz from data/motions/) in MuJoCo.

Subscribes to the ``MotionFrameMessage`` stream produced by
``scripts/replay_motion_zmq.py`` and drives a passive MuJoCo viewer. By default
the publisher subprocess is spawned automatically, so a single command is
enough to view a motion file.

Keyboard (focus the MuJoCo window):
    Space   pause / resume
    N       single-step one frame while paused
    Q       quit

Examples
--------
Single pass, default speed:

    python scripts/vis/motion_viewer.py \\
        --motion data/motions/lafan/jumps1_subject1.npz

Loop forever at half speed:

    python scripts/vis/motion_viewer.py \\
        --motion data/motions/amass/Form_1_stageii.npz --loop --speed 0.5

Connect to an externally running replay_motion_zmq.py instead of spawning one:

    python scripts/vis/motion_viewer.py --motion <same path> --no-spawn-publisher
"""

from __future__ import annotations

import argparse
import atexit
import os
import sched
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
import zmq

import mujoco
import mujoco.viewer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.motion_loader import load_motion  # noqa: E402
from utils.common import PORTS, MotionFrameMessage  # noqa: E402
from utils.strings import unitree_joint_names  # noqa: E402


# In this project Isaac joint order == Unitree joint order (see
# config/policy/motivo_newG1.yaml:isaac_joint_names, which is identical to
# utils.strings.unitree_joint_names). motion_loader.load_motion() already
# returns joint_pos in this order, and the MuJoCo XML declares joints in the
# same order — so we map by name directly.
ISAAC_JOINT_NAMES = list(unitree_joint_names)
DEFAULT_SCENE_CONFIG = REPO_ROOT / "config/scene/g1_29dof.yaml"
DEFAULT_SCENE_XML = REPO_ROOT / "data/robots/g1/scene_29dof_freebase.xml"


def _load_scene_xml(scene_config_path: Path) -> Path:
    """Resolve the MuJoCo scene XML path from a scene yaml (or fall back)."""
    if not scene_config_path.is_file():
        print(f"[viewer] scene_config {scene_config_path} not found, using fallback {DEFAULT_SCENE_XML}")
        return DEFAULT_SCENE_XML
    with open(scene_config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    raw = cfg.get("ROBOT_SCENE", str(DEFAULT_SCENE_XML))
    p = Path(raw)
    if not p.is_absolute():
        p = (REPO_ROOT / raw).resolve()
    return p


def _build_isaac_to_mujoco_mapping(model: mujoco.MjModel):
    """Map Isaac-order joint indices to MuJoCo qpos addresses by joint name."""
    src_ids = []
    tgt_qpos_adrs = []
    for isaac_idx, joint_name in enumerate(ISAAC_JOINT_NAMES):
        mj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if mj_jid == -1:
            raise RuntimeError(f"joint '{joint_name}' not found in MuJoCo model")
        src_ids.append(isaac_idx)
        tgt_qpos_adrs.append(model.jnt_qposadr[mj_jid])
    src_ids = np.asarray(src_ids, dtype=np.int64)
    tgt_qpos_adrs = np.asarray(tgt_qpos_adrs, dtype=np.int64)

    fb_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if fb_jid == -1:
        raise RuntimeError("floating_base_joint not found in MuJoCo model")
    pelvis_qpos_adr = int(model.jnt_qposadr[fb_jid])

    pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_body_id == -1:
        raise RuntimeError("pelvis body not found in MuJoCo model")

    return src_ids, tgt_qpos_adrs, pelvis_qpos_adr, pelvis_body_id


def _spawn_publisher(args, fps: int) -> subprocess.Popen:
    rate = float(fps) * float(args.speed)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "replay_motion_zmq.py"),
        "--motion", args.motion,
        "--port", str(PORTS["motion_frame"]),
        "--rate", f"{rate:.6f}",
        "--start", str(args.start),
        "--end", str(args.end),
    ]
    if args.loop:
        cmd.append("--loop")
    print(f"[viewer] spawning publisher: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))

    def _cleanup():
        if proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except ProcessLookupError:
                pass

    atexit.register(_cleanup)
    return proc


class MotionViewer:
    def __init__(self, scene_xml: Path, fps: int, pub_proc: subprocess.Popen | None):
        self.fps = fps
        self.update_dt = 1.0 / float(fps)
        self.pub_proc = pub_proc

        print(f"[viewer] loading MuJoCo model: {scene_xml}")
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)

        (
            self.src_ids,
            self.tgt_qpos_adrs,
            self.pelvis_qpos_adr,
            self.pelvis_body_id,
        ) = _build_isaac_to_mujoco_mapping(self.model)

        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        # CONFLATE keeps only the latest message — pause→resume won't burst.
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.setsockopt(zmq.RCVTIMEO, 0)
        self.sock.connect(f"tcp://localhost:{PORTS['motion_frame']}")
        print(f"[viewer] subscribed to tcp://localhost:{PORTS['motion_frame']}")

        self.paused = False
        self.step_once = False
        self.quit_requested = False
        self.last_frame_idx = -1
        self.last_end_logged = False

        self.viewer = mujoco.viewer.launch_passive(
            self.model,
            self.data,
            show_left_ui=False,
            show_right_ui=False,
            key_callback=self._on_key,
        )
        self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.viewer.cam.trackbodyid = self.pelvis_body_id

    def _on_key(self, keycode: int) -> None:
        # MuJoCo passes ASCII codes; letters arrive as upper-case.
        if keycode == ord(' '):
            self.paused = not self.paused
            print(f"[viewer] {'PAUSED' if self.paused else 'RESUMED'}")
        elif keycode == ord('N'):
            self.step_once = True
            print("[viewer] single step")
        elif keycode == ord('Q'):
            self.quit_requested = True
            print("[viewer] quit requested")

    def _drain_latest_frame(self):
        """Return the most recent MotionFrameMessage available, or None."""
        msg = None
        while True:
            try:
                buf = self.sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                msg = MotionFrameMessage.from_bytes(buf)
            except ValueError as e:
                print(f"[viewer] bad frame: {e}")
                continue
        return msg

    def _apply_frame(self, msg: MotionFrameMessage) -> None:
        # Floating base: [x, y, z, qw, qx, qy, qz]
        adr = self.pelvis_qpos_adr
        self.data.qpos[adr: adr + 3] = msg.root_pos
        self.data.qpos[adr + 3: adr + 7] = msg.root_quat
        # 29 joints from Isaac order → MuJoCo qpos addresses
        self.data.qpos[self.tgt_qpos_adrs] = msg.joint_pos[self.src_ids]
        mujoco.mj_forward(self.model, self.data)

        if msg.frame_idx != self.last_frame_idx:
            self.last_frame_idx = msg.frame_idx
            if msg.flags & MotionFrameMessage.FLAG_END and not self.last_end_logged:
                print(f"[viewer] received END flag at frame_idx={msg.frame_idx}")
                self.last_end_logged = True

    def run(self) -> None:
        print(f"[viewer] starting update loop at {self.fps} Hz")
        scheduler = sched.scheduler(time.perf_counter, time.sleep)
        next_t = time.perf_counter()

        try:
            while self.viewer.is_running() and not self.quit_requested:
                scheduler.enterabs(next_t, 1, self._tick, ())
                scheduler.run()
                next_t += self.update_dt
        except KeyboardInterrupt:
            print("[viewer] interrupted")
        finally:
            self._shutdown()

    def _tick(self) -> None:
        if (not self.paused) or self.step_once:
            msg = self._drain_latest_frame()
            if msg is not None:
                self._apply_frame(msg)
                if self.step_once:
                    self.step_once = False
                    self.paused = True
        self.viewer.sync()

    def _shutdown(self) -> None:
        try:
            self.viewer.close()
        except Exception:
            pass
        try:
            self.sock.close(linger=0)
        except Exception:
            pass
        if self.pub_proc is not None and self.pub_proc.poll() is None:
            self.pub_proc.terminate()
            try:
                self.pub_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.pub_proc.kill()
        print("[viewer] shutdown complete")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--motion", required=True, help="path to motion .npz")
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier (e.g. 0.5, 2.0)")
    parser.add_argument("--loop", action="store_true", help="loop motion playback")
    parser.add_argument("--start", type=int, default=0, help="start frame index")
    parser.add_argument("--end", type=int, default=-1, help="end frame index (-1 = end)")
    parser.add_argument(
        "--scene_config",
        type=str,
        default=str(DEFAULT_SCENE_CONFIG),
        help="path to scene yaml (reads ROBOT_SCENE field)",
    )
    parser.add_argument(
        "--no-spawn-publisher",
        action="store_true",
        help="don't spawn replay_motion_zmq.py; connect to an externally running publisher",
    )
    args = parser.parse_args()

    motion_path = Path(args.motion)
    if not motion_path.is_absolute():
        motion_path = (Path.cwd() / motion_path).resolve()
    if not motion_path.is_file():
        print(f"[viewer] motion file not found: {motion_path}", file=sys.stderr)
        sys.exit(1)
    args.motion = str(motion_path)

    # Peek the motion just to learn fps + frame count; the actual stream comes via ZMQ.
    motion = load_motion(args.motion, start=args.start, end=args.end)
    fps = int(motion["fps"])
    n_frames = motion["joint_pos"].shape[0]
    print(
        f"[viewer] motion={args.motion}\n"
        f"[viewer] format={motion['format']}, fps={fps}, frames={n_frames}, "
        f"speed={args.speed}, loop={args.loop}"
    )

    scene_xml = _load_scene_xml(Path(args.scene_config))
    if not scene_xml.is_file():
        print(f"[viewer] scene XML not found: {scene_xml}", file=sys.stderr)
        sys.exit(1)

    pub_proc = None
    if not args.no_spawn_publisher:
        pub_proc = _spawn_publisher(args, fps)
        # Give the publisher a moment to bind and start broadcasting.
        time.sleep(0.7)

    def _sigint_handler(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _sigint_handler)

    viewer = MotionViewer(scene_xml=scene_xml, fps=fps, pub_proc=pub_proc)
    viewer.run()


if __name__ == "__main__":
    main()
