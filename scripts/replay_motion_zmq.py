#!/usr/bin/env python3
"""Replay an .npz motion file as a ZMQ ``MotionFrameMessage`` PUB stream.

Used to drive ``tracking_online`` with ``source: zmq``. Publishes at the
requested rate (default 50 Hz, matching the policy). Each frame includes a
monotonically increasing ``frame_idx``; on the final frame (when ``--loop`` is
NOT set), an END flag is set so the policy stops cleanly.

Examples
--------
Single pass at 50 Hz:

    python scripts/replay_motion_zmq.py \\
        --motion data/motions/amass/Form_1_stageii.npz --rate 50

Loop forever (no END flag):

    python scripts/replay_motion_zmq.py --motion data/motions/amass/Form_1_stageii.npz --loop
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import zmq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.motion_loader import load_motion  # noqa: E402
from utils.common import PORTS, MotionFrameMessage  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion", required=True, help="path to motion .npz")
    p.add_argument("--port", type=int, default=PORTS["motion_frame"])
    p.add_argument("--rate", type=float, default=50.0, help="publish rate (Hz)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=-1)
    p.add_argument("--loop", action="store_true", help="loop instead of sending END flag")
    args = p.parse_args()

    motion = load_motion(args.motion, start=args.start, end=args.end)
    n = motion["joint_pos"].shape[0]
    print(f"[replay] loaded {n} frames @ {motion['fps']} fps from {args.motion} (format={motion['format']})")

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[replay] publishing on tcp://*:{args.port} at {args.rate} Hz")
    # Give subscribers time to attach before the first message.
    time.sleep(0.5)

    dt = 1.0 / args.rate
    next_t = time.perf_counter()
    frame_idx = 0
    try:
        while True:
            for i in range(n):
                is_last = (i == n - 1) and not args.loop
                msg = MotionFrameMessage(
                    frame_idx=frame_idx,
                    joint_pos=motion["joint_pos"][i],
                    joint_vel=motion["joint_vel"][i],
                    root_pos=motion["body_pos_w"][i, 0, :],
                    root_quat=motion["body_quat_w"][i, 0, :],
                    root_lin_vel_w=motion["body_lin_vel_w"][i, 0, :],
                    root_ang_vel_w=motion["body_ang_vel_w"][i, 0, :],
                    flags=MotionFrameMessage.FLAG_END if is_last else 0,
                )
                sock.send(msg.to_bytes())
                frame_idx += 1

                next_t += dt
                sleep_for = next_t - time.perf_counter()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    # Falling behind — reset schedule to avoid burst-catch-up.
                    next_t = time.perf_counter()

                if frame_idx % 200 == 0:
                    print(f"[replay] sent frame_idx={frame_idx} (motion i={i}/{n-1})")
            if not args.loop:
                break
    except KeyboardInterrupt:
        print("\n[replay] interrupted")
    finally:
        sock.close(linger=200)
        ctx.term()
        print("[replay] closed")


if __name__ == "__main__":
    main()
