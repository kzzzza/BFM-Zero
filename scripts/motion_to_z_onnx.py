#!/usr/bin/env python3
"""Generate latent z (.pkl) from motion .npz using the exported backward ONNX.

Standalone replacement for the relevant parts of BFM-Zero_inf: no PyTorch /
no inference-repo imports. Supports two task modes today — ``tracking`` and
``goal`` — both producing pkl files compatible with the existing
``model/{tracking_inference,goal_inference}/`` layout.

Examples
--------
Single tracking motion:

    python scripts/motion_to_z_onnx.py tracking \\
        --input  ../BFM-Zero_inf/example_motion/walk2_subject3_50_jpos.npz \\
        --output model/tracking_inference/walk2_subject3_z.pkl

Batch tracking over a directory:

    python scripts/motion_to_z_onnx.py tracking \\
        --input_dir  ../BFM-Zero_inf/example_motion/ \\
        --output_dir model/tracking_inference/

Goal dict from a YAML clip list:

    python scripts/motion_to_z_onnx.py goal \\
        --clips_config config/z_inference/goal_clips.yaml

Verify against an existing pkl:

    python scripts/motion_to_z_onnx.py tracking \\
        --input  ../BFM-Zero_inf/example_motion/walk2_subject3_50_jpos.npz \\
        --output /tmp/new_walk2_z.pkl \\
        --reference_pkl model/tracking_inference/walk2_subject3_z.pkl
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import onnxruntime as ort
import yaml

# Make the deploy repo importable when running this script directly.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.backward_obs_builder import (  # noqa: E402
    BackwardObsBuilder,
    PRIV_DIM,
    STATE_DIM,
)
from scripts.utils.motion_loader import load_motion  # noqa: E402


DEFAULT_MODEL_PATH = "model/exported/FBcprAuxModel_backward_test.onnx"
DEFAULT_SCENE_XML = "data/robots/g1/g1_for_backward_obs.xml"
Z_DIM = 256

# Mirrors common.DEFAULT_JOINT_POS in BFM-Zero_inf and the default_joint_pos
# block of config/policy/motivo_newG1.yaml. Order is Isaac (policy_joint_names).
DEFAULT_JOINT_POS = np.zeros(29, dtype=np.float32)
DEFAULT_JOINT_POS[0] = -0.1   # left_hip_pitch
DEFAULT_JOINT_POS[3] = 0.3    # left_knee
DEFAULT_JOINT_POS[4] = -0.2   # left_ankle_pitch
DEFAULT_JOINT_POS[6] = -0.1   # right_hip_pitch
DEFAULT_JOINT_POS[9] = 0.3    # right_knee
DEFAULT_JOINT_POS[10] = -0.2  # right_ankle_pitch


def _make_session(model_path: str) -> tuple[ort.InferenceSession, str, str]:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    expected_dim = STATE_DIM + PRIV_DIM
    shape = list(inp.shape)
    last = shape[-1] if shape else None
    if last not in (expected_dim, None):
        raise RuntimeError(
            f"ONNX input {inp.name} has shape {shape}; expected last dim {expected_dim} "
            f"(state={STATE_DIM} + priv={PRIV_DIM})."
        )
    print(f"[ok] ONNX inputs={inp.name}{inp.shape}  outputs={out.name}{out.shape}")
    return sess, inp.name, out.name


def _project_z(z: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(z, axis=-1, keepdims=True)
    norm = np.where(norm < 1e-12, 1.0, norm)
    return np.sqrt(float(Z_DIM)) * z / norm


def _build_env(scene_xml: str, dt: float) -> BackwardObsBuilder:
    return BackwardObsBuilder(scene_xml, default_joint_pos=DEFAULT_JOINT_POS, dt=dt)


def _set_frame(env: BackwardObsBuilder, motion: Dict[str, np.ndarray], i: int) -> None:
    env.set_state(
        dof_positions=motion["joint_pos"][i],
        dof_velocities=motion["joint_vel"][i],
        root_pos=motion["body_pos_w"][i, 0, :],
        root_quat=motion["body_quat_w"][i, 0, :],
        root_lin_vel_w=motion["body_lin_vel_w"][i, 0, :],
        root_ang_vel_w=motion["body_ang_vel_w"][i, 0, :],
    )


def _run_backward(sess: ort.InferenceSession, in_name: str, out_name: str, b_obs: np.ndarray) -> np.ndarray:
    z = sess.run([out_name], {in_name: b_obs.reshape(1, -1).astype(np.float32)})[0]
    return z.reshape(-1).astype(np.float32)


def _compare_against_reference(z: np.ndarray, ref_path: str, title: str) -> None:
    ref = joblib.load(ref_path)
    if isinstance(ref, dict):
        print(f"[warn] reference {ref_path} is a dict; skipping array-style comparison")
        return
    ref = np.asarray(ref)
    if ref.shape != z.shape:
        print(f"[warn] {title}: shape mismatch ref={ref.shape} vs ours={z.shape}; comparing common prefix")
        n = min(ref.shape[0], z.shape[0])
        ref = ref[:n]
        z = z[:n]
    dot = np.sum(ref * z, axis=-1)
    n_ref = np.linalg.norm(ref, axis=-1)
    n_ours = np.linalg.norm(z, axis=-1)
    cos = dot / (n_ref * n_ours + 1e-12)
    l2 = np.linalg.norm(ref - z, axis=-1)
    print(f"[verify:{title}]  cos: mean={cos.mean():.6f} min={cos.min():.6f}  "
          f"L2: mean={l2.mean():.4f} max={l2.max():.4f}  "
          f"norm(ours): mean={n_ours.mean():.3f}  norm(ref): mean={n_ref.mean():.3f}")


# ---------- tracking ----------

def _convert_tracking_single(
    motion_path: str,
    output_path: str,
    sess: ort.InferenceSession,
    in_name: str,
    out_name: str,
    seq_length: int,
    start: int,
    end: int,
    scene_xml: str,
    reference_pkl: str | None = None,
) -> None:
    print(f"\n>> tracking: {motion_path}")
    motion = load_motion(motion_path, start=start, end=end)
    n_frames = motion["joint_pos"].shape[0]
    fps = motion["fps"]
    print(f"   {n_frames} frames @ {fps} fps  ({n_frames/fps:.1f}s)  format={motion['format']}")

    env = _build_env(scene_xml, dt=1.0 / fps)
    env.reset()

    z_raw = np.zeros((n_frames, Z_DIM), dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(n_frames):
        _set_frame(env, motion, i)
        b_obs = env.build_b_obs()
        z_raw[i] = _run_backward(sess, in_name, out_name, b_obs)
        if (i + 1) % 200 == 0:
            dt = time.perf_counter() - t0
            print(f"   frame {i+1}/{n_frames}  ({(i+1)/dt:.1f} fps)")

    # Sliding window averaging followed by re-projection (matches
    # FBModel.tracking_inference in BFM-Zero_inf/fb/model.py).
    z = z_raw.copy()
    for step in range(n_frames):
        end_idx = min(step + seq_length, n_frames)
        z[step] = z_raw[step:end_idx].mean(axis=0)
    z = _project_z(z).astype(np.float32)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(z, output_path)
    print(f"   saved {output_path}  shape={z.shape}  norm[0]={np.linalg.norm(z[0]):.3f}")

    if reference_pkl is not None and os.path.exists(reference_pkl):
        _compare_against_reference(z, reference_pkl, title=Path(motion_path).stem)


def _tracking_batch(args) -> None:
    sess, in_name, out_name = _make_session(args.model_path)
    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    npz_files = sorted(p for p in in_dir.rglob("*.npz"))
    if not npz_files:
        print(f"no .npz under {in_dir}")
        return
    print(f"batch tracking: {len(npz_files)} files under {in_dir}")
    for idx, p in enumerate(npz_files, 1):
        rel = p.relative_to(in_dir)
        out_path = out_dir / rel.with_name(rel.stem + "_z.pkl")
        print(f"\n[{idx}/{len(npz_files)}]")
        try:
            _convert_tracking_single(
                motion_path=str(p),
                output_path=str(out_path),
                sess=sess, in_name=in_name, out_name=out_name,
                seq_length=args.seq_length,
                start=args.start, end=args.end,
                scene_xml=args.scene_xml,
                reference_pkl=None,
            )
        except Exception as exc:
            print(f"   ERROR: {exc}")


def cmd_tracking(args) -> None:
    if args.input_dir or args.output_dir:
        if not (args.input_dir and args.output_dir):
            raise SystemExit("batch tracking requires both --input_dir and --output_dir")
        _tracking_batch(args)
        return
    if not (args.input and args.output):
        raise SystemExit("single tracking requires both --input and --output")
    sess, in_name, out_name = _make_session(args.model_path)
    _convert_tracking_single(
        motion_path=args.input,
        output_path=args.output,
        sess=sess, in_name=in_name, out_name=out_name,
        seq_length=args.seq_length,
        start=args.start, end=args.end,
        scene_xml=args.scene_xml,
        reference_pkl=args.reference_pkl,
    )


# ---------- goal ----------

def cmd_goal(args) -> None:
    sess, in_name, out_name = _make_session(args.model_path)

    with open(args.clips_config, "r") as f:
        cfg = yaml.safe_load(f)

    if "clips" not in cfg or not cfg["clips"]:
        raise SystemExit(f"{args.clips_config} must contain a non-empty 'clips' list")

    motion_root = Path(cfg.get("motion_root", ".")).expanduser()
    output_path = args.output or cfg.get("output")
    if not output_path:
        raise SystemExit("goal output path must be set via --output or the YAML 'output' field")

    motion_cache: dict[str, Dict[str, np.ndarray] | None] = {}

    def get_motion(motion_file: str) -> Dict[str, np.ndarray] | None:
        """Cached motion loader. Returns None when the file is missing or unreadable."""
        if motion_file not in motion_cache:
            full = motion_root / motion_file if not Path(motion_file).is_absolute() else Path(motion_file)
            if not full.exists():
                print(f"   [missing] {full}")
                motion_cache[motion_file] = None
            else:
                print(f"   loading {full}")
                try:
                    motion_cache[motion_file] = load_motion(str(full))
                except Exception as exc:
                    print(f"   [load failed] {full}: {exc}")
                    motion_cache[motion_file] = None
        return motion_cache[motion_file]

    z_dict: dict[str, np.ndarray] = {}
    for clip in cfg["clips"]:
        name = clip["name"]
        motion_file = clip["motion_file"]
        frame_idx = int(clip["frame_idx"])
        motion = get_motion(motion_file)
        if motion is None:
            print(f"   [skip] {name}: motion file unavailable")
            continue
        n = motion["joint_pos"].shape[0]
        if not (0 <= frame_idx < n):
            print(f"   [skip] {name}: frame_idx {frame_idx} out of range [0,{n})")
            continue

        env = _build_env(args.scene_xml, dt=1.0 / motion["fps"])
        env.reset()
        # Warm-up with previous frame so the body finite-difference velocities
        # match what BFM-Zero_inf would see when iterating through the motion.
        if frame_idx > 0:
            _set_frame(env, motion, frame_idx - 1)
            env.get_privileged_state()  # populate body_pos_prev / body_quat_prev
        _set_frame(env, motion, frame_idx)
        b_obs = env.build_b_obs()
        z = _run_backward(sess, in_name, out_name, b_obs)
        z_dict[name] = z.reshape(1, Z_DIM).astype(np.float32)
        print(f"   [{len(z_dict):3d}] {name:<40s}  norm={np.linalg.norm(z):.3f}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(z_dict, output_path)
    print(f"\nsaved {output_path}  ({len(z_dict)} goals)")

    if args.reference_pkl and os.path.exists(args.reference_pkl):
        ref = joblib.load(args.reference_pkl)
        common = sorted(set(z_dict) & set(ref))
        if common:
            cos_vals, l2_vals = [], []
            for k in common:
                a = z_dict[k].reshape(-1)
                b = np.asarray(ref[k]).reshape(-1)
                cos_vals.append(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                l2_vals.append(np.linalg.norm(a - b))
            print(f"[verify:goal]  common={len(common)}  "
                  f"cos: mean={np.mean(cos_vals):.6f} min={np.min(cos_vals):.6f}  "
                  f"L2: mean={np.mean(l2_vals):.4f} max={np.max(l2_vals):.4f}")
        else:
            print(f"[verify:goal] no overlapping keys with {args.reference_pkl}")


# ---------- reward (placeholder) ----------

def cmd_reward(_args) -> None:
    raise SystemExit("reward mode is not implemented in this script yet")


# ---------- entry ----------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model_path", default=DEFAULT_MODEL_PATH, help="backward ONNX path")
    common.add_argument("--scene_xml", default=DEFAULT_SCENE_XML, help="MuJoCo XML for obs construction")
    common.add_argument("--reference_pkl", default=None, help="optional pkl to compare against")

    t = sub.add_parser("tracking", parents=[common], help="motion sequence → tracking z [T,256]")
    t.add_argument("--input", "-i", default=None)
    t.add_argument("--output", "-o", default=None)
    t.add_argument("--input_dir", default=None)
    t.add_argument("--output_dir", default=None)
    t.add_argument("--start", type=int, default=0)
    t.add_argument("--end", type=int, default=-1)
    t.add_argument("--seq_length", type=int, default=1,
                   help="sliding-window length for tracking averaging (default 1)")
    t.set_defaults(func=cmd_tracking)

    g = sub.add_parser("goal", parents=[common], help="motion+frame → goal z dict")
    g.add_argument("--clips_config", required=True, help="YAML with motion_root + clips list")
    g.add_argument("--output", "-o", default=None, help="overrides 'output' in YAML")
    g.set_defaults(func=cmd_goal)

    r = sub.add_parser("reward", parents=[common], help="not implemented")
    r.set_defaults(func=cmd_reward)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
