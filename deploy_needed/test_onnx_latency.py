"""ONNX inference latency benchmark for BFM-Zero on Jetson.

Usage:
    python test_onnx_latency.py [model_path]

Default model_path: ../BFM-Zero_deploy/model/exported/FBcprAuxModel.onnx
"""
import argparse
import os
import sys
import time
import numpy as np
import onnxruntime as ort


def find_default_model():
    candidates = [
        "/home/unitree/workspace/yky/BFM-Zero_deploy/model/exported/FBcprAuxModel.onnx",
        os.path.expanduser("~/workspace/yky/BFM-Zero_deploy/model/exported/FBcprAuxModel.onnx"),
        "./model/exported/FBcprAuxModel.onnx",
        "../model/exported/FBcprAuxModel.onnx",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?", default=find_default_model())
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=500)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"[ERROR] Model not found: {args.model_path}")
        print("Pass the absolute model path as an argument.")
        sys.exit(1)

    print(f"Model: {args.model_path}")
    print(f"onnxruntime: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print()

    sess = ort.InferenceSession(
        args.model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    print(f"Active providers: {sess.get_providers()}")
    if sess.get_providers()[0] != "CUDAExecutionProvider":
        print("[WARN] CUDA is NOT the primary provider — model is running on CPU!")

    inputs = {}
    print("\nInput tensors:")
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in i.shape]
        inputs[i.name] = np.zeros(shape, dtype=np.float32)
        print(f"  {i.name:25s}  shape={i.shape}  dtype={i.type}")

    print(f"\nWarming up ({args.warmup} iters)...")
    for _ in range(args.warmup):
        sess.run(None, inputs)

    print(f"Benchmarking ({args.iters} iters)...")
    latencies = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(None, inputs)
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    print()
    print(f"  mean   : {arr.mean():.3f} ms")
    print(f"  median : {np.median(arr):.3f} ms")
    print(f"  p90    : {np.percentile(arr, 90):.3f} ms")
    print(f"  p99    : {np.percentile(arr, 99):.3f} ms")
    print(f"  max    : {arr.max():.3f} ms")
    print(f"  min    : {arr.min():.3f} ms")
    print()

    period_ms = 1000.0 / 50.0
    if arr.mean() < period_ms * 0.4:
        print(f"[OK] mean < {period_ms*0.4:.1f} ms (40% of 50 Hz period) — safe for real-time control")
    elif arr.mean() < period_ms:
        print(f"[MARGINAL] mean < {period_ms:.1f} ms but close — check Jetson power mode (sudo nvpmodel -m 0 && sudo jetson_clocks)")
    else:
        print(f"[FAIL] mean >= {period_ms:.1f} ms — 50 Hz control will miss deadlines")


if __name__ == "__main__":
    main()
