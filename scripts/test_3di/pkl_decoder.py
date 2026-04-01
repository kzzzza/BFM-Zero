#!/usr/bin/env python3
"""PKL 数据文件解码器 —— 递归打印 pkl 文件的结构与内容摘要。支持 pickle / joblib / torch 格式。"""

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np


def describe(obj, indent=0, max_items=5):
    """递归描述对象结构。"""
    prefix = "  " * indent

    if isinstance(obj, dict):
        print(f"{prefix}dict ({len(obj)} keys)")
        for i, (k, v) in enumerate(obj.items()):
            print(f"{prefix}  [{k!r}]:")
            describe(v, indent + 2, max_items)
            if i >= max_items - 1 and len(obj) > max_items:
                print(f"{prefix}  ... and {len(obj) - max_items} more keys")
                break

    elif isinstance(obj, (list, tuple)):
        tag = "list" if isinstance(obj, list) else "tuple"
        print(f"{prefix}{tag} (len={len(obj)})")
        for i, v in enumerate(obj[:max_items]):
            print(f"{prefix}  [{i}]:")
            describe(v, indent + 2, max_items)
        if len(obj) > max_items:
            print(f"{prefix}  ... and {len(obj) - max_items} more items")

    elif isinstance(obj, np.ndarray):
        print(f"{prefix}ndarray  shape={obj.shape}  dtype={obj.dtype}  "
              f"min={obj.min():.4f}  max={obj.max():.4f}  mean={obj.mean():.4f}")

    else:
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                print(f"{prefix}Tensor  shape={tuple(obj.shape)}  dtype={obj.dtype}  "
                      f"device={obj.device}")
                return
        except ImportError:
            pass
        print(f"{prefix}{type(obj).__name__}: {repr(obj)[:120]}")


def main():
    if len(sys.argv) < 2:
        print(f"用法: python {Path(__file__).name} <file.pkl> [max_items]")
        sys.exit(1)

    path = Path(sys.argv[1])
    max_items = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    if not path.exists():
        print(f"文件不存在: {path}")
        sys.exit(1)

    print(f"📦 解码: {path}  ({path.stat().st_size / 1024 / 1024:.1f} MB)\n")

    data = load_pkl(path)

    describe(data, max_items=max_items)


def load_pkl(path: Path):
    """依次尝试 joblib → torch.load → pickle 加载文件。"""
    # 1. 尝试 joblib（处理含 NumpyArrayWrapper 的文件）
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        pass

    # 2. 尝试 torch.load（处理含 torch.Tensor 的文件）
    try:
        import torch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        pass

    # 3. 回退到标准 pickle
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    main()
