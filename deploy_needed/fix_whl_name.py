#!/usr/bin/env python3
"""Rename a misnamed .whl file to the PEP 427 compliant filename.

Reads Name/Version from *.dist-info/METADATA and Tag from *.dist-info/WHEEL
inside the wheel (which is a zip archive), then renames the file accordingly.

Usage:
    python fix_whl_name.py <path-to-wheel.whl> [--dry-run]
"""
import argparse
import re
import sys
import zipfile
from pathlib import Path


def read_metadata(whl_path: Path) -> tuple[str, str, list[str]]:
    name = version = None
    tags: list[str] = []
    with zipfile.ZipFile(whl_path) as zf:
        meta_names = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        wheel_names = [n for n in zf.namelist() if n.endswith(".dist-info/WHEEL")]
        if not meta_names or not wheel_names:
            raise RuntimeError("Not a valid wheel: missing METADATA or WHEEL file")

        for line in zf.read(meta_names[0]).decode("utf-8", "replace").splitlines():
            if line.startswith("Name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("Version:"):
                version = line.split(":", 1)[1].strip()
            if name and version:
                break

        for line in zf.read(wheel_names[0]).decode("utf-8", "replace").splitlines():
            if line.startswith("Tag:"):
                tags.append(line.split(":", 1)[1].strip())

    if not (name and version and tags):
        raise RuntimeError(f"Incomplete metadata: name={name}, version={version}, tags={tags}")
    return name, version, tags


def normalize_name(name: str) -> str:
    # PEP 427: replace runs of non-alphanumeric with underscore
    return re.sub(r"[^\w\d.]+", "_", name, flags=re.UNICODE)


def build_filename(name: str, version: str, tag: str) -> str:
    return f"{normalize_name(name)}-{version}-{tag}.whl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="Path to the .whl file")
    parser.add_argument("--dry-run", action="store_true", help="Print target name only, do not rename")
    args = parser.parse_args()

    src: Path = args.wheel
    if not src.is_file():
        print(f"error: file not found: {src}", file=sys.stderr)
        return 1

    name, version, tags = read_metadata(src)
    # A wheel may declare multiple compressed tags (e.g. "cp310-cp310-linux_x86_64").
    # Per PEP 425 compressed tag set the filename uses dot-separated alternatives;
    # in practice each Tag: line is already one filename-ready compressed tag.
    target_name = build_filename(name, version, tags[0])
    target = src.with_name(target_name)

    print(f"detected   : Name={name}  Version={version}  Tag={tags[0]}")
    print(f"source     : {src.name}")
    print(f"target     : {target.name}")

    if src.resolve() == target.resolve():
        print("already correctly named, nothing to do.")
        return 0
    if target.exists():
        print(f"error: target already exists: {target}", file=sys.stderr)
        return 2
    if args.dry_run:
        print("(dry-run, no rename performed)")
        return 0

    src.rename(target)
    print(f"renamed -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
