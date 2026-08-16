#!/usr/bin/env python3
"""Verify a safetensors file by size, optional SHA-256, and readable tensors."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--bytes", type=int, dest="expected_bytes")
    parser.add_argument("--sha256", dest="expected_sha256")
    args = parser.parse_args()

    path = args.path
    if not path.is_file():
        raise SystemExit(f"FAIL: not a file: {path}")

    actual_bytes = path.stat().st_size
    print(f"File: {path}")
    print(f"Bytes: {actual_bytes}")
    if args.expected_bytes is not None and actual_bytes != args.expected_bytes:
        raise SystemExit(f"FAIL: expected {args.expected_bytes} bytes")

    if args.expected_sha256:
        actual_sha256 = sha256(path)
        print(f"SHA-256: {actual_sha256}")
        if actual_sha256.lower() != args.expected_sha256.lower():
            raise SystemExit("FAIL: SHA-256 mismatch")

    try:
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            tensor_count = len(handle.keys())
    except Exception as exc:
        raise SystemExit(f"FAIL: safetensors readability check failed: {exc}") from exc

    print(f"Readable tensors: {tensor_count}")
    print("PASS: file integrity checks completed")


if __name__ == "__main__":
    main()
