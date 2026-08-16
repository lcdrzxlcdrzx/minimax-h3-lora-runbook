#!/usr/bin/env python3
"""Verify that the active Python environment can run CUDA BF16 work."""

from __future__ import annotations

import sys


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


try:
    import torch
except ImportError as exc:
    fail(f"PyTorch import failed: {exc}")

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    fail("CUDA is unavailable in this environment")

index = torch.cuda.current_device()
props = torch.cuda.get_device_properties(index)
print(f"GPU: {props.name}")
print(f"VRAM: {props.total_memory / 1024**3:.2f} GiB")
print(f"CUDA runtime: {torch.version.cuda}")

try:
    probe = torch.zeros((16, 16), device="cuda", dtype=torch.bfloat16)
    result = probe @ probe
    torch.cuda.synchronize()
except Exception as exc:
    fail(f"BF16 CUDA probe failed: {exc}")

if result.dtype != torch.bfloat16:
    fail(f"Unexpected BF16 probe dtype: {result.dtype}")

print("PASS: CUDA and BF16 tensor matmul are operational")
