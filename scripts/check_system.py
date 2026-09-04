#!/usr/bin/env python3
from __future__ import annotations

import platform
import sys

import torch

from uav_search.runtime import configure_runtime, resolve_device


def main() -> None:
    device = resolve_device("auto")
    profile = configure_runtime(device, deterministic=False, amp_mode="auto")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OS: {platform.platform()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    print(f"Selected device: {profile.device}")
    print(f"Device name: {profile.device_name}")
    if profile.compute_capability is not None:
        print(f"Compute capability: {profile.compute_capability[0]}.{profile.compute_capability[1]}")
    if profile.total_memory_gb is not None:
        print(f"VRAM: {profile.total_memory_gb:.2f} GiB")
    print(f"AMP auto: {'enabled' if profile.amp_enabled else 'disabled'}")
    print(f"Pinned replay memory: {profile.pin_memory}")
    print("Recommended train flags: --device auto --amp auto")
    print("For strict reproducibility add: --deterministic")


if __name__ == "__main__":
    main()
