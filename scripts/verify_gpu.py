#!/usr/bin/env python3
"""Verify the Python/PyTorch/CUDA runtime used for Swarm-SAR training."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from typing import Any


def _run_nvidia_smi() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - depends on host driver tools
        return f"nvidia-smi failed: {exc}"
    text = (out.stdout or out.stderr).strip()
    return text or None


def collect_status() -> dict[str, Any]:
    status: dict[str, Any] = {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_installed": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_device_count": 0,
        "gpu_name": None,
        "tiny_tensor_ok": False,
        "nvidia_smi": _run_nvidia_smi(),
    }
    try:
        import torch
    except Exception as exc:
        status["torch_error"] = str(exc)
        return status

    status["torch_installed"] = True
    status["torch_version"] = torch.__version__
    status["cuda_available"] = bool(torch.cuda.is_available())
    status["cuda_device_count"] = int(torch.cuda.device_count()) if status["cuda_available"] else 0
    if status["cuda_available"]:
        status["gpu_name"] = torch.cuda.get_device_name(0)
        try:
            x = torch.ones(8, device="cuda")
            y = (x * 2).sum().item()
            status["tiny_tensor_ok"] = bool(y == 16.0)
        except Exception as exc:  # pragma: no cover - depends on CUDA runtime
            status["tiny_tensor_error"] = str(exc)
    return status


def print_human(status: dict[str, Any]) -> None:
    print("Swarm-SAR GPU verification")
    print(f"  Python: {status['python_version']} ({status['python_executable']})")
    print(f"  Platform: {status['platform']}")
    print(f"  Torch installed: {status['torch_installed']}")
    print(f"  Torch version: {status['torch_version']}")
    print(f"  CUDA available: {status['cuda_available']}")
    print(f"  CUDA device count: {status['cuda_device_count']}")
    print(f"  GPU name: {status['gpu_name']}")
    print(f"  Tiny CUDA tensor op: {status['tiny_tensor_ok']}")
    if status.get("nvidia_smi"):
        print("  nvidia-smi:")
        for line in str(status["nvidia_smi"]).splitlines():
            print(f"    {line}")
    if status.get("torch_error"):
        print(f"  Torch error: {status['torch_error']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--require-cuda", action="store_true", help="exit nonzero when CUDA is unavailable")
    args = parser.parse_args()

    status = collect_status()
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print_human(status)

    if args.require_cuda and not status["cuda_available"]:
        return 2
    if not status["torch_installed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
