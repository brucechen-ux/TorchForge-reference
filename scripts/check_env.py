#!/usr/bin/env python3
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def which(binary: str) -> str | None:
    return shutil.which(binary)


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        output = (completed.stdout or completed.stderr).strip()
        return completed.returncode, output
    except Exception as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    disk = shutil.disk_usage(project_root)

    gpu_rc, gpu_out = run_cmd(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"]
    )

    env_info = {
        "project_root": str(project_root),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi_available": which("nvidia-smi") is not None,
        "gpu_query_success": gpu_rc == 0,
        "gpu_query_output": gpu_out,
        "deepspeed_available": has_module("deepspeed"),
        "tensorboard_available": has_module("tensorboard"),
        "datasets_available": has_module("datasets"),
        "transformers_available": has_module("transformers"),
        "disk_total_bytes": disk.total,
        "disk_used_bytes": disk.used,
        "disk_free_bytes": disk.free,
    }

    print(json.dumps(env_info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
