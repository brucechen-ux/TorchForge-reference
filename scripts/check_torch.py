#!/usr/bin/env python3
import json
import platform
import sys

import torch


def main() -> int:
    info: dict[str, object] = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cudnn_available": torch.backends.cudnn.is_available(),
        "mps_available": getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available(),
        "distributed_available": torch.distributed.is_available(),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }
    if torch.cuda.is_available():
        devices = []
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            devices.append(
                {
                    "index": idx,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "multi_processor_count": props.multi_processor_count,
                    "major": props.major,
                    "minor": props.minor,
                }
            )
        info["devices"] = devices
        info["current_device"] = torch.cuda.current_device()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
