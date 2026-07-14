from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from src.checkpoint_io import build_model_from_checkpoint, load_config
from src.modeling_v3 import DeepSeekV3LikeLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--prompt_len", type=int, default=128)
    parser.add_argument("--gen_len", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp32"), default="auto")
    return parser.parse_args()


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {
            "allocated_bytes": 0.0,
            "reserved_bytes": 0.0,
            "peak_allocated_bytes": 0.0,
            "peak_reserved_bytes": 0.0,
        }
    return {
        "allocated_bytes": float(torch.cuda.memory_allocated(device)),
        "reserved_bytes": float(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": float(torch.cuda.max_memory_reserved(device)),
    }


def get_autocast_dtype(device: torch.device, dtype_arg: str) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if dtype_arg == "fp32":
        return None
    if dtype_arg == "bf16":
        return torch.bfloat16
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def effective_dtype_name(autocast_dtype: torch.dtype | None) -> str:
    if autocast_dtype is None:
        return "float32"
    if autocast_dtype == torch.bfloat16:
        return "bfloat16"
    return str(autocast_dtype)


def forward_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    autocast_dtype: torch.dtype | None,
) -> torch.Tensor:
    autocast_enabled = autocast_dtype is not None
    with torch.autocast(
        device_type=input_ids.device.type,
        dtype=autocast_dtype if autocast_dtype is not None else torch.float32,
        enabled=autocast_enabled,
    ):
        outputs = model(input_ids=input_ids)
    return outputs["logits"]


def mean_or_zero(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


@torch.inference_mode()
def run_single_benchmark(
    model: torch.nn.Module,
    *,
    device: torch.device,
    vocab_size: int,
    model_context: int,
    batch_size: int,
    prompt_len: int,
    gen_len: int,
    autocast_dtype: torch.dtype | None,
) -> dict[str, Any]:
    effective_prompt_len = min(prompt_len, model_context)
    input_ids = torch.randint(0, vocab_size, (batch_size, effective_prompt_len), device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync_device(device)
    memory_before = memory_stats(device)

    prefill_start = time.perf_counter()
    logits = forward_logits(model, input_ids, autocast_dtype=autocast_dtype)
    sync_device(device)
    prefill_latency = max(time.perf_counter() - prefill_start, 1e-6)
    memory_after_prefill = memory_stats(device)

    next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = torch.cat([input_ids, next_token], dim=1)

    decode_latencies: list[float] = []
    for _ in range(max(gen_len - 1, 0)):
        step_input = generated[:, -model_context:]
        step_start = time.perf_counter()
        logits = forward_logits(model, step_input, autocast_dtype=autocast_dtype)
        sync_device(device)
        decode_latencies.append(max(time.perf_counter() - step_start, 1e-6))
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    total_decode_time = sum(decode_latencies)
    ttft = prefill_latency
    itl = mean_or_zero(decode_latencies)
    decode_tokens = max(gen_len - 1, 0) * batch_size
    decode_tokens_per_sec = decode_tokens / max(total_decode_time, 1e-6) if decode_tokens > 0 else 0.0
    end_to_end_latency = prefill_latency + total_decode_time

    peak_stats = memory_stats(device)
    return {
        "prefill_latency_sec": prefill_latency,
        "prefill_tokens_per_sec": (batch_size * effective_prompt_len) / prefill_latency,
        "ttft_sec": ttft,
        "itl_sec": itl,
        "decode_tokens_per_sec": decode_tokens_per_sec,
        "end_to_end_latency_sec": end_to_end_latency,
        "end_to_end_tokens_per_sec": (batch_size * gen_len) / max(end_to_end_latency, 1e-6),
        "actual_kv_cache_memory_bytes": 0.0,
        "memory_allocated_before_bytes": memory_before["allocated_bytes"],
        "memory_reserved_before_bytes": memory_before["reserved_bytes"],
        "memory_allocated_after_prefill_bytes": memory_after_prefill["allocated_bytes"],
        "memory_reserved_after_prefill_bytes": memory_after_prefill["reserved_bytes"],
        "peak_memory_allocated_bytes": peak_stats["peak_allocated_bytes"],
        "peak_memory_reserved_bytes": peak_stats["peak_reserved_bytes"],
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = get_autocast_dtype(device, args.dtype)

    checkpoint_meta: dict[str, Any]
    if args.checkpoint:
        model, config, checkpoint_meta = build_model_from_checkpoint(args.config, args.checkpoint, device)
    else:
        config = load_config(args.config)
        model = DeepSeekV3LikeLM(config).to(device)
        model.eval()
        checkpoint_meta = {
            "checkpoint_format": "random_init",
            "checkpoint_path": None,
        }

    vocab_size = int(config["model"]["vocab_size"])
    model_context = int(config["model"]["seq_len"])

    for _ in range(args.warmup):
        run_single_benchmark(
            model,
            device=device,
            vocab_size=vocab_size,
            model_context=model_context,
            batch_size=args.batch_size,
            prompt_len=args.prompt_len,
            gen_len=args.gen_len,
            autocast_dtype=autocast_dtype,
        )

    runs = [
        run_single_benchmark(
            model,
            device=device,
            vocab_size=vocab_size,
            model_context=model_context,
            batch_size=args.batch_size,
            prompt_len=args.prompt_len,
            gen_len=args.gen_len,
            autocast_dtype=autocast_dtype,
        )
        for _ in range(args.num_runs)
    ]

    scalar_keys = [key for key, value in runs[0].items() if isinstance(value, (int, float))]
    summary = {key: mean_or_zero([float(run[key]) for run in runs]) for key in scalar_keys}
    payload = {
        **checkpoint_meta,
        "config_path": str(Path(args.config).expanduser().resolve()),
        "checkpoint_input": str(Path(args.checkpoint).expanduser().resolve()) if args.checkpoint else None,
        "device": str(device),
        "dtype": effective_dtype_name(autocast_dtype),
        "batch_size": args.batch_size,
        "prompt_len": args.prompt_len,
        "effective_prompt_len": min(args.prompt_len, model_context),
        "generation_len": args.gen_len,
        "model_context": model_context,
        "warmup_runs": args.warmup,
        "measured_runs": args.num_runs,
        "cache_supported": False,
        "cache_note": "The current v4-attn training fork does not expose generation KV cache; metrics use the no-cache path.",
        "summary": summary,
        "runs": runs,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
