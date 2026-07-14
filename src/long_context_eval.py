from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from src.checkpoint_io import (
    apply_config_overrides,
    load_config,
    load_state_dict_from_checkpoint,
    public_checkpoint_metadata,
)
from src.data import build_datasets
from src.modeling_v3 import DeepSeekV3LikeLM
from src.utils import move_batch_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--context_lengths", default="4096,8192,16384,32768,65536,131072")
    parser.add_argument("--max_batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--contexts", default=None)
    return parser.parse_args()


def parse_context_lengths(raw: str) -> list[int]:
    contexts: list[int] = []
    for item in raw.split(","):
        value = item.strip()
        if not value:
            continue
        contexts.append(int(value))
    if not contexts:
        raise ValueError("At least one context length is required.")
    return contexts


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def loss_to_ppl(loss: float) -> float:
    if not math.isfinite(loss):
        return float("inf")
    try:
        return float(math.exp(loss))
    except OverflowError:
        return float("inf")


@torch.inference_mode()
def evaluate_single_context(
    *,
    state_dict: dict[str, torch.Tensor],
    base_config: dict[str, Any],
    seq_len: int,
    device: torch.device,
    max_batches: int,
) -> dict[str, Any]:
    config = apply_config_overrides(base_config, seq_len=seq_len)
    config["train"]["micro_batch_size"] = int(config["train"].get("micro_batch_size", 1))
    config["data"]["num_workers"] = int(config["data"].get("num_workers", 0))
    config["data"]["pin_memory"] = False
    config["data"]["persistent_workers"] = False
    model = DeepSeekV3LikeLM(config).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    _, valid_ds = build_datasets(config)
    losses = {
        "loss": [],
        "lm_loss": [],
        "mtp_loss": [],
        "aux_loss": [],
    }
    for idx in range(min(max_batches, len(valid_ds))):
        batch = valid_ds[idx]
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        batch = move_batch_to_device(batch, device)
        outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
        for key in losses:
            losses[key].append(float(outputs[key].item()))

    num_batches = max(len(losses["loss"]), 1)
    avg_loss = sum(losses["loss"]) / num_batches
    avg_lm_loss = sum(losses["lm_loss"]) / num_batches
    avg_mtp_loss = sum(losses["mtp_loss"]) / num_batches
    avg_aux_loss = sum(losses["aux_loss"]) / num_batches
    return {
        "context_len": seq_len,
        "status": "ok",
        "avg_valid_loss": avg_loss,
        "valid_loss": avg_loss,
        "valid_total_loss": avg_loss,
        "valid_lm_loss": avg_lm_loss,
        "valid_mtp_loss": avg_mtp_loss,
        "valid_aux_loss": avg_aux_loss,
        "ppl": loss_to_ppl(avg_lm_loss),
        "ppl_lm": loss_to_ppl(avg_lm_loss),
        "ppl_total": loss_to_ppl(avg_loss),
        "num_batches": len(losses["loss"]),
    }


@torch.inference_mode()
def main() -> int:
    args = parse_args()
    contexts = parse_context_lengths(args.contexts or args.context_lengths)
    device = resolve_device(args.device)
    state_dict, checkpoint_meta = load_state_dict_from_checkpoint(args.checkpoint)
    base_config = load_config(
        args.config,
        checkpoint_config=checkpoint_meta.get("config"),
        data_dir=args.data_dir,
    )
    base_config["train"]["micro_batch_size"] = args.batch_size
    base_config["data"]["num_workers"] = args.num_workers
    base_config["data"]["pin_memory"] = False
    base_config["data"]["persistent_workers"] = False

    results: list[dict[str, Any]] = []
    for seq_len in contexts:
        try:
            results.append(
                evaluate_single_context(
                    state_dict=state_dict,
                    base_config=base_config,
                    seq_len=seq_len,
                    device=device,
                    max_batches=args.max_batches,
                )
            )
        except RuntimeError as exc:
            if device.type == "cuda" and "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                results.append(
                    {
                        "context_len": seq_len,
                        "status": "oom",
                        "error": str(exc),
                    }
                )
                continue
            raise

    payload = {
        **public_checkpoint_metadata(checkpoint_meta),
        "checkpoint_input": str(Path(args.checkpoint).expanduser().resolve()),
        "device": str(device),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "max_batches": args.max_batches,
        "max_supported_context_len": max(
            (int(item["context_len"]) for item in results if item.get("status") == "ok"),
            default=None,
        ),
        "contexts": contexts,
        "results": results,
    }
    if args.config:
        payload["config_path"] = str(Path(args.config).expanduser().resolve())
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
