from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from src.checkpoint_io import build_model_from_checkpoint, public_checkpoint_metadata
from src.data import build_datasets
from src.utils import move_batch_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max_batches", type=int, default=128)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    return parser.parse_args()


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
def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    model, config, checkpoint_meta = build_model_from_checkpoint(
        args.config,
        args.checkpoint,
        device,
        data_dir=args.data_dir,
    )

    _, valid_ds = build_datasets(config)
    losses = {
        "loss": [],
        "lm_loss": [],
        "mtp_loss": [],
        "aux_loss": [],
    }
    for idx in range(min(args.max_batches, len(valid_ds))):
        batch = valid_ds[idx]
        batch = {k: v.unsqueeze(0) for k, v in batch.items()}
        batch = move_batch_to_device(batch, device)
        outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
        for key in losses:
            losses[key].append(float(outputs[key].item()))

    num_batches = max(len(losses["loss"]), 1)
    avg_valid_loss = sum(losses["loss"]) / num_batches
    avg_valid_lm_loss = sum(losses["lm_loss"]) / num_batches
    avg_valid_mtp_loss = sum(losses["mtp_loss"]) / num_batches
    avg_valid_aux_loss = sum(losses["aux_loss"]) / num_batches
    payload = {
        **public_checkpoint_metadata(checkpoint_meta),
        "checkpoint_input": str(Path(args.checkpoint).expanduser().resolve()),
        "avg_valid_loss": avg_valid_loss,
        "valid_loss": avg_valid_loss,
        "valid_total_loss": avg_valid_loss,
        "valid_lm_loss": avg_valid_lm_loss,
        "valid_mtp_loss": avg_valid_mtp_loss,
        "valid_aux_loss": avg_valid_aux_loss,
        "ppl": loss_to_ppl(avg_valid_lm_loss),
        "ppl_lm": loss_to_ppl(avg_valid_lm_loss),
        "ppl_total": loss_to_ppl(avg_valid_loss),
        "num_batches": len(losses["loss"]),
        "seq_len": int(config["train"]["seq_len"]),
    }
    if args.config:
        payload["config_path"] = str(Path(args.config).expanduser().resolve())
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
