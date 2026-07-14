from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from src.data import build_dataloaders
from src.metrics import MeanAccumulator, compute_grad_norm
from src.modeling_v3 import DeepSeekV3LikeLM
from src.muon import InstrumentedAdamW, MuonWithAuxAdamW
from src.utils import (
    build_scheduler,
    cleanup_distributed,
    ensure_dir,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    load_yaml,
    move_batch_to_device,
    save_json,
    seed_everything,
    timestamp_tag,
    unwrap_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--deepspeed_config", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--override_max_steps", type=int, default=None)
    parser.add_argument("--override_seq_len", type=int, default=None)
    parser.add_argument("--override_micro_batch_size", type=int, default=None)
    parser.add_argument("--override_gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--override_gradient_clipping", type=float, default=None)
    parser.add_argument("--override_valid_max_batches", type=int, default=None)
    parser.add_argument("--override_activation_checkpointing", type=str, default=None)
    parser.add_argument("--override_moe_ep_size", type=int, default=None)
    parser.add_argument("--override_moe_top_k", type=int, default=None)
    parser.add_argument("--override_moe_capacity_factor", type=float, default=None)
    parser.add_argument("--override_moe_aux_loss_weight", type=float, default=None)
    parser.add_argument("--override_moe_balance_bias", type=str, default=None)
    parser.add_argument("--override_moe_drop_tokens", type=str, default=None)
    parser.add_argument("--override_mtp_enabled", type=str, default=None)
    parser.add_argument("--override_mtp_use_moe", type=str, default=None)
    parser.add_argument("--override_moe_use_tutel", type=str, default=None)
    parser.add_argument("--override_output_dir", default=None)
    parser.add_argument("--override_tensorboard_dir", default=None)
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--data_type", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--stop_valid_lm_loss_below", type=float, default=None)
    parser.add_argument("--load_balance_bias", default=None)
    parser.add_argument("--save_balance_bias", default=None)
    parser.add_argument("--balance_bias_calibration", action="store_true")
    parser.add_argument("--skip_final_checkpoint", action="store_true")
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", "0")))
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.override_max_steps is not None:
        config["train"]["max_steps"] = args.override_max_steps
        config["train"].pop("target_tokens", None)
    if args.override_seq_len is not None:
        config["train"]["seq_len"] = args.override_seq_len
        config["model"]["seq_len"] = args.override_seq_len
    if args.override_micro_batch_size is not None:
        config["train"]["micro_batch_size"] = args.override_micro_batch_size
    if args.override_gradient_accumulation_steps is not None:
        config["train"]["gradient_accumulation_steps"] = args.override_gradient_accumulation_steps
    if args.override_gradient_clipping is not None:
        config["train"]["gradient_clipping"] = args.override_gradient_clipping
    if args.override_valid_max_batches is not None:
        config["train"]["valid_max_batches"] = args.override_valid_max_batches
    if args.override_activation_checkpointing is not None:
        config["train"]["activation_checkpointing"] = args.override_activation_checkpointing.lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if args.override_moe_ep_size is not None:
        config.setdefault("moe", {})["moe_ep_size"] = args.override_moe_ep_size
    if args.override_moe_top_k is not None:
        config.setdefault("moe", {})["num_experts_per_token"] = args.override_moe_top_k
        config.setdefault("moe", {})["top_k"] = args.override_moe_top_k
    if args.override_moe_capacity_factor is not None:
        config.setdefault("moe", {})["moe_capacity_factor"] = args.override_moe_capacity_factor
        config.setdefault("moe", {})["moe_eval_capacity_factor"] = args.override_moe_capacity_factor
    if args.override_moe_aux_loss_weight is not None:
        config.setdefault("moe", {})["aux_loss_weight"] = args.override_moe_aux_loss_weight
    if args.override_moe_balance_bias is not None:
        config.setdefault("moe", {})["balance_bias"] = args.override_moe_balance_bias.lower() in {"1", "true", "yes", "on"}
    if args.override_moe_drop_tokens is not None:
        config.setdefault("moe", {})["moe_drop_tokens"] = args.override_moe_drop_tokens.lower() in {"1", "true", "yes", "on"}
    if args.override_mtp_enabled is not None:
        config.setdefault("mtp", {})["enabled"] = args.override_mtp_enabled.lower() in {"1", "true", "yes", "on"}
    if args.override_mtp_use_moe is not None:
        config.setdefault("mtp", {})["mtp_use_moe"] = args.override_mtp_use_moe.lower() in {"1", "true", "yes", "on"}
    if args.override_moe_use_tutel is not None:
        config.setdefault("moe", {})["moe_use_tutel"] = args.override_moe_use_tutel.lower() in {"1", "true", "yes", "on"}
    if args.override_output_dir:
        config["train"]["output_dir"] = args.override_output_dir
    if args.override_tensorboard_dir:
        config["train"]["tensorboard_dir"] = args.override_tensorboard_dir
    if args.data_type:
        config["data"]["type"] = args.data_type
    if args.stop_valid_lm_loss_below is not None:
        config["train"]["stop_valid_lm_loss_below"] = args.stop_valid_lm_loss_below
    return config


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    config: dict[str, Any],
) -> None:
    state = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "config": config,
    }
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
) -> int:
    state = torch.load(path, map_location=device)
    unwrap_model(model).load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return int(state.get("step", 0))


def build_deepspeed_config(path: str | Path, train_cfg: dict[str, Any], world_size: int) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        ds_config = json.load(handle)

    micro_batch = int(train_cfg["micro_batch_size"])
    grad_accum = int(train_cfg["gradient_accumulation_steps"])
    ds_config["train_micro_batch_size_per_gpu"] = micro_batch
    ds_config["gradient_accumulation_steps"] = grad_accum
    ds_config["train_batch_size"] = micro_batch * grad_accum * max(world_size, 1)
    ds_config["gradient_clipping"] = float(train_cfg.get("gradient_clipping", ds_config.get("gradient_clipping", 1.0)))
    if "bf16" in ds_config:
        ds_config["bf16"]["enabled"] = bool(
            train_cfg.get("bf16", False) and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        )
    if os.environ.get("DS_WALL_CLOCK_BREAKDOWN", "").lower() in {"1", "true", "yes", "on"}:
        ds_config["wall_clock_breakdown"] = True
    return ds_config


def to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        detached = value.detach().float().cpu()
        if detached.ndim == 0:
            return float(detached.item())
        return detached.tolist()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def summarize_model_config(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config["model"]
    moe_cfg = config.get("moe", {})
    num_layers = int(model_cfg["num_layers"])
    mtp_depth = int(config.get("mtp", {}).get("mtp_depth", 0)) if config.get("mtp", {}).get("enabled", False) else 0
    mlp_layer_types = moe_cfg.get("mlp_layer_types")
    if mlp_layer_types is None:
        num_hash_layers = int(moe_cfg.get("num_hash_layers", 3))
        first_dense_layers = int(model_cfg.get("first_dense_layers", model_cfg.get("n_dense_layers", 0)))
        dense_count = min(num_layers, max(first_dense_layers, 0))
        remaining_layers = max(num_layers - dense_count, 0)
        hash_count = min(remaining_layers, max(num_hash_layers, 0))
        mlp_layer_types = ["dense"] * dense_count
        mlp_layer_types.extend(["hash_moe"] * hash_count)
        mlp_layer_types.extend(["moe"] * max(num_layers - len(mlp_layer_types), 0))
    mlp_layer_types = list(mlp_layer_types)
    main_dense_layers = [idx for idx, layer_type in enumerate(mlp_layer_types[:num_layers]) if layer_type == "dense"]
    main_hash_moe_layers = [idx for idx, layer_type in enumerate(mlp_layer_types[:num_layers]) if layer_type == "hash_moe"]
    main_learned_moe_layers = [idx for idx, layer_type in enumerate(mlp_layer_types[:num_layers]) if layer_type == "moe"]
    main_moe_layers = main_hash_moe_layers + main_learned_moe_layers
    mtp_moe_layers = list(range(num_layers, num_layers + mtp_depth))
    return {
        "num_layers": num_layers,
        "hidden_size": int(model_cfg["hidden_size"]),
        "num_experts": int(moe_cfg.get("num_routed_experts", 0)),
        "moe_top_k": int(moe_cfg.get("num_experts_per_token", moe_cfg.get("top_k", 0))),
        "moe_layer_freq": "all transformer layers; hash_moe bootstrap then learned moe plus MTP layers",
        "main_dense_layers": main_dense_layers,
        "main_moe_layers": main_moe_layers,
        "main_hash_moe_layers": main_hash_moe_layers,
        "main_learned_moe_layers": main_learned_moe_layers,
        "mtp_moe_layers": mtp_moe_layers,
        "capacity_factor": float(moe_cfg.get("moe_capacity_factor", 0.0)),
        "min_capacity": int(moe_cfg.get("moe_min_capacity", 0)),
        "drop_tokens": bool(moe_cfg.get("moe_drop_tokens", False)),
        "moe_aux_loss_coef": float(moe_cfg.get("aux_loss_weight", 0.0)),
        "balance_bias": bool(moe_cfg.get("balance_bias", False)),
        "mtp_enabled": bool(config.get("mtp", {}).get("enabled", False)),
        "mtp_use_moe": bool(config.get("mtp", {}).get("mtp_use_moe", True)),
        "moe_implementation": str(moe_cfg.get("implementation", "torch")),
        "moe_ep_size": int(moe_cfg.get("moe_ep_size", 1)),
        "moe_use_tutel": bool(moe_cfg.get("moe_use_tutel", False)),
    }


def collect_ep_group_info(rank: int, world_size: int, moe_cfg: dict[str, Any]) -> dict[str, Any]:
    ep_size = int(moe_cfg.get("moe_ep_size", 1))
    group_name = f"ep_size_{ep_size}"
    ep_group_ranks: list[int]
    if ep_size > 0 and world_size % ep_size == 0:
        group_start = (rank // ep_size) * ep_size
        ep_group_ranks = list(range(group_start, group_start + ep_size))
    else:
        ep_group_ranks = []

    try:
        from deepspeed.utils import groups

        group_ranks = groups._get_expert_parallel_group_ranks(group_name)
        if group_ranks is not None:
            ep_group_ranks = list(group_ranks)
    except Exception:
        pass
    return {"rank": rank, "ep_size": ep_size, "ep_group_name": group_name, "ep_group_ranks": ep_group_ranks}


def gather_rank_payload(payload: dict[str, Any], distributed: bool, device: torch.device) -> list[dict[str, Any]]:
    if not distributed or not torch.distributed.is_initialized():
        return [payload]
    gathered: list[Any] = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered, payload)
    return [item for item in gathered if isinstance(item, dict)]


def maybe_print_run_diagnostics(
    config: dict[str, Any],
    ds_config: dict[str, Any] | None,
    rank: int,
    world_size: int,
    distributed: bool,
    device: torch.device,
) -> None:
    ep_info = collect_ep_group_info(rank, world_size, config.get("moe", {}))
    all_ep_info = gather_rank_payload(ep_info, distributed, device)
    if is_main_process():
        payload = {
            "diagnostic": "run_config",
            "deepspeed_config": ds_config,
            "model_config": summarize_model_config(config),
            "rank_ep_groups": sorted(all_ep_info, key=lambda item: int(item.get("rank", -1))),
        }
        print(json.dumps(payload, ensure_ascii=False))


def summarize_moe_routing(outputs: dict[str, Any]) -> dict[str, Any]:
    layer_stats = outputs.get("moe_routing_stats") or []
    if not layer_stats:
        return {}

    total_routed = 0.0
    total_dropped = 0.0
    total_kept = 0.0
    capacity_utils = []
    max_capacity_utils = []
    load_cvs = []
    l_aux_values = []
    layers = []
    for item in layer_stats:
        routed = float(to_jsonable(item.get("routed_assignments", 0.0)))
        dropped = float(to_jsonable(item.get("dropped_assignments", 0.0)))
        kept = float(to_jsonable(item.get("kept_assignments", 0.0)))
        total_routed += routed
        total_dropped += dropped
        total_kept += kept
        capacity_utils.append(float(to_jsonable(item.get("capacity_utilization", 0.0))))
        max_capacity_utils.append(float(to_jsonable(item.get("max_capacity_utilization", 0.0))))
        load_cvs.append(float(to_jsonable(item.get("load_cv", 0.0))))
        l_aux_values.append(float(to_jsonable(item.get("l_aux", 0.0))))
        layers.append(
            {
                "layer": item.get("layer"),
                "scope": item.get("scope"),
                "num_tokens": to_jsonable(item.get("num_tokens")),
                "top_k": to_jsonable(item.get("top_k")),
                "num_experts": len(to_jsonable(item.get("expert_counts", [])) or []),
                "capacity": to_jsonable(item.get("capacity")),
                "routed_assignments": to_jsonable(item.get("routed_assignments")),
                "kept_assignments": to_jsonable(item.get("kept_assignments")),
                "dropped_assignments": to_jsonable(item.get("dropped_assignments")),
                "drop_rate": to_jsonable(item.get("drop_rate")),
                "capacity_utilization": to_jsonable(item.get("capacity_utilization")),
                "max_capacity_utilization": to_jsonable(item.get("max_capacity_utilization")),
                "expert_load_mean": to_jsonable(item.get("expert_load_mean")),
                "expert_load_max": to_jsonable(item.get("expert_load_max")),
                "expert_load_p95": to_jsonable(item.get("expert_load_p95")),
                "expert_load_std": to_jsonable(item.get("expert_load_std")),
                "max_load_over_mean": to_jsonable(item.get("max_load_over_mean")),
                "load_cv": to_jsonable(item.get("load_cv")),
                "l_aux": to_jsonable(item.get("l_aux")),
                "gate_logits_mean": to_jsonable(item.get("gate_logits_mean")),
                "gate_logits_std": to_jsonable(item.get("gate_logits_std")),
                "router_entropy": to_jsonable(item.get("router_entropy")),
                "top1_expert_counts": to_jsonable(item.get("top1_expert_counts")),
                "topk_expert_counts": to_jsonable(item.get("topk_expert_counts")),
                "balance_bias": to_jsonable(item.get("balance_bias")),
                "pre_expert_counts": to_jsonable(item.get("pre_expert_counts")),
                "expert_counts": to_jsonable(item.get("expert_counts")),
                "dropped_per_expert": to_jsonable(item.get("dropped_per_expert")),
                "router_time_ms": to_jsonable(item.get("router_time_ms")),
                "moe_dispatch_time_ms": to_jsonable(item.get("moe_dispatch_time_ms")),
                "moe_combine_time_ms": to_jsonable(item.get("moe_combine_time_ms")),
            }
        )

    timing_stats = outputs.get("moe_timing_stats") or []
    moe_time = sum(float(to_jsonable(item.get("moe_time", 0.0))) for item in timing_stats)
    moe_first_a2a = sum(float(to_jsonable(item.get("first_alltoall_time", 0.0))) for item in timing_stats)
    moe_second_a2a = sum(float(to_jsonable(item.get("second_alltoall_time", 0.0))) for item in timing_stats)
    router_time = sum(float(to_jsonable(item.get("router_time", 0.0))) for item in timing_stats)
    dispatch_time = sum(float(to_jsonable(item.get("moe_dispatch_time", 0.0))) for item in timing_stats)
    expert_mlp_time = sum(float(to_jsonable(item.get("expert_mlp_time", 0.0))) for item in timing_stats)
    combine_time = sum(float(to_jsonable(item.get("moe_combine_time", 0.0))) for item in timing_stats)
    block_timing_stats = outputs.get("block_timing_stats") or []
    attention_time = sum(float(to_jsonable(item.get("attention_time", 0.0))) for item in block_timing_stats)
    dense_mlp_time = sum(float(to_jsonable(item.get("dense_mlp_time", 0.0))) for item in block_timing_stats)
    return {
        "layers": layers,
        "num_moe_layers": len(layers),
        "routed_assignments": total_routed,
        "kept_assignments": total_kept,
        "dropped_assignments": total_dropped,
        "drop_rate": total_dropped / max(total_routed, 1.0),
        "avg_capacity_utilization": sum(capacity_utils) / max(len(capacity_utils), 1),
        "max_capacity_utilization": max(max_capacity_utils) if max_capacity_utils else 0.0,
        "avg_load_cv": sum(load_cvs) / max(len(load_cvs), 1),
        "max_load_cv": max(load_cvs) if load_cvs else 0.0,
        "avg_l_aux": sum(l_aux_values) / max(len(l_aux_values), 1),
        "moe_time_ms": moe_time,
        "moe_first_alltoall_time_ms": moe_first_a2a,
        "moe_second_alltoall_time_ms": moe_second_a2a,
        "router_time_ms": router_time,
        "moe_dispatch_time_ms": dispatch_time,
        "expert_mlp_time_ms": expert_mlp_time,
        "moe_combine_time_ms": combine_time,
        "attention_time_ms": attention_time,
        "dense_mlp_time_ms": dense_mlp_time,
    }


def aggregate_moe_routing_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [summary for summary in summaries if summary]
    if not summaries:
        return {}
    total_routed = sum(float(summary.get("routed_assignments", 0.0)) for summary in summaries)
    total_kept = sum(float(summary.get("kept_assignments", 0.0)) for summary in summaries)
    total_dropped = sum(float(summary.get("dropped_assignments", 0.0)) for summary in summaries)
    count = max(len(summaries), 1)

    def avg(key: str) -> float:
        return sum(float(summary.get(key, 0.0)) for summary in summaries) / count

    return {
        "layers": summaries[-1].get("layers", []),
        "num_moe_layers": summaries[-1].get("num_moe_layers", 0),
        "routed_assignments": total_routed,
        "kept_assignments": total_kept,
        "dropped_assignments": total_dropped,
        "drop_rate": total_dropped / max(total_routed, 1.0),
        "avg_capacity_utilization": avg("avg_capacity_utilization"),
        "max_capacity_utilization": max(float(summary.get("max_capacity_utilization", 0.0)) for summary in summaries),
        "avg_load_cv": avg("avg_load_cv"),
        "max_load_cv": max(float(summary.get("max_load_cv", 0.0)) for summary in summaries),
        "avg_l_aux": avg("avg_l_aux"),
        "moe_time_ms": avg("moe_time_ms"),
        "moe_first_alltoall_time_ms": avg("moe_first_alltoall_time_ms"),
        "moe_second_alltoall_time_ms": avg("moe_second_alltoall_time_ms"),
        "router_time_ms": avg("router_time_ms"),
        "moe_dispatch_time_ms": avg("moe_dispatch_time_ms"),
        "expert_mlp_time_ms": avg("expert_mlp_time_ms"),
        "moe_combine_time_ms": avg("moe_combine_time_ms"),
        "attention_time_ms": avg("attention_time_ms"),
        "dense_mlp_time_ms": avg("dense_mlp_time_ms"),
    }


def get_deepspeed_wall_clock(model: torch.nn.Module) -> dict[str, float]:
    if not hasattr(model, "wall_clock_breakdown") or not model.wall_clock_breakdown():
        return {}
    timer_names = {
        "ds_forward_ms": "fwd",
        "ds_backward_ms": "bwd",
        "ds_backward_inner_ms": "bwd_inner",
        "ds_backward_reduce_ms": "bwd_allreduce",
        "ds_step_ms": "step",
        "ds_optimizer_gradients_ms": "optimizer_gradients",
        "ds_optimizer_step_ms": "optimizer_step",
        "ds_optimizer_allgather_ms": "optimizer_allgather",
    }
    timers = getattr(model, "timers", None)
    if timers is None:
        return {}
    cached_timers = {}
    if hasattr(model, "get_wall_clock_timers"):
        try:
            cached_timers = model.get_wall_clock_timers() or {}
        except Exception:
            cached_timers = {}
    existing_timers = {}
    if hasattr(timers, "get_timers"):
        try:
            existing_timers = timers.get_timers()
        except Exception:
            existing_timers = {}
    result: dict[str, float] = {}
    for output_name, timer_name in timer_names.items():
        if timer_name in cached_timers:
            result[output_name] = float(cached_timers[timer_name])
            continue
        if existing_timers and timer_name not in existing_timers:
            continue
        try:
            result[output_name] = float(timers(timer_name).elapsed(reset=False))
        except Exception:
            pass
    return result


def initialize_deepspeed(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    ds_config: dict[str, Any],
):
    try:
        import deepspeed
        import deepspeed.utils.nvtx as ds_nvtx
    except ImportError as exc:
        raise RuntimeError(
            "--deepspeed_config was provided, but deepspeed is not importable in this Python environment."
        ) from exc
    ds_nvtx.enable_nvtx = False

    engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
    )
    return engine, optimizer, scheduler


def build_optimizer(
    model: torch.nn.Module,
    train_cfg: dict[str, Any],
    deepspeed_enabled: bool,
) -> torch.optim.Optimizer:
    optimizer_cfg = train_cfg.get("optimizer", {})
    optimizer_name = str(optimizer_cfg.get("name", "adamw")).lower()
    metrics_interval = int(optimizer_cfg.get("metrics_interval", train_cfg.get("log_steps", 10)))
    named_parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    model_parameters: Any
    if optimizer_name == "adamw":
        model_parameters = [param for _, param in named_parameters]
    elif optimizer_name == "muon":
        muon_params: list[torch.nn.Parameter] = []
        adamw_params: list[torch.nn.Parameter] = []
        excluded_fragments = tuple(optimizer_cfg.get("exclude_from_muon", []))
        for name, param in named_parameters:
            lowered = name.lower()
            report_adamw = (
                "embed_tokens" in lowered
                or "lm_head" in lowered
                or "norm" in lowered
                or ".router." in lowered
                or ".ffn.gate." in lowered
                or "balance_bias" in lowered
                or "static_bias" in lowered
                or "gating_factor" in lowered
            )
            eligible = param.ndim in {2, 3} and not report_adamw and not any(
                fragment in lowered for fragment in excluded_fragments
            )
            (muon_params if eligible else adamw_params).append(param)
        model_parameters = [
            {"params": muon_params, "name": "muon-matrices", "use_muon": True},
            {"params": adamw_params, "name": "adamw-aux", "use_muon": False},
        ]
        model_parameters = [group for group in model_parameters if group["params"]]
    else:
        raise ValueError(f"Unsupported optimizer name: {optimizer_name}")
    if deepspeed_enabled:
        try:
            from deepspeed.moe.utils import configure_moe_param_groups
        except ImportError as exc:
            raise RuntimeError(
                "--deepspeed_config was provided, but deepspeed is not importable in this Python environment."
            ) from exc
        model_parameters = configure_moe_param_groups(model_parameters)
    if optimizer_name == "adamw":
        return InstrumentedAdamW(
            model_parameters,
            lr=train_cfg["learning_rate"],
            betas=tuple(optimizer_cfg.get("betas", (0.9, 0.95))),
            eps=float(optimizer_cfg.get("eps", 1.0e-8)),
            weight_decay=train_cfg["weight_decay"],
            metrics_interval=metrics_interval,
        )
    ns_method = str(optimizer_cfg.get("newton_schulz", optimizer_cfg.get("ns_method", "hybrid")))
    hybrid_cfg = optimizer_cfg.get("hybrid", {})
    standard_cfg = optimizer_cfg.get("standard", {})
    optimizer = MuonWithAuxAdamW(
        model_parameters,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        momentum=float(optimizer_cfg.get("momentum", 0.95)),
        nesterov=bool(optimizer_cfg.get("nesterov", True)),
        ns_method=ns_method,
        ns_iterations=int(optimizer_cfg.get("newton_schulz_iterations", 10)),
        ns_first_stage_steps=int(hybrid_cfg.get("first_stage_steps", 8)),
        ns_second_stage_steps=int(hybrid_cfg.get("second_stage_steps", 2)),
        ns_first_stage_coefficients=tuple(hybrid_cfg.get("first_stage_coefficients", (3.4445, -4.7750, 2.0315))),
        ns_second_stage_coefficients=tuple(hybrid_cfg.get("second_stage_coefficients", (2.0, -1.5, 0.5))),
        ns_standard_steps=int(standard_cfg.get("steps", 10)),
        ns_standard_coefficients=tuple(standard_cfg.get("coefficients", (2.0, -1.5, 0.5))),
        update_rms_target=float(optimizer_cfg.get("update_rms_target", 0.18)),
        adamw_betas=tuple(optimizer_cfg.get("betas", (0.9, 0.95))),
        adamw_eps=float(optimizer_cfg.get("eps", 1.0e-8)),
        metrics_interval=metrics_interval,
        diagnostics_interval=int(optimizer_cfg.get("diagnostics_interval", 500)),
        diagnostics_max_matrices=int(optimizer_cfg.get("diagnostics_max_matrices", 4)),
        diagnostics_max_singular_values=int(optimizer_cfg.get("diagnostics_max_singular_values", 128)),
    )
    optimizer.parameter_names = {id(param): name for name, param in named_parameters}
    return optimizer


def find_instrumented_optimizer(optimizer: Any) -> Any | None:
    """Find our base optimizer through DeepSpeed wrappers."""
    current = optimizer
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "last_step_metrics"):
            return current
        next_optimizer = None
        for attribute in ("optimizer", "basic_optimizer", "base_optimizer"):
            candidate = getattr(current, attribute, None)
            if candidate is not None and candidate is not current:
                next_optimizer = candidate
                break
        current = next_optimizer
    return None


def optimizer_state_bytes(optimizer: Any) -> int:
    base = find_instrumented_optimizer(optimizer) or optimizer
    total = 0
    for state in getattr(base, "state", {}).values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.numel() * value.element_size()
    return total


def append_csv_row(path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def init_single_process_deepspeed_group(local_rank: int, device: torch.device) -> bool:
    if torch.distributed.is_initialized():
        return True

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    backend = "nccl" if device.type == "cuda" else "gloo"
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend=backend, rank=0, world_size=1)
    return True


def save_deepspeed_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    step: int,
    config: dict[str, Any],
    tag: str,
) -> None:
    client_state = {
        "step": step,
        "config": config,
    }
    model.save_checkpoint(str(output_dir), tag=tag, client_state=client_state)


def load_deepspeed_checkpoint(path: str | Path, model: torch.nn.Module) -> int:
    _load_path, client_state = model.load_checkpoint(str(path))
    if client_state is None:
        return 0
    return int(client_state.get("step", 0))


def _balance_bias_modules(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    root = unwrap_model(model)
    result: dict[str, torch.Tensor] = {}
    for name, module in root.named_modules():
        bias = getattr(module, "balance_bias", None)
        if torch.is_tensor(bias):
            result[name] = bias
    return result


def apply_pending_balance_bias_updates(model: torch.nn.Module) -> None:
    root = unwrap_model(model)
    for module in root.modules():
        apply_update = getattr(module, "apply_pending_balance_bias_update", None)
        if callable(apply_update):
            apply_update()


def _balance_bias_summary(named_biases: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    summary = []
    for name, bias in sorted(named_biases.items()):
        values = bias.detach().float().cpu()
        summary.append(
            {
                "name": name,
                "numel": int(values.numel()),
                "min": float(values.min().item()) if values.numel() else 0.0,
                "max": float(values.max().item()) if values.numel() else 0.0,
                "mean": float(values.mean().item()) if values.numel() else 0.0,
                "std": float(values.std(unbiased=False).item()) if values.numel() else 0.0,
            }
        )
    return summary


def save_balance_bias(path: str | Path, model: torch.nn.Module, rank: int, world_size: int) -> None:
    named_biases = _balance_bias_modules(model)
    rank_payload = {
        "rank": rank,
        "biases": {name: bias.detach().float().cpu() for name, bias in named_biases.items()},
        "summary": _balance_bias_summary(named_biases),
    }
    gathered: list[Any] = [None for _ in range(world_size)] if torch.distributed.is_initialized() else []
    if torch.distributed.is_initialized():
        torch.distributed.all_gather_object(gathered, rank_payload)
    else:
        gathered = [rank_payload]
    if is_main_process():
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "deepseek_v3_tiny_balance_bias_v1",
            "world_size": world_size,
            "rank_biases": {int(item["rank"]): item["biases"] for item in gathered if isinstance(item, dict)},
            "rank_summaries": {int(item["rank"]): item["summary"] for item in gathered if isinstance(item, dict)},
        }
        torch.save(payload, save_path)
        print(
            json.dumps(
                {
                    "balance_bias_saved": str(save_path),
                    "world_size": world_size,
                    "rank0_summary": payload["rank_summaries"].get(0, []),
                },
                ensure_ascii=False,
            )
        )


def load_balance_bias(path: str | Path, model: torch.nn.Module, rank: int) -> dict[str, Any]:
    load_path = Path(path)
    payload = torch.load(load_path, map_location="cpu")
    if isinstance(payload, dict) and "rank_biases" in payload:
        rank_biases = payload["rank_biases"]
        state = rank_biases.get(rank) or rank_biases.get(str(rank)) or rank_biases.get(0) or rank_biases.get("0") or {}
    else:
        state = payload
    named_biases = _balance_bias_modules(model)
    loaded: list[str] = []
    missing: list[str] = []
    shape_mismatch: list[str] = []
    with torch.no_grad():
        for name, bias in named_biases.items():
            source = state.get(name) if isinstance(state, dict) else None
            if source is None:
                missing.append(name)
                continue
            source_tensor = torch.as_tensor(source, dtype=bias.dtype, device=bias.device)
            if tuple(source_tensor.shape) != tuple(bias.shape):
                shape_mismatch.append(name)
                continue
            bias.copy_(source_tensor)
            loaded.append(name)
    result = {
        "path": str(load_path),
        "rank": rank,
        "loaded": loaded,
        "missing": missing,
        "shape_mismatch": shape_mismatch,
        "summary": _balance_bias_summary(named_biases),
    }
    if is_main_process():
        print(json.dumps({"balance_bias_loaded": result}, ensure_ascii=False))
    return result


def get_current_lr(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
) -> float:
    param_groups = getattr(optimizer, "param_groups", None)
    if param_groups:
        return float(param_groups[0]["lr"])
    if hasattr(scheduler, "get_last_lr"):
        return float(scheduler.get_last_lr()[0])
    return 0.0


def get_gradient_norm(model: torch.nn.Module) -> float:
    if hasattr(model, "get_global_grad_norm"):
        grad_norm = model.get_global_grad_norm()
        if grad_norm is not None:
            if torch.is_tensor(grad_norm):
                return float(grad_norm.detach().float().item())
            return float(grad_norm)
    return compute_grad_norm(model.parameters())


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compute_ppl(loss: float | None) -> float | None:
    if loss is None:
        return None
    try:
        return float(torch.exp(torch.tensor(float(loss), dtype=torch.float64)).item())
    except OverflowError:
        return float("inf")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    valid_loader,
    device: torch.device,
    max_batches: int | None = 8,
) -> dict[str, float]:
    model.eval()
    meters = {
        "loss": MeanAccumulator(),
        "lm_loss": MeanAccumulator(),
        "mtp_loss": MeanAccumulator(),
        "aux_loss": MeanAccumulator(),
    }
    for batch_idx, batch in enumerate(valid_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
        for key, meter in meters.items():
            meter.update(float(outputs[key].item()))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        payload = []
        for meter in meters.values():
            payload.extend([meter.total, float(meter.count)])
        stats = torch.tensor(payload, device=device, dtype=torch.float64)
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        for idx, meter in enumerate(meters.values()):
            meter.total = float(stats[idx * 2].item())
            meter.count = int(stats[idx * 2 + 1].item())
    model.train()
    return {key: meter.average for key, meter in meters.items()}


def main() -> int:
    args = parse_args()
    config = apply_overrides(load_yaml(args.config), args)
    deepspeed_enabled = args.deepspeed_config is not None
    ds_config: dict[str, Any] | None = None
    seed_everything(int(config.get("seed", config.get("train", {}).get("seed", 2026))))

    distributed, device = init_distributed(args.local_rank)
    if deepspeed_enabled and not torch.distributed.is_initialized():
        distributed = init_single_process_deepspeed_group(args.local_rank, device)
    rank = get_rank()
    world_size = get_world_size()
    is_cuda = device.type == "cuda"
    train_cfg = config["train"]
    global_batch_tokens = (
        int(train_cfg["micro_batch_size"])
        * int(train_cfg["gradient_accumulation_steps"])
        * int(train_cfg["seq_len"])
        * world_size
    )
    target_tokens = train_cfg.get("target_tokens")
    if target_tokens is not None:
        train_cfg["max_steps"] = int(math.ceil(int(target_tokens) / max(global_batch_tokens, 1)))
        train_cfg["resolved_tokens"] = int(train_cfg["max_steps"]) * global_batch_tokens
        train_cfg["resolved_world_size"] = world_size

    output_dir = ensure_dir(config["train"]["output_dir"])
    tb_dir = ensure_dir(config["train"]["tensorboard_dir"])
    logs_dir = ensure_dir(args.log_dir if args.log_dir else Path(args.config).resolve().parents[1] / "logs")
    run_tag = timestamp_tag()
    metrics_path = logs_dir / f"train_metrics_{run_tag}.jsonl"
    csv_metrics_path = logs_dir / f"train_metrics_{run_tag}.csv"
    meta_path = logs_dir / f"train_meta_{run_tag}.json"
    perf_path = logs_dir / "train_perf_rank0.jsonl"
    moe_routing_stats_path = logs_dir / "moe_routing_stats.jsonl"
    ns_diagnostics_path = logs_dir / f"newton_schulz_diagnostics_{run_tag}.jsonl"

    if is_main_process():
        save_json(
            meta_path,
            {
                "config_path": str(Path(args.config).resolve()),
                "world_size": world_size,
                "rank": rank,
                "device": str(device),
                "deepspeed_config": args.deepspeed_config,
                "deepspeed_enabled": deepspeed_enabled,
                "resume_from_checkpoint": args.resume_from_checkpoint,
                "load_balance_bias": args.load_balance_bias,
                "save_balance_bias": args.save_balance_bias,
                "balance_bias_calibration": args.balance_bias_calibration,
                "skip_final_checkpoint": args.skip_final_checkpoint,
                "target_tokens": target_tokens,
                "tokens_per_step": global_batch_tokens,
                "resolved_max_steps": train_cfg["max_steps"],
                "resolved_tokens": int(train_cfg["max_steps"]) * global_batch_tokens,
            },
        )

    train_loader, valid_loader = build_dataloaders(config, world_size=world_size, rank=rank)

    model = DeepSeekV3LikeLM(config).to(device)
    if distributed and not deepspeed_enabled:
        ddp_find_unused = bool(
            train_cfg.get("ddp_find_unused_parameters", train_cfg.get("find_unused_parameters", False))
        )
        model = DDP(
            model,
            device_ids=[args.local_rank] if is_cuda else None,
            find_unused_parameters=ddp_find_unused,
        )

    optimizer = build_optimizer(model, train_cfg, deepspeed_enabled)
    scheduler = build_scheduler(
        optimizer=optimizer,
        warmup_steps=train_cfg["warmup_steps"],
        total_steps=train_cfg["max_steps"],
        min_lr_ratio=train_cfg["min_lr"] / train_cfg["learning_rate"],
    )

    if deepspeed_enabled:
        ds_config = build_deepspeed_config(args.deepspeed_config, train_cfg, world_size)
        model, optimizer, scheduler = initialize_deepspeed(model, optimizer, scheduler, ds_config)

    if args.load_balance_bias:
        load_balance_bias(args.load_balance_bias, model=model, rank=rank)

    maybe_print_run_diagnostics(
        config=config,
        ds_config=ds_config,
        rank=rank,
        world_size=world_size,
        distributed=distributed,
        device=device,
    )

    start_step = 0
    if args.resume_from_checkpoint:
        if deepspeed_enabled:
            start_step = load_deepspeed_checkpoint(args.resume_from_checkpoint, model)
        else:
            start_step = load_checkpoint(
                Path(args.resume_from_checkpoint),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
            )

    writer = SummaryWriter(log_dir=str(tb_dir)) if is_main_process() else None
    scaler_enabled = bool(train_cfg.get("bf16", False) and is_cuda and torch.cuda.is_bf16_supported())

    model.train()
    if not deepspeed_enabled:
        optimizer.zero_grad(set_to_none=True)
    else:
        model.zero_grad()
    iterator = iter(train_loader)
    step_time = time.perf_counter()
    grad_accum_steps = int(train_cfg["gradient_accumulation_steps"])
    perf_log_interval = int(train_cfg.get("perf_log_interval", train_cfg.get("log_steps", 10)))
    log_steps = int(train_cfg.get("log_steps", 1))
    stop_valid_lm_loss_below = train_cfg.get("stop_valid_lm_loss_below")
    if stop_valid_lm_loss_below is not None:
        stop_valid_lm_loss_below = float(stop_valid_lm_loss_below)
    last_step = start_step
    training_started_at = time.perf_counter()
    loss_thresholds = [float(value) for value in train_cfg.get("loss_thresholds", [4.0, 3.5, 3.2, 3.0])]
    tokens_to_loss: dict[str, int | None] = {str(value): None for value in loss_thresholds}
    csv_fields = [
        "step",
        "tokens",
        "loss",
        "validation_loss",
        "learning_rate",
        "grad_norm",
        "update_norm",
        "parameter_norm",
        "optimizer_time",
        "newton_schulz_time",
        "memory_usage",
        "peak_memory",
        "optimizer_states_memory",
        "tokens_per_sec",
        "step_time",
        "forward_time",
        "backward_time",
        "training_time",
    ]

    for step in range(start_step, train_cfg["max_steps"]):
        sync_cuda(device)
        full_step_start = time.perf_counter()
        accum_loss = 0.0
        accum_lm_loss = 0.0
        accum_mtp_loss = 0.0
        accum_aux_loss = 0.0
        accum_router_entropy = 0.0
        accum_load_var = 0.0
        data_time_total = 0.0
        forward_time_total = 0.0
        backward_time_total = 0.0
        optimizer_time_total = 0.0
        ddp_sync_micro_steps = 0
        moe_routing_summaries: list[dict[str, Any]] = []

        for micro_step in range(grad_accum_steps):
            sync_cuda(device)
            data_start = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                sampler = getattr(train_loader, "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(step)
                iterator = iter(train_loader)
                batch = next(iterator)
            batch = move_batch_to_device(batch, device)
            sync_cuda(device)
            data_time_total += time.perf_counter() - data_start

            autocast_enabled = scaler_enabled and not deepspeed_enabled
            if args.balance_bias_calibration:
                sync_cuda(device)
                forward_start = time.perf_counter()
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                        outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
                sync_cuda(device)
                forward_time_total += time.perf_counter() - forward_start
            elif not deepspeed_enabled:
                is_last_micro_step = micro_step == grad_accum_steps - 1
                ddp_sync_enabled = not (distributed and grad_accum_steps > 1 and not is_last_micro_step)
                sync_context = model.no_sync() if not ddp_sync_enabled else nullcontext()
                if ddp_sync_enabled:
                    ddp_sync_micro_steps += 1
                with sync_context:
                    sync_cuda(device)
                    forward_start = time.perf_counter()
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                        outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
                        loss = outputs["loss"] / grad_accum_steps
                    sync_cuda(device)
                    forward_time_total += time.perf_counter() - forward_start

                    backward_start = time.perf_counter()
                    loss.backward()
                    sync_cuda(device)
                    backward_time_total += time.perf_counter() - backward_start
            else:
                sync_cuda(device)
                forward_start = time.perf_counter()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
                sync_cuda(device)
                forward_time_total += time.perf_counter() - forward_start

                backward_start = time.perf_counter()
                model.backward(outputs["loss"])
                sync_cuda(device)
                backward_time_total += time.perf_counter() - backward_start

                optimizer_start = time.perf_counter()
                model.step()
                sync_cuda(device)
                optimizer_time_total += time.perf_counter() - optimizer_start
            accum_loss += float(outputs["loss"].item())
            accum_lm_loss += float(outputs["lm_loss"].item())
            accum_mtp_loss += float(outputs["mtp_loss"].item())
            accum_aux_loss += float(outputs["aux_loss"].item())
            accum_router_entropy += float(outputs["router_entropy"].item())
            accum_load_var += float(outputs["expert_load_variance"].item())
            apply_pending_balance_bias_updates(model)
            moe_routing_summary = summarize_moe_routing(outputs)
            if moe_routing_summary:
                moe_routing_summaries.append(moe_routing_summary)

        grad_norm = 0.0 if args.balance_bias_calibration else get_gradient_norm(model)
        if not deepspeed_enabled and not args.balance_bias_calibration:
            optimizer_start = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(train_cfg.get("gradient_clipping", 1.0)),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            sync_cuda(device)
            optimizer_time_total += time.perf_counter() - optimizer_start

        elapsed = max(time.perf_counter() - step_time, 1e-6)
        step_time = time.perf_counter()
        tokens_per_sec = global_batch_tokens / elapsed
        sync_cuda(device)
        full_step_time = max(time.perf_counter() - full_step_start, 1e-6)
        valid_metrics = None
        if (step + 1) % train_cfg["valid_steps"] == 0 or (step + 1) == int(train_cfg["max_steps"]):
            valid_metrics = evaluate(
                model=model,
                valid_loader=valid_loader,
                device=device,
                max_batches=train_cfg.get("valid_max_batches", 8),
            )

        if not args.balance_bias_calibration and (step + 1) % train_cfg["save_steps"] == 0 and is_main_process():
            if not deepspeed_enabled:
                ckpt_path = output_dir / f"step_{step + 1:06d}.pt"
                save_checkpoint(ckpt_path, model, optimizer, scheduler, step + 1, config)

        if not args.balance_bias_calibration and deepspeed_enabled and (step + 1) % train_cfg["save_steps"] == 0:
            save_deepspeed_checkpoint(
                output_dir=output_dir,
                model=model,
                step=step + 1,
                config=config,
                tag=f"step_{step + 1:06d}",
            )

        train_total_loss = accum_loss / train_cfg["gradient_accumulation_steps"]
        train_lm_loss = accum_lm_loss / train_cfg["gradient_accumulation_steps"]
        train_mtp_loss = accum_mtp_loss / train_cfg["gradient_accumulation_steps"]
        train_aux_loss = accum_aux_loss / train_cfg["gradient_accumulation_steps"]
        valid_total_loss = valid_metrics["loss"] if valid_metrics is not None else None
        valid_lm_loss = valid_metrics["lm_loss"] if valid_metrics is not None else None
        valid_mtp_loss = valid_metrics["mtp_loss"] if valid_metrics is not None else None
        valid_aux_loss = valid_metrics["aux_loss"] if valid_metrics is not None else None
        cumulative_tokens = (step + 1) * global_batch_tokens
        for threshold in loss_thresholds:
            key = str(threshold)
            if tokens_to_loss[key] is None and train_lm_loss <= threshold:
                tokens_to_loss[key] = cumulative_tokens

        instrumented_optimizer = find_instrumented_optimizer(optimizer)
        optimizer_metrics: dict[str, Any] = {}
        if instrumented_optimizer is not None:
            candidate = getattr(instrumented_optimizer, "last_step_metrics", {})
            if int(candidate.get("sample_step", -1)) == step + 1:
                optimizer_metrics = dict(candidate)
        peak_memory = float(torch.cuda.max_memory_allocated(device) if is_cuda else 0.0)
        cumulative_training_time = time.perf_counter() - training_started_at

        metrics = {
            "step": step + 1,
            "train/loss": train_total_loss,
            "train/total_loss": train_total_loss,
            "train/lm_loss": train_lm_loss,
            "train/mtp_loss": train_mtp_loss,
            "train/aux_loss": train_aux_loss,
            "train/ppl_lm": compute_ppl(train_lm_loss),
            "train/ppl_total": compute_ppl(train_total_loss),
            "train/grad_norm": grad_norm,
            "train/optimizer_grad_norm": optimizer_metrics.get("grad_norm"),
            "train/router_entropy": accum_router_entropy / train_cfg["gradient_accumulation_steps"],
            "train/expert_load_variance": accum_load_var / train_cfg["gradient_accumulation_steps"],
            "train/tokens_per_sec": tokens_per_sec,
            "train/step_time": full_step_time,
            "train/data_time": data_time_total,
            "train/forward_time": forward_time_total,
            "train/backward_time": backward_time_total,
            "train/optimizer_time": optimizer_time_total,
            "train/ddp_sync_micro_steps": ddp_sync_micro_steps,
            "train/lr": get_current_lr(optimizer, scheduler),
            "train/tokens": cumulative_tokens,
            "train/update_norm": optimizer_metrics.get("update_norm"),
            "train/parameter_norm": optimizer_metrics.get("parameter_norm"),
            "train/newton_schulz_time": optimizer_metrics.get("newton_schulz_time"),
            "train/muon_matrix_count": optimizer_metrics.get("muon_matrix_count", 0),
            "train/gpu_mem_allocated": float(torch.cuda.memory_allocated(device) if is_cuda else 0.0),
            "train/gpu_mem_reserved": float(torch.cuda.memory_reserved(device) if is_cuda else 0.0),
            "train/gpu_peak_memory": peak_memory,
            "train/optimizer_states_memory": optimizer_state_bytes(optimizer) if optimizer_metrics else None,
            "train/training_time": cumulative_training_time,
            "valid/loss": valid_total_loss,
            "valid/total_loss": valid_total_loss,
            "valid/lm_loss": valid_lm_loss,
            "valid/mtp_loss": valid_mtp_loss,
            "valid/aux_loss": valid_aux_loss,
            "valid/ppl_lm": compute_ppl(valid_lm_loss),
            "valid/ppl_total": compute_ppl(valid_total_loss),
        }
        last_step = step + 1

        perf_metrics = {
            "step": step + 1,
            "rank": rank,
            "world_size": world_size,
            "data_time": data_time_total,
            "forward_time": forward_time_total,
            "backward_time": backward_time_total,
            "optimizer_time": optimizer_time_total,
            "step_time": full_step_time,
            "tokens_per_step": global_batch_tokens,
            "tokens_per_second": global_batch_tokens / full_step_time,
            "gradient_accumulation_steps": grad_accum_steps,
            "ddp_sync_micro_steps": ddp_sync_micro_steps,
            "ranks_entered_step": world_size,
            "gpu_mem_allocated": float(torch.cuda.memory_allocated(device) if is_cuda else 0.0),
            "gpu_mem_reserved": float(torch.cuda.memory_reserved(device) if is_cuda else 0.0),
            "gpu_peak_memory": peak_memory,
        }
        moe_routing_summary = aggregate_moe_routing_summaries(moe_routing_summaries)

        should_log_perf = (step + 1) % perf_log_interval == 0 or (step + 1) == int(train_cfg["max_steps"])
        if distributed and should_log_perf:
            entered = torch.ones((), device=device, dtype=torch.int32)
            torch.distributed.all_reduce(entered, op=torch.distributed.ReduceOp.SUM)
            perf_metrics["ranks_entered_step"] = int(entered.item())
            if moe_routing_summary:
                moe_sum_scalars = torch.tensor(
                    [
                        float(moe_routing_summary.get("routed_assignments", 0.0)),
                        float(moe_routing_summary.get("kept_assignments", 0.0)),
                        float(moe_routing_summary.get("dropped_assignments", 0.0)),
                        float(moe_routing_summary.get("avg_capacity_utilization", 0.0)),
                        float(moe_routing_summary.get("avg_load_cv", 0.0)),
                        float(moe_routing_summary.get("avg_l_aux", 0.0)),
                        float(moe_routing_summary.get("moe_time_ms", 0.0)),
                        float(moe_routing_summary.get("moe_first_alltoall_time_ms", 0.0)),
                        float(moe_routing_summary.get("moe_second_alltoall_time_ms", 0.0)),
                        float(moe_routing_summary.get("router_time_ms", 0.0)),
                        float(moe_routing_summary.get("moe_dispatch_time_ms", 0.0)),
                        float(moe_routing_summary.get("expert_mlp_time_ms", 0.0)),
                        float(moe_routing_summary.get("moe_combine_time_ms", 0.0)),
                        float(moe_routing_summary.get("attention_time_ms", 0.0)),
                        float(moe_routing_summary.get("dense_mlp_time_ms", 0.0)),
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                moe_max_scalars = torch.tensor(
                    [
                        float(moe_routing_summary.get("max_capacity_utilization", 0.0)),
                        float(moe_routing_summary.get("max_load_cv", 0.0)),
                    ],
                    device=device,
                    dtype=torch.float64,
                )
                torch.distributed.all_reduce(moe_sum_scalars, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(moe_max_scalars, op=torch.distributed.ReduceOp.MAX)
                divisor = max(world_size, 1)
                routed_total = float(moe_sum_scalars[0].item())
                kept_total = float(moe_sum_scalars[1].item())
                dropped_total = float(moe_sum_scalars[2].item())
                perf_metrics["moe"] = {
                    "routed_assignments": routed_total,
                    "kept_assignments": kept_total,
                    "dropped_assignments": dropped_total,
                    "drop_rate": dropped_total / max(routed_total, 1.0),
                    "avg_capacity_utilization": float(moe_sum_scalars[3].item()) / divisor,
                    "max_capacity_utilization": float(moe_max_scalars[0].item()),
                    "avg_load_cv": float(moe_sum_scalars[4].item()) / divisor,
                    "max_load_cv": float(moe_max_scalars[1].item()),
                    "avg_l_aux": float(moe_sum_scalars[5].item()) / divisor,
                    "moe_time_ms": float(moe_sum_scalars[6].item()) / divisor,
                    "moe_first_alltoall_time_ms": float(moe_sum_scalars[7].item()) / divisor,
                    "moe_second_alltoall_time_ms": float(moe_sum_scalars[8].item()) / divisor,
                    "router_time_ms": float(moe_sum_scalars[9].item()) / divisor,
                    "moe_dispatch_time_ms": float(moe_sum_scalars[10].item()) / divisor,
                    "expert_mlp_time_ms": float(moe_sum_scalars[11].item()) / divisor,
                    "moe_combine_time_ms": float(moe_sum_scalars[12].item()) / divisor,
                    "attention_time_ms": float(moe_sum_scalars[13].item()) / divisor,
                    "dense_mlp_time_ms": float(moe_sum_scalars[14].item()) / divisor,
                    "rank0_layers": moe_routing_summary.get("layers", []),
                }
        elif moe_routing_summary:
            perf_metrics["moe"] = moe_routing_summary
        if should_log_perf:
            perf_metrics["deepspeed_wall_clock"] = get_deepspeed_wall_clock(model)

        if is_main_process():
            if (step + 1) % log_steps == 0:
                print(json.dumps(metrics, ensure_ascii=False))
            if should_log_perf:
                print(json.dumps({"perf": perf_metrics}, ensure_ascii=False))
                with open(perf_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(perf_metrics, ensure_ascii=False) + "\n")
                moe_payload = perf_metrics.get("moe") or {}
                with open(moe_routing_stats_path, "a", encoding="utf-8") as handle:
                    for layer_payload in moe_payload.get("rank0_layers", moe_payload.get("layers", [])) or []:
                        expert_counts = layer_payload.get("expert_counts") or []
                        pre_counts = layer_payload.get("pre_expert_counts") or []
                        dropped_counts = layer_payload.get("dropped_per_expert") or []
                        record = {
                            "run_name": output_dir.name,
                            "step": step + 1,
                            "rank": rank,
                            "layer_id": layer_payload.get("layer"),
                            "scope": layer_payload.get("scope"),
                            "num_tokens": layer_payload.get("num_tokens"),
                            "top_k": layer_payload.get("top_k"),
                            "num_experts": layer_payload.get("num_experts"),
                            "capacity_per_expert": layer_payload.get("capacity"),
                            "total_assignments": layer_payload.get("routed_assignments", sum(pre_counts)),
                            "dropped_assignments": layer_payload.get("dropped_assignments", sum(dropped_counts)),
                            "drop_rate": layer_payload.get("drop_rate"),
                            "expert_load_mean": layer_payload.get("expert_load_mean"),
                            "expert_load_max": layer_payload.get("expert_load_max"),
                            "expert_load_p95": layer_payload.get("expert_load_p95"),
                            "expert_load_std": layer_payload.get("expert_load_std"),
                            "max_load_over_mean": layer_payload.get("max_load_over_mean"),
                            "load_balance_cv": layer_payload.get("load_cv"),
                            "gate_logits_mean": layer_payload.get("gate_logits_mean"),
                            "gate_logits_std": layer_payload.get("gate_logits_std"),
                            "router_probability_entropy": layer_payload.get("router_entropy"),
                            "top1_expert_distribution": layer_payload.get("top1_expert_counts"),
                            "top8_expert_distribution": layer_payload.get("topk_expert_counts"),
                            "balance_bias": layer_payload.get("balance_bias"),
                            "expert_counts": expert_counts,
                            "dropped_per_expert": dropped_counts,
                            "aux_loss_l_aux": layer_payload.get("l_aux"),
                            "moe_aux_loss_coef": float(config.get("moe", {}).get("aux_loss_weight", 0.0)),
                            "attention_time_ms": moe_payload.get("attention_time_ms"),
                            "router_time_ms": layer_payload.get("router_time_ms"),
                            "moe_dispatch_time_ms": layer_payload.get("moe_dispatch_time_ms"),
                            "expert_mlp_time_ms": moe_payload.get("expert_mlp_time_ms"),
                            "moe_combine_time_ms": layer_payload.get("moe_combine_time_ms"),
                            "dense_mlp_time_ms": moe_payload.get("dense_mlp_time_ms"),
                            "all_to_all_time_ms": (
                                float(moe_payload.get("moe_first_alltoall_time_ms", 0.0))
                                + float(moe_payload.get("moe_second_alltoall_time_ms", 0.0))
                            ),
                            "zero_comm_time_ms": (perf_metrics.get("deepspeed_wall_clock") or {}).get(
                                "ds_backward_reduce_ms"
                            ),
                        }
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            with open(metrics_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            append_csv_row(
                csv_metrics_path,
                csv_fields,
                {
                    "step": step + 1,
                    "tokens": cumulative_tokens,
                    "loss": train_lm_loss,
                    "validation_loss": valid_lm_loss,
                    "learning_rate": metrics["train/lr"],
                    "grad_norm": grad_norm if grad_norm > 0.0 else optimizer_metrics.get("grad_norm"),
                    "update_norm": optimizer_metrics.get("update_norm"),
                    "parameter_norm": optimizer_metrics.get("parameter_norm"),
                    "optimizer_time": optimizer_time_total,
                    "newton_schulz_time": optimizer_metrics.get("newton_schulz_time"),
                    "memory_usage": metrics["train/gpu_mem_allocated"],
                    "peak_memory": peak_memory,
                    "optimizer_states_memory": metrics["train/optimizer_states_memory"],
                    "tokens_per_sec": global_batch_tokens / full_step_time,
                    "step_time": full_step_time,
                    "forward_time": forward_time_total,
                    "backward_time": backward_time_total,
                    "training_time": cumulative_training_time,
                },
            )
            if instrumented_optimizer is not None:
                diagnostics = getattr(instrumented_optimizer, "last_ns_diagnostics", [])
                if diagnostics and int(diagnostics[0].get("step", -1)) == step + 1:
                    with open(ns_diagnostics_path, "a", encoding="utf-8") as handle:
                        for diagnostic in diagnostics:
                            handle.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")
            if writer is not None:
                for key, value in metrics.items():
                    if value is None or key == "step":
                        continue
                    writer.add_scalar(key, value, step + 1)

        stop_now = False
        stop_reason = None
        if (
            stop_valid_lm_loss_below is not None
            and valid_lm_loss is not None
            and valid_lm_loss < stop_valid_lm_loss_below
        ):
            stop_now = True
            stop_reason = {
                "metric": "valid/lm_loss",
                "value": valid_lm_loss,
                "threshold": stop_valid_lm_loss_below,
            }
        if distributed:
            stop_tensor = torch.tensor(
                1 if (is_main_process() and stop_now) else 0,
                device=device,
                dtype=torch.int32,
            )
            torch.distributed.broadcast(stop_tensor, src=0)
            stop_now = bool(stop_tensor.item())
        if stop_now:
            if is_main_process():
                print(
                    json.dumps(
                        {
                            "early_stop": {
                                "step": step + 1,
                                **(stop_reason or {}),
                            }
                        },
                        ensure_ascii=False,
                    )
                )
            if deepspeed_enabled and not args.balance_bias_calibration:
                save_deepspeed_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    step=step + 1,
                    config=config,
                    tag=f"stop_lm_loss_step_{step + 1:06d}",
                )
            elif is_main_process() and not args.balance_bias_calibration:
                ckpt_path = output_dir / f"stop_lm_loss_step_{step + 1:06d}.pt"
                save_checkpoint(ckpt_path, model, optimizer, scheduler, step + 1, config)
            break

    if args.save_balance_bias:
        save_balance_bias(args.save_balance_bias, model=model, rank=rank, world_size=world_size)

    if is_main_process():
        save_json(
            logs_dir / f"token_efficiency_{run_tag}.json",
            {
                "target_tokens": target_tokens,
                "processed_tokens": last_step * global_batch_tokens,
                "tokens_to_loss": tokens_to_loss,
                "final_step": last_step,
            },
        )
        if not deepspeed_enabled:
            if not args.balance_bias_calibration and not args.skip_final_checkpoint:
                final_path = output_dir / "last.pt"
                save_checkpoint(final_path, model, optimizer, scheduler, last_step, config)
        if writer is not None:
            writer.close()

    if not args.balance_bias_calibration and deepspeed_enabled and not args.skip_final_checkpoint:
        save_deepspeed_checkpoint(
            output_dir=output_dir,
            model=model,
            step=last_step,
            config=config,
            tag="last",
        )

    cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
