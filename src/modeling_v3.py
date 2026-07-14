from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.mla import RMSNorm
from src.moe import DeepSeekV3MoE, SwiGLUExpert
from src.mtp import DeepSeekV3MTPModule
from src.v4_attention import (
    DeepseekV4Attention,
    DeepseekV4AttentionConfig,
    DeepseekV4RotaryEmbedding,
    build_sliding_window_causal_mask,
)


@dataclass
class ModelShape:
    vocab_size: int
    seq_len: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    dense_intermediate_size: int
    first_dense_layers: int
    rms_norm_eps: float
    tie_word_embeddings: bool


def _get_any(payload: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return default


class DenseSwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.expert = SwiGLUExpert(hidden_size, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.expert(hidden_states)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        model_cfg: ModelShape,
        attn_cfg: DeepseekV4AttentionConfig,
        moe_cfg: dict[str, Any],
        mlp_layer_type: str = "moe",
        force_dense_ffn: bool = False,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.mlp_layer_type = mlp_layer_type
        self.use_dense_ffn = force_dense_ffn or mlp_layer_type == "dense"
        self.collect_timing = attn_cfg.collect_timing

        self.attn_norm = RMSNorm(model_cfg.hidden_size, model_cfg.rms_norm_eps)
        attn_module = DeepseekV4Attention(attn_cfg, layer_idx)
        if attn_cfg.compile_attention and hasattr(torch, "compile"):
            compile_kwargs = {}
            if attn_cfg.compile_mode:
                compile_kwargs["mode"] = attn_cfg.compile_mode
            attn_module = torch.compile(attn_module, **compile_kwargs)
        self.attn = attn_module

        self.ffn_norm = RMSNorm(model_cfg.hidden_size, model_cfg.rms_norm_eps)
        if self.use_dense_ffn:
            self.ffn = DenseSwiGLUFFN(model_cfg.hidden_size, model_cfg.dense_intermediate_size)
        else:
            self.ffn = DeepSeekV3MoE(
                hidden_size=model_cfg.hidden_size,
                expert_intermediate_size=moe_cfg["expert_intermediate_size"],
                num_routed_experts=moe_cfg["num_routed_experts"],
                num_shared_experts=moe_cfg["num_shared_experts"],
                top_k=moe_cfg["top_k"],
                aux_loss_weight=moe_cfg["aux_loss_weight"],
                normalize_topk_prob=moe_cfg["normalize_topk_prob"],
                num_expert_groups=moe_cfg["num_expert_groups"],
                num_limited_groups=moe_cfg["num_limited_groups"],
                route_scale=moe_cfg["route_scale"],
                score_function=moe_cfg["score_function"],
                use_correction_bias=moe_cfg["use_correction_bias"],
                balance_bias=moe_cfg["balance_bias"],
                balance_bias_lr=moe_cfg["balance_bias_lr"],
                balance_bias_clamp=moe_cfg["balance_bias_clamp"],
                implementation=moe_cfg["implementation"],
                moe_ep_size=moe_cfg["moe_ep_size"],
                moe_capacity_factor=moe_cfg["moe_capacity_factor"],
                moe_eval_capacity_factor=moe_cfg["moe_eval_capacity_factor"],
                moe_min_capacity=moe_cfg["moe_min_capacity"],
                moe_drop_tokens=moe_cfg["moe_drop_tokens"],
                moe_drop_policy=moe_cfg["moe_drop_policy"],
                moe_use_rts=moe_cfg["moe_use_rts"],
                moe_use_tutel=moe_cfg["moe_use_tutel"],
                swiglu_limit=moe_cfg["swiglu_limit"],
                use_packed_experts=moe_cfg["use_packed_experts"],
                router_type=mlp_layer_type,
                vocab_size=model_cfg.vocab_size,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_ids: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], None]:
        del use_cache
        if position_embeddings is None or position_ids is None:
            raise ValueError("DeepSeek-V4 attention requires position_embeddings and position_ids.")
        attn_input = self.attn_norm(hidden_states)
        attention_start = time.perf_counter() if self.collect_timing else 0.0
        attn_output, _ = self.attn(
            attn_input,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        attention_time_ms = (time.perf_counter() - attention_start) * 1000.0 if self.collect_timing else 0.0
        hidden_states = hidden_states + attn_output

        ffn_input = self.ffn_norm(hidden_states)
        ffn_start = time.perf_counter() if self.collect_timing else 0.0
        if self.use_dense_ffn:
            ffn_output = self.ffn(ffn_input)
            dense_mlp_time_ms = (time.perf_counter() - ffn_start) * 1000.0 if self.collect_timing else 0.0
            aux_loss = hidden_states.new_zeros(())
            stats = {
                "router_entropy": hidden_states.new_zeros(()),
                "expert_load_variance": hidden_states.new_zeros(()),
                "aux_loss": hidden_states.new_zeros(()),
            }
            if self.collect_timing:
                stats["dense_mlp_time"] = hidden_states.new_tensor(dense_mlp_time_ms)
        else:
            ffn_output, aux_loss, stats = self.ffn(ffn_input, input_ids=input_ids)
            if self.collect_timing:
                stats["dense_mlp_time"] = hidden_states.new_zeros(())
        if self.collect_timing:
            stats["attention_time"] = hidden_states.new_tensor(attention_time_ms)
        hidden_states = hidden_states + ffn_output
        return hidden_states, aux_loss, stats, None


class DeepSeekV3TinyLM(nn.Module):
    """Scaled-down DeepSeek-V3 architecture for local training."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        data_cfg = config.get("data", {})
        if "vocab_size" in data_cfg and int(model_cfg["vocab_size"]) != int(data_cfg["vocab_size"]):
            raise ValueError(
                f"model.vocab_size={int(model_cfg['vocab_size'])} does not match "
                f"data.vocab_size={int(data_cfg['vocab_size'])}."
            )
        self.model_cfg = ModelShape(
            vocab_size=int(model_cfg["vocab_size"]),
            seq_len=int(model_cfg["seq_len"]),
            num_layers=int(model_cfg["num_layers"]),
            hidden_size=int(model_cfg["hidden_size"]),
            num_attention_heads=int(model_cfg["num_attention_heads"]),
            dense_intermediate_size=int(
                _get_any(model_cfg, ("dense_intermediate_size", "intermediate_size"))
            ),
            first_dense_layers=int(_get_any(model_cfg, ("first_dense_layers", "n_dense_layers"), 3)),
            rms_norm_eps=float(model_cfg["rms_norm_eps"]),
            tie_word_embeddings=bool(model_cfg["tie_word_embeddings"]),
        )
        self.mtp_cfg = config.get("mtp", {})
        self.activation_checkpointing = bool(config.get("train", {}).get("activation_checkpointing", False))
        mtp_depth = int(_get_any(self.mtp_cfg, ("mtp_depth", "depth"), 1)) if self.mtp_cfg.get("enabled", False) else 0
        self.v4_attn_cfg = self._normalize_v4_attention_config(
            config.get("v4_attention", {}),
            config.get("mla", {}),
            num_main_layers=self.model_cfg.num_layers,
            mtp_depth=mtp_depth,
        )
        self.moe_cfg = self._normalize_moe_config(config.get("moe", {}), model_cfg)
        self.mlp_layer_types = self._normalize_mlp_layer_types(
            config.get("moe", {}),
            num_main_layers=self.model_cfg.num_layers,
            mtp_depth=mtp_depth,
        )

        self.embed_tokens = nn.Embedding(self.model_cfg.vocab_size, self.model_cfg.hidden_size)
        self.rotary_emb = DeepseekV4RotaryEmbedding(self.v4_attn_cfg)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    layer_idx=idx,
                    model_cfg=self.model_cfg,
                    attn_cfg=self.v4_attn_cfg,
                    moe_cfg=self.moe_cfg,
                    mlp_layer_type=self.mlp_layer_types[idx],
                )
                for idx in range(self.model_cfg.num_layers)
            ]
        )
        self.final_norm = RMSNorm(self.model_cfg.hidden_size, self.model_cfg.rms_norm_eps)
        self.lm_head = nn.Linear(self.model_cfg.hidden_size, self.model_cfg.vocab_size, bias=False)
        if self.model_cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.mtp_modules = nn.ModuleList()
        if self.mtp_cfg.get("enabled", False):
            for depth_idx in range(mtp_depth):
                block = TransformerBlock(
                    layer_idx=self.model_cfg.num_layers + depth_idx,
                    model_cfg=self.model_cfg,
                    attn_cfg=self.v4_attn_cfg,
                    moe_cfg=self.moe_cfg,
                    mlp_layer_type=self.mlp_layer_types[self.model_cfg.num_layers + depth_idx],
                    force_dense_ffn=not bool(self.mtp_cfg.get("mtp_use_moe", True)),
                )
                block.activation_checkpointing = self.activation_checkpointing
                self.mtp_modules.append(
                    DeepSeekV3MTPModule(
                        hidden_size=self.model_cfg.hidden_size,
                        block=block,
                        norm_factory=lambda hidden_size: RMSNorm(hidden_size, self.model_cfg.rms_norm_eps),
                    )
                )

    def _default_v4_layer_types(self, num_main_layers: int, mtp_depth: int) -> list[str]:
        main_layer_types = ["heavily_compressed_attention"] * min(num_main_layers, 2)
        main_layer_types.extend(
            "compressed_sparse_attention" if idx % 2 == 0 else "heavily_compressed_attention"
            for idx in range(max(num_main_layers - 2, 0))
        )
        return main_layer_types + ["sliding_attention"] * mtp_depth

    def _default_o_groups(self) -> int:
        for candidate in (16, 8, 4, 2, 1):
            if candidate <= self.model_cfg.num_attention_heads and self.model_cfg.num_attention_heads % candidate == 0:
                return candidate
        return 1

    def _normalize_v4_attention_config(
        self,
        attn_cfg: dict[str, Any],
        legacy_mla_cfg: dict[str, Any],
        num_main_layers: int,
        mtp_depth: int,
    ) -> DeepseekV4AttentionConfig:
        total_attention_layers = num_main_layers + mtp_depth
        head_dim = int(
            _get_any(
                attn_cfg,
                ("head_dim",),
                _get_any(legacy_mla_cfg, ("v_head_dim",), self.model_cfg.hidden_size // self.model_cfg.num_attention_heads),
            )
        )
        q_lora_rank = int(_get_any(attn_cfg, ("q_lora_rank",), legacy_mla_cfg.get("q_lora_rank", head_dim * 4)))
        qk_rope_head_dim = _get_any(
            attn_cfg,
            ("qk_rope_head_dim", "rope_head_dim"),
            _get_any(legacy_mla_cfg, ("qk_rope_head_dim", "rope_dim"), max(2, head_dim // 8)),
        )
        partial_rotary_factor = _get_any(attn_cfg, ("partial_rotary_factor",), None)
        if partial_rotary_factor is None:
            partial_rotary_factor = int(qk_rope_head_dim) / head_dim
        rope_head_dim = int(head_dim * float(partial_rotary_factor))
        if rope_head_dim <= 0 or rope_head_dim % 2 != 0:
            raise ValueError(f"DeepSeek-V4 qk_rope_head_dim must be a positive even value, got {rope_head_dim}.")

        layer_types = attn_cfg.get("layer_types")
        if layer_types is None:
            layer_types = self._default_v4_layer_types(num_main_layers, mtp_depth)
        layer_types = list(layer_types)
        if len(layer_types) != total_attention_layers:
            raise ValueError(
                f"v4_attention.layer_types length {len(layer_types)} must equal "
                f"main layers + MTP depth ({total_attention_layers})."
            )

        compress_rates = dict(
            attn_cfg.get(
                "compress_rates",
                {"compressed_sparse_attention": 4, "heavily_compressed_attention": 128},
            )
        )
        rope_theta = float(attn_cfg.get("rope_theta", legacy_mla_cfg.get("rope_theta", 10000.0)))
        compress_rope_theta = float(attn_cfg.get("compress_rope_theta", 160000.0))
        rope_parameters = attn_cfg.get("rope_parameters")
        if isinstance(rope_parameters, dict) and isinstance(rope_parameters.get("main"), dict):
            rope_parameters = {"main": rope_parameters["main"], "compress": rope_parameters["compress"]}
        else:
            rope_parameters = {
                "main": {
                    "rope_type": "default",
                    "rope_theta": rope_theta,
                    "partial_rotary_factor": partial_rotary_factor,
                },
                "compress": {
                    "rope_type": "default",
                    "rope_theta": compress_rope_theta,
                    "partial_rotary_factor": partial_rotary_factor,
                },
            }

        o_groups = int(attn_cfg.get("o_groups", self._default_o_groups()))
        o_lora_rank = int(attn_cfg.get("o_lora_rank", head_dim))
        if (self.model_cfg.num_attention_heads * head_dim) % o_groups != 0:
            raise ValueError("num_attention_heads * head_dim must be divisible by v4_attention.o_groups.")
        if (o_groups * o_lora_rank) <= 0:
            raise ValueError("v4_attention.o_lora_rank and o_groups must be positive.")

        index_head_dim = int(attn_cfg.get("index_head_dim", max(rope_head_dim, head_dim // 4)))
        if index_head_dim < rope_head_dim:
            raise ValueError("v4_attention.index_head_dim must be >= qk_rope_head_dim for V4 indexer RoPE.")

        return DeepseekV4AttentionConfig(
            hidden_size=self.model_cfg.hidden_size,
            num_hidden_layers=total_attention_layers,
            num_attention_heads=self.model_cfg.num_attention_heads,
            head_dim=head_dim,
            q_lora_rank=q_lora_rank,
            rms_norm_eps=self.model_cfg.rms_norm_eps,
            max_position_embeddings=int(attn_cfg.get("max_position_embeddings", self.model_cfg.seq_len)),
            rope_parameters=rope_parameters,
            layer_types=layer_types,
            compress_rates=compress_rates,
            sliding_window=int(attn_cfg.get("sliding_window", 128)),
            attention_dropout=float(attn_cfg.get("attention_dropout", 0.0)),
            o_groups=o_groups,
            o_lora_rank=o_lora_rank,
            index_n_heads=int(attn_cfg.get("index_n_heads", self.model_cfg.num_attention_heads)),
            index_head_dim=index_head_dim,
            index_topk=int(attn_cfg.get("index_topk", 512)),
            _attn_implementation=str(attn_cfg.get("_attn_implementation", "sdpa")),
            compile_attention=bool(attn_cfg.get("compile_attention", False)),
            compile_mode=attn_cfg.get("compile_mode"),
            collect_timing=bool(attn_cfg.get("collect_timing", False)),
        )

    def _normalize_moe_config(self, moe_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> dict[str, Any]:
        top_k = int(_get_any(moe_cfg, ("num_experts_per_token", "top_k"), 8))
        return {
            "expert_intermediate_size": int(
                _get_any(moe_cfg, ("expert_intermediate_size", "moe_intermediate_size"))
            ),
            "num_routed_experts": int(moe_cfg["num_routed_experts"]),
            "num_shared_experts": int(moe_cfg.get("num_shared_experts", 1)),
            "top_k": top_k,
            "aux_loss_weight": float(moe_cfg.get("aux_loss_weight", 0.0)),
            "normalize_topk_prob": bool(_get_any(moe_cfg, ("normalize_topk_prob", "norm_topk_prob"), True)),
            "num_expert_groups": int(_get_any(moe_cfg, ("num_expert_groups", "n_group"), 8)),
            "num_limited_groups": int(_get_any(moe_cfg, ("num_limited_groups", "topk_group"), 4)),
            "route_scale": float(_get_any(moe_cfg, ("route_scale", "routed_scaling_factor"), 2.5)),
            "score_function": str(moe_cfg.get("score_function", "sigmoid")),
            "use_correction_bias": bool(moe_cfg.get("use_correction_bias", True)),
            "balance_bias": bool(moe_cfg.get("balance_bias", False)),
            "balance_bias_lr": float(moe_cfg.get("balance_bias_lr", 1.0e-3)),
            "balance_bias_clamp": float(moe_cfg.get("balance_bias_clamp", 5.0)),
            "implementation": str(moe_cfg.get("implementation", "torch")),
            "moe_ep_size": int(moe_cfg.get("moe_ep_size", 1)),
            "moe_capacity_factor": float(moe_cfg.get("moe_capacity_factor", 1.25)),
            "moe_eval_capacity_factor": float(moe_cfg.get("moe_eval_capacity_factor", 2.0)),
            "moe_min_capacity": int(moe_cfg.get("moe_min_capacity", 4)),
            "moe_drop_tokens": bool(moe_cfg.get("moe_drop_tokens", False)),
            "moe_drop_policy": str(moe_cfg.get("moe_drop_policy", "probs")),
            "moe_use_rts": bool(moe_cfg.get("moe_use_rts", True)),
            "moe_use_tutel": bool(moe_cfg.get("moe_use_tutel", False)),
            "swiglu_limit": None if moe_cfg.get("swiglu_limit", 10.0) is None else float(moe_cfg.get("swiglu_limit", 10.0)),
            "use_packed_experts": bool(moe_cfg.get("use_packed_experts", True)),
        }

    def _normalize_mlp_layer_types(
        self,
        moe_cfg: dict[str, Any],
        num_main_layers: int,
        mtp_depth: int,
    ) -> list[str]:
        total_layers = num_main_layers + mtp_depth
        layer_types = moe_cfg.get("mlp_layer_types")
        if layer_types is None:
            num_hash_layers = int(moe_cfg.get("num_hash_layers", 3))
            first_dense_layers = int(self.model_cfg.first_dense_layers)
            dense_count = min(num_main_layers, max(first_dense_layers, 0))
            remaining_layers = max(num_main_layers - dense_count, 0)
            hash_count = min(remaining_layers, max(num_hash_layers, 0))
            main_types = ["dense"] * dense_count
            main_types.extend(["hash_moe"] * hash_count)
            main_types.extend(["moe"] * max(num_main_layers - len(main_types), 0))
            layer_types = main_types + ["moe"] * mtp_depth
        layer_types = list(layer_types)
        if len(layer_types) == num_main_layers and mtp_depth:
            layer_types.extend(["moe"] * mtp_depth)
        if len(layer_types) != total_layers:
            raise ValueError(
                f"moe.mlp_layer_types length {len(layer_types)} must equal "
                f"main layers + MTP depth ({total_layers})."
            )
        allowed = {"dense", "hash_moe", "moe"}
        invalid = sorted(set(layer_types) - allowed)
        if invalid:
            raise ValueError(f"Unsupported moe.mlp_layer_types values: {invalid}")
        return layer_types

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        past_key_values: Any | None = None,
    ) -> dict[str, Any]:
        if use_cache or past_key_values is not None:
            raise NotImplementedError("The V4-attention tiny training fork does not expose generation cache yet.")

        hidden_states = self.embed_tokens(input_ids)
        batch, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        position_embeddings = {
            "main": self.rotary_emb(hidden_states, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(hidden_states, position_ids=position_ids, layer_type="compress"),
        }
        causal_mask = build_sliding_window_causal_mask(
            config=self.v4_attn_cfg,
            position_ids=position_ids,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
            attention_mask=attention_mask,
        )

        aux_losses = []
        router_entropies = []
        load_variances = []
        moe_routing_stats: list[dict[str, Any]] = []
        moe_timing_stats: list[dict[str, torch.Tensor]] = []
        block_timing_stats: list[dict[str, torch.Tensor]] = []
        collect_timing = self.v4_attn_cfg.collect_timing

        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                hidden_states, aux_loss, stats, _ = checkpoint(
                    lambda layer_hidden, block=layer: block(
                        hidden_states=layer_hidden,
                        attention_mask=causal_mask,
                        position_embeddings=position_embeddings,
                        position_ids=position_ids,
                        input_ids=input_ids,
                        use_cache=False,
                    ),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states, aux_loss, stats, _ = layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                    input_ids=input_ids,
                    use_cache=False,
                )
            aux_losses.append(aux_loss)
            router_entropies.append(stats["router_entropy"])
            load_variances.append(stats["expert_load_variance"])
            if collect_timing:
                block_timing_stats.append(
                    {
                        "layer": torch.tensor(float(layer.layer_idx), device=hidden_states.device),
                        "scope": "main",
                        "attention_time": stats.get("attention_time", hidden_states.new_zeros(())),
                        "dense_mlp_time": stats.get("dense_mlp_time", hidden_states.new_zeros(())),
                    }
                )
            if not layer.use_dense_ffn:
                routing = stats.get("moe_routing")
                if routing is not None:
                    moe_routing_stats.append({"layer": layer.layer_idx, "scope": "main", **routing})
                if collect_timing:
                    moe_timing_stats.append(
                        {
                            "layer": torch.tensor(float(layer.layer_idx), device=hidden_states.device),
                            "moe_time": stats.get("moe_time", hidden_states.new_zeros(())),
                            "first_alltoall_time": stats.get("moe_first_alltoall_time", hidden_states.new_zeros(())),
                            "second_alltoall_time": stats.get("moe_second_alltoall_time", hidden_states.new_zeros(())),
                            "router_time": stats.get("router_time", hidden_states.new_zeros(())),
                            "moe_dispatch_time": stats.get("moe_dispatch_time", hidden_states.new_zeros(())),
                            "expert_mlp_time": stats.get("expert_mlp_time", hidden_states.new_zeros(())),
                            "moe_combine_time": stats.get("moe_combine_time", hidden_states.new_zeros(())),
                        }
                    )

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        lm_loss = hidden_states.new_zeros(())
        mtp_loss = hidden_states.new_zeros(())
        aux_loss = hidden_states.new_zeros(())
        total_loss = hidden_states.new_zeros(())

        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )

            mtp_losses = []
            mtp_hidden = hidden_states
            for depth_idx, mtp_module in enumerate(self.mtp_modules):
                mtp_result = mtp_module(
                    hidden_states=mtp_hidden,
                    labels=labels,
                    depth_idx=depth_idx,
                    embed_tokens=self.embed_tokens,
                    lm_head=self.lm_head,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    position_ids=position_ids,
                )
                mtp_losses.append(mtp_result["loss"])
                mtp_hidden = mtp_result["hidden_states"]
                aux_losses.append(mtp_result["aux_loss"])
                router_entropies.append(mtp_result["stats"]["router_entropy"])
                load_variances.append(mtp_result["stats"]["expert_load_variance"])
                if collect_timing:
                    block_timing_stats.append(
                        {
                            "layer": torch.tensor(
                                float(self.model_cfg.num_layers + depth_idx),
                                device=hidden_states.device,
                            ),
                            "scope": f"mtp{depth_idx}",
                            "attention_time": mtp_result["stats"].get("attention_time", hidden_states.new_zeros(())),
                            "dense_mlp_time": mtp_result["stats"].get("dense_mlp_time", hidden_states.new_zeros(())),
                        }
                    )
                routing = mtp_result["stats"].get("moe_routing")
                if routing is not None:
                    moe_routing_stats.append(
                        {
                            "layer": self.model_cfg.num_layers + depth_idx,
                            "scope": f"mtp{depth_idx}",
                            **routing,
                        }
                    )
                if collect_timing:
                    moe_timing_stats.append(
                        {
                            "layer": torch.tensor(
                                float(self.model_cfg.num_layers + depth_idx),
                                device=hidden_states.device,
                            ),
                            "moe_time": mtp_result["stats"].get("moe_time", hidden_states.new_zeros(())),
                            "first_alltoall_time": mtp_result["stats"].get(
                                "moe_first_alltoall_time", hidden_states.new_zeros(())
                            ),
                            "second_alltoall_time": mtp_result["stats"].get(
                                "moe_second_alltoall_time", hidden_states.new_zeros(())
                            ),
                            "router_time": mtp_result["stats"].get("router_time", hidden_states.new_zeros(())),
                            "moe_dispatch_time": mtp_result["stats"].get(
                                "moe_dispatch_time",
                                hidden_states.new_zeros(()),
                            ),
                            "expert_mlp_time": mtp_result["stats"].get("expert_mlp_time", hidden_states.new_zeros(())),
                            "moe_combine_time": mtp_result["stats"].get(
                                "moe_combine_time",
                                hidden_states.new_zeros(()),
                            ),
                        }
                    )

            if mtp_losses:
                mtp_loss = torch.stack([loss.float() for loss in mtp_losses]).mean().to(hidden_states.dtype)

        if aux_losses:
            aux_loss = torch.stack([loss.float() for loss in aux_losses]).mean().to(hidden_states.dtype)

        if labels is not None:
            total_loss = lm_loss + self.mtp_cfg.get("mtp_loss_weight", 0.0) * mtp_loss + aux_loss

        router_entropy = torch.stack([value.float() for value in router_entropies]).mean().to(hidden_states.dtype)
        expert_load_variance = torch.stack([value.float() for value in load_variances]).mean().to(hidden_states.dtype)

        return {
            "logits": logits,
            "loss": total_loss,
            "lm_loss": lm_loss,
            "mtp_loss": mtp_loss,
            "aux_loss": aux_loss,
            "router_entropy": router_entropy,
            "expert_load_variance": expert_load_variance,
            "moe_routing_stats": moe_routing_stats,
            "moe_timing_stats": moe_timing_stats,
            "block_timing_stats": block_timing_stats,
            "past_key_values": None,
        }


DeepSeekV3LikeLM = DeepSeekV3TinyLM
