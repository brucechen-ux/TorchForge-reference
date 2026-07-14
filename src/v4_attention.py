from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func
except Exception:
    flash_attn_func = None


@dataclass
class DeepseekV4AttentionConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    q_lora_rank: int
    rms_norm_eps: float
    max_position_embeddings: int
    rope_parameters: dict[str, dict[str, Any]]
    layer_types: list[str]
    compress_rates: dict[str, int]
    sliding_window: int
    attention_dropout: float
    o_groups: int
    o_lora_rank: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    _attn_implementation: str = "sdpa"
    compile_attention: bool = False
    compile_mode: str | None = None
    collect_timing: bool = False


class DeepseekV4RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class DeepseekV4UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


class DeepseekV4RotaryEmbedding(nn.Module):
    """DeepSeek-V4 interleaved partial RoPE.

    This mirrors Hugging Face's DeepseekV4RotaryEmbedding default-RoPE path:
    cos/sin are half-sized, one value per interleaved pair, and
    apply_rotary_pos_emb expands them next to the rotation math.
    """

    inv_freq: torch.Tensor

    def __init__(self, config: DeepseekV4AttentionConfig) -> None:
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.layer_types = [k for k, v in config.rope_parameters.items() if isinstance(v, dict)]
        self.rope_type: dict[str, str] = {}
        for layer_type in self.layer_types:
            rope_params = config.rope_parameters[layer_type]
            self.rope_type[layer_type] = str(rope_params.get("rope_type", rope_params.get("type", "default")))
            if self.rope_type[layer_type] != "default":
                raise NotImplementedError(
                    "This local V4 attention port currently supports DeepSeek-V4 default RoPE only."
                )
            inv_freq, attention_scaling = self.compute_default_rope_parameters(config, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: DeepseekV4AttentionConfig,
        device: torch.device | None = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, float]:
        del seq_len
        if layer_type is None:
            raise ValueError("layer_type is required for DeepSeek-V4 RoPE.")
        base = config.rope_parameters[layer_type]["rope_theta"]
        partial_rotary_factor = config.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)
        dim = int(config.head_dim * partial_rotary_factor)
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"DeepSeek-V4 RoPE dim must be a positive even value, got {dim}.")
        inv_freq = 1.0 / (
            float(base)
            ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32) / dim)
        )
        return inv_freq, 1.0

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        layer_type: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_type is None:
            raise ValueError("layer_type is required for DeepSeek-V4 RoPE.")
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        cos = freqs.cos() * attention_scaling
        sin = freqs.sin() * attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DeepseekV4GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear used by DeepSeek-V4 grouped output projection."""

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False) -> None:
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, self.n_groups, -1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    """V4 interleaved RoPE applied to the trailing rope slice of x."""
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = ((rope.float() * cos) + (rotate_half(rope).float() * sin)).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


class DeepseekV4SlidingCache:
    def __init__(self, config: DeepseekV4AttentionConfig) -> None:
        self.sliding_window = config.sliding_window
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.cumulative_length = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args: Any, **kwargs: Any):
        del value_states, args, kwargs
        if self.keys is None:
            self.keys = key_states[:, :, :0, :]
            self.values = self.keys
        self.cumulative_length += key_states.shape[-2]
        full = torch.cat([self.keys, key_states], dim=-2)
        self.keys = full[:, :, -self.sliding_window + 1 :, :]
        self.values = self.keys
        return full, full


class DeepseekV4HCACache(DeepseekV4SlidingCache):
    layer_type = "heavily_compressed_attention"

    def __init__(self, config: DeepseekV4AttentionConfig) -> None:
        super().__init__(config)
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.buffer_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.buffer_gate: dict[str, torch.Tensor | None] = {"compressor": None}
        self.compressed_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.entry_count: dict[str, int] = {"compressor": 0}

    def store_compression_weights(
        self,
        name: str,
        kv: torch.Tensor,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        first_window_position = self.entry_count[name] * self.compress_rate
        buffered_kv, buffered_gate = self.buffer_kv[name], self.buffer_gate[name]
        if buffered_kv is not None and buffered_kv.shape[1]:
            kv = torch.cat([buffered_kv, kv], dim=1)
            gate = torch.cat([buffered_gate, gate], dim=1)
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        self.buffer_kv[name], self.buffer_gate[name] = kv[:, usable:], gate[:, usable:]
        return kv[:, :usable], gate[:, :usable], first_window_position

    def update_compressor_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
        if self.compressed_kv[name] is None:
            self.compressed_kv[name] = compressed
        elif compressed.shape[1] > 0:
            self.compressed_kv[name] = torch.cat([self.compressed_kv[name], compressed], dim=1)
        self.entry_count[name] += compressed.shape[1]
        return self.compressed_kv[name]


class DeepseekV4CSACache(DeepseekV4HCACache):
    layer_type = "compressed_sparse_attention"

    def __init__(self, config: DeepseekV4AttentionConfig) -> None:
        super().__init__(config)
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.buffer_kv["indexer"] = None
        self.buffer_gate["indexer"] = None
        self.compressed_kv["indexer"] = None
        self.entry_count["indexer"] = 0
        self.overlap_kv: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}
        self.overlap_gate: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}

    def update_overlap_state(
        self,
        name: str,
        chunk_kv: torch.Tensor,
        chunk_gate: torch.Tensor,
        head_dim: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        prior_kv, prior_gate = self.overlap_kv[name], self.overlap_gate[name]
        self.overlap_kv[name] = chunk_kv[:, -1, :, :head_dim].clone()
        self.overlap_gate[name] = chunk_gate[:, -1, :, :head_dim].clone()
        return prior_kv, prior_gate


class DeepseekV4DynamicCache:
    def __init__(self, config: DeepseekV4AttentionConfig) -> None:
        self.layers = []
        for layer_type in config.layer_types:
            if layer_type == "compressed_sparse_attention":
                self.layers.append(DeepseekV4CSACache(config))
            elif layer_type == "heavily_compressed_attention":
                self.layers.append(DeepseekV4HCACache(config))
            else:
                self.layers.append(DeepseekV4SlidingCache(config))

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int):
        return self.layers[layer_idx].update(key_states, value_states)

    def get_seq_length(self) -> int:
        if not self.layers:
            return 0
        return int(self.layers[0].cumulative_length)


class DeepseekV4HCACompressor(nn.Module):
    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4AttentionConfig) -> None:
        super().__init__()
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(self.compress_rate, self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: DeepseekV4DynamicCache | None,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del q_residual
        batch, _, _ = hidden_states.shape
        cache_layer: DeepseekV4HCACache | None = past_key_values.layers[layer_idx] if past_key_values is not None else None
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)
        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("compressor", kv, gate)

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + self.position_bias
            compressed = self.kv_norm(
                (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = (positions * self.compress_rate + first_window_position).unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        if cache_layer is not None:
            compressed = cache_layer.update_compressor_states("compressor", compressed)
        compressed_kv = compressed.unsqueeze(1)

        compressed_len = compressed_kv.shape[2]
        seq_len = position_ids.shape[1]
        if seq_len == 1 or compressed_len == 0:
            return compressed_kv, None

        entry_indices = torch.arange(compressed_len, device=compressed_kv.device)
        causal_threshold = (position_ids + 1) // self.compress_rate
        block_bias = compressed_kv.new_zeros((batch, 1, seq_len, compressed_len))
        block_bias = block_bias.masked_fill(
            entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
            float("-inf"),
        )
        return compressed_kv, block_bias


class DeepseekV4IndexerScorer(nn.Module):
    def __init__(self, config: DeepseekV4AttentionConfig) -> None:
        super().__init__()
        self.softmax_scale = config.index_head_dim**-0.5
        self.weights_scaling = config.index_n_heads**-0.5
        self.weights_proj = nn.Linear(config.hidden_size, config.index_n_heads, bias=False)

    def forward(self, q: torch.Tensor, compressed_kv: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * self.weights_scaling
        return (scores * weights.unsqueeze(-1)).sum(dim=2)


class DeepseekV4Indexer(nn.Module):
    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4AttentionConfig) -> None:
        super().__init__()
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.scorer = DeepseekV4IndexerScorer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: DeepseekV4DynamicCache | None,
        layer_idx: int,
    ) -> torch.LongTensor:
        batch, seq_len, _ = hidden_states.shape
        cache_layer: DeepseekV4CSACache | None = past_key_values.layers[layer_idx] if past_key_values is not None else None
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)

        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("indexer", kv, gate)

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
            if cache_layer is not None:
                prior_kv, prior_gate = cache_layer.update_overlap_state("indexer", chunk_kv, chunk_gate, self.head_dim)
                if prior_kv is not None:
                    new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                    new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

            compressed = self.kv_norm(
                (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = positions * self.compress_rate + first_window_position
            positions = positions.unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        compressed_kv = (
            compressed if cache_layer is None else cache_layer.update_compressor_states("indexer", compressed)
        )

        cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
        q = self.q_b_proj(q_residual).view(batch, seq_len, -1, self.head_dim).transpose(1, 2)
        q = apply_rotary_pos_emb(q, cos_q, sin_q).transpose(1, 2)

        index_scores = self.scorer(q, compressed_kv, hidden_states)
        compressed_len = compressed_kv.shape[1]
        top_k = min(self.index_topk, compressed_len)

        if compressed_len > 0:
            causal_threshold = (position_ids + 1) // self.compress_rate
            entry_indices = torch.arange(compressed_len, device=index_scores.device)
            future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
            index_scores = index_scores.masked_fill(future_mask, float("-inf"))
            top_k_indices = index_scores.topk(top_k, dim=-1).indices
            invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
            return torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)

        return index_scores.topk(top_k, dim=-1).indices


class DeepseekV4CSACompressor(nn.Module):
    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4AttentionConfig) -> None:
        super().__init__()
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.indexer = DeepseekV4Indexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: DeepseekV4DynamicCache | None,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = hidden_states.shape
        cache_layer: DeepseekV4CSACache | None = past_key_values.layers[layer_idx] if past_key_values is not None else None
        kv = self.kv_proj(hidden_states)
        gate = self.gate_proj(hidden_states)

        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("compressor", kv, gate)

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
            if cache_layer is not None:
                prior_kv, prior_gate = cache_layer.update_overlap_state(
                    "compressor",
                    chunk_kv,
                    chunk_gate,
                    self.head_dim,
                )
                if prior_kv is not None:
                    new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                    new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

            compressed = self.kv_norm(
                (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = positions * self.compress_rate + first_window_position
            positions = positions.unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        if cache_layer is not None:
            compressed = cache_layer.update_compressor_states("compressor", compressed)
        compressed_kv = compressed.unsqueeze(1)

        top_k_indices = self.indexer(hidden_states, q_residual, position_ids, past_key_values, layer_idx)
        compressed_len = compressed_kv.shape[2]
        valid = top_k_indices >= 0
        safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
        block_bias = compressed_kv.new_full((batch, 1, seq_len, compressed_len + 1), float("-inf"))
        block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
        return compressed_kv, block_bias[..., :compressed_len]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def flash_local_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    window: int,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if flash_attn_func is None or window <= 0:
        return None
    if query.device.type != "cuda":
        return None
    if query.dtype not in (torch.float16, torch.bfloat16):
        return None
    if key.dtype != query.dtype or value.dtype != query.dtype:
        return None

    local_output, softmax_lse, _ = flash_attn_func(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        dropout_p=0.0,
        softmax_scale=scaling,
        causal=True,
        window_size=(window - 1, 0),
        return_attn_probs=True,
    )
    return local_output.transpose(1, 2).contiguous(), softmax_lse


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    del kwargs
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    query_len = query.shape[-2]
    original_key_len = min(query_len, key_states.shape[-2])
    compressed_len = key_states.shape[-2] - original_key_len

    if getattr(module.config, "_attn_implementation", "sdpa") == "sdpa":
        batch, heads, query_len, _ = query.shape
        head_dim = query.shape[-1]
        padded_head_dim = ((head_dim + 1 + 7) // 8) * 8
        query_states = query.new_zeros((*query.shape[:-1], padded_head_dim))
        query_states[..., :head_dim] = query
        query_states[..., head_dim] = 1.0

        padded_key_states = key_states.new_zeros((*key_states.shape[:-1], padded_head_dim))
        padded_key_states[..., :head_dim] = key_states
        sink_key = key_states.new_zeros((batch, heads, 1, padded_head_dim))
        sink_key[..., head_dim] = (module.sinks / scaling).to(key_states.dtype).reshape(1, heads, 1)
        key_states = torch.cat([padded_key_states, sink_key], dim=2)

        sink_value = value_states.new_zeros((batch, heads, 1, value_states.shape[-1]))
        value_states = torch.cat([value_states, sink_value], dim=2)

        if attention_mask is not None:
            sink_mask = attention_mask.new_zeros((*attention_mask.shape[:-1], 1))
            attention_mask = torch.cat([attention_mask, sink_mask], dim=-1)

        attn_output = F.scaled_dot_product_attention(
            query_states.contiguous(),
            key_states.contiguous(),
            value_states.contiguous(),
            attn_mask=attention_mask,
            dropout_p=float(dropout) if module.training else 0.0,
            scale=scaling,
        )
        return attn_output.transpose(1, 2).contiguous(), None

    if query_len > 1 and original_key_len == query_len and module.sliding_window > 0:
        window = min(int(module.sliding_window), query_len)
        key_orig = key_states[:, :, :original_key_len, :]
        value_orig = value_states[:, :, :original_key_len, :]
        key_comp = key_states[:, :, original_key_len:, :] if compressed_len > 0 else None
        value_comp = value_states[:, :, original_key_len:, :] if compressed_len > 0 else None
        comp_mask = attention_mask[..., original_key_len:] if attention_mask is not None and compressed_len > 0 else None
        chunk_size = int(getattr(module.config, "local_attention_chunk_size", 2048))
        if float(dropout) == 0.0:
            flash_local = flash_local_attention_forward(query, key_orig, value_orig, scaling=scaling, window=window)
            if flash_local is not None:
                local_output, local_lse = flash_local
                outputs = []

                for start in range(0, query_len, chunk_size):
                    end = min(start + chunk_size, query_len)
                    chunk_query = query[:, :, start:end, :]
                    chunk_output = local_output[:, :, start:end, :]
                    chunk_local_lse = local_lse[:, :, start:end].unsqueeze(-1).float()
                    logits = [chunk_local_lse]

                    if compressed_len > 0:
                        comp_logits = torch.matmul(chunk_query.float(), key_comp.transpose(2, 3).float()) * scaling
                        if comp_mask is not None:
                            comp_logits = comp_logits + comp_mask[..., start:end, :].float()
                        logits.append(comp_logits)

                    sink_logits = module.sinks.reshape(1, -1, 1, 1).expand(
                        query.shape[0],
                        -1,
                        end - start,
                        -1,
                    ).float()
                    combined_logits = torch.cat([*logits, sink_logits], dim=-1)
                    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
                    probs = F.softmax(combined_logits, dim=-1, dtype=torch.float32)

                    local_prob = probs[..., :1].to(chunk_output.dtype)
                    fused_output = chunk_output * local_prob
                    if compressed_len > 0:
                        comp_probs = probs[..., 1 : 1 + compressed_len].to(value_comp.dtype)
                        fused_output = fused_output + torch.matmul(comp_probs, value_comp)
                    outputs.append(fused_output)

                attn_output = torch.cat(outputs, dim=2)
                return attn_output.transpose(1, 2).contiguous(), None

        offsets = torch.arange(window - 1, -1, -1, device=query.device)
        outputs = []

        for start in range(0, query_len, chunk_size):
            end = min(start + chunk_size, query_len)
            chunk_query = query[:, :, start:end, :]
            positions = torch.arange(start, end, device=query.device)
            local_indices = positions[:, None] - offsets[None, :]
            local_valid = local_indices >= 0
            local_indices = local_indices.clamp_min(0)

            local_keys = key_orig[:, :, local_indices, :]
            local_values = value_orig[:, :, local_indices, :]
            local_logits = (chunk_query.unsqueeze(-2).float() * local_keys.float()).sum(dim=-1) * scaling
            local_logits = local_logits.masked_fill(
                ~local_valid.view(1, 1, end - start, window),
                float("-inf"),
            )

            logits = [local_logits]
            values = [local_values]
            if compressed_len > 0:
                comp_logits = torch.matmul(chunk_query.float(), key_comp.transpose(2, 3).float()) * scaling
                if comp_mask is not None:
                    comp_logits = comp_logits + comp_mask[..., start:end, :].float()
                logits.append(comp_logits)
                values.append(value_comp)

            sink_logits = module.sinks.reshape(1, -1, 1, 1).expand(
                query.shape[0],
                -1,
                end - start,
                -1,
            ).float()
            combined_logits = torch.cat([*logits, sink_logits], dim=-1)
            combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
            probs = F.softmax(combined_logits, dim=-1, dtype=torch.float32)
            probs = F.dropout(probs, p=float(dropout), training=module.training)

            cursor = 0
            local_probs = probs[..., cursor : cursor + window].to(local_values.dtype)
            cursor += window
            chunk_output = (local_probs.unsqueeze(-1) * local_values).sum(dim=-2)
            if compressed_len > 0:
                comp_probs = probs[..., cursor : cursor + compressed_len].to(value_comp.dtype)
                chunk_output = chunk_output + torch.matmul(comp_probs, value_comp)
            outputs.append(chunk_output)

        attn_output = torch.cat(outputs, dim=2)
        return attn_output.transpose(1, 2).contiguous(), None

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)
    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]
    attn_weights = F.dropout(scores, p=dropout, training=module.training).to(value_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


COMPRESSOR_CLASSES = {
    "sliding_attention": None,
    "compressed_sparse_attention": DeepseekV4CSACompressor,
    "heavily_compressed_attention": DeepseekV4HCACompressor,
}


class DeepseekV4Attention(nn.Module):
    """DeepSeek-V4 attention ported from Hugging Face for this tiny V3 fork."""

    def __init__(self, config: DeepseekV4AttentionConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.rope_layer_type = "main" if self.layer_type == "sliding_attention" else "compress"
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.zeros(self.num_heads))
        self.compressor = (
            COMPRESSOR_CLASSES[self.layer_type](config) if self.layer_type != "sliding_attention" else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: DeepseekV4DynamicCache | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)

        if past_key_values is not None:
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        block_bias = None
        if self.compressor is not None:
            compressed_kv, block_bias = self.compressor(
                hidden_states,
                q_residual,
                position_ids,
                past_key_values,
                self.layer_idx,
            )
            kv = torch.cat([kv, compressed_kv], dim=2)

        if isinstance(attention_mask, torch.Tensor) and kv.shape[2] > attention_mask.shape[-1]:
            if block_bias is not None:
                if attention_mask.shape[0] != block_bias.shape[0]:
                    attention_mask = attention_mask.expand(block_bias.shape[0], -1, -1, -1)
                attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
            else:
                attention_mask = F.pad(attention_mask, (0, kv.shape[2] - attention_mask.shape[-1]), value=0.0)

        attn_output, attn_weights = eager_attention_forward(
            self,
            q,
            kv,
            kv,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )

        attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)
        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self.o_b_proj(grouped)
        return output, attn_weights


def build_sliding_window_causal_mask(
    config: DeepseekV4AttentionConfig,
    position_ids: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
    key_len: int | None = None,
) -> torch.Tensor:
    batch, query_len = position_ids.shape
    if key_len is None:
        key_len = query_len
    key_positions = torch.arange(key_len, device=device)
    query_positions = position_ids[:, :, None]
    allowed = key_positions.view(1, 1, -1) <= query_positions
    if config.sliding_window > 0:
        allowed = allowed & (key_positions.view(1, 1, -1) >= query_positions - config.sliding_window + 1)

    if attention_mask is not None:
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must have shape [batch, seq_len] before V4 mask construction.")
        key_mask = attention_mask.to(dtype=torch.bool)
        if key_mask.shape[-1] != key_len:
            key_mask = key_mask[:, -key_len:]
        allowed = allowed & key_mask[:, None, :]

    mask_batch = int(attention_mask.shape[0]) if attention_mask is not None else int(allowed.shape[0])
    mask = torch.zeros((mask_batch, 1, query_len, key_len), dtype=dtype, device=device)
    return mask.masked_fill(~allowed[:, None, :, :], float("-inf"))


def slice_position_embeddings(
    position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]],
    length: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {name: (cos[:, :length], sin[:, :length]) for name, (cos, sin) in position_embeddings.items()}
