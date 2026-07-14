from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight.to(dtype=input_dtype) * hidden_states.to(dtype=input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.flatten(-2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=dtype)[None, None, :, :]
        sin = emb.sin().to(dtype=dtype)[None, None, :, :]
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


class DeepSeekV3MLA(nn.Module):
    """DeepSeek-V3 MLA projections with eager dense attention.

    The training path expands K/V before attention. When use_cache=True, the
    cache stores the low-rank KV latent plus the RoPE K branch, matching the
    DeepSeek-V3 cache topology rather than expanded per-head K/V tensors.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        rope_theta: float,
        rms_norm_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if q_lora_rank <= 0:
            raise ValueError("DeepSeek-V3 MLA expects q_lora_rank > 0.")
        if kv_lora_rank <= 0:
            raise ValueError("DeepSeek-V3 MLA expects kv_lora_rank > 0.")
        if qk_rope_head_dim % 2 != 0:
            raise ValueError("qk_rope_head_dim must be even for RoPE.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.scale = 1.0 / math.sqrt(self.qk_head_dim)

        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(q_lora_rank, rms_norm_eps)
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * self.qk_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size,
            kv_lora_rank + qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(kv_lora_rank, rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_heads * (qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(num_heads * v_head_dim, hidden_size, bias=False)
        self.rotary = RotaryEmbedding(qk_rope_head_dim, theta=rope_theta)

    def _causal_mask(
        self,
        query_len: int,
        key_len: int,
        past_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        q_pos = torch.arange(past_len, past_len + query_len, device=device)[:, None]
        k_pos = torch.arange(key_len, device=device)[None, :]
        return k_pos <= q_pos

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype
        device = hidden_states.device

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(batch, seq_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_rope = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent, k_rope = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)
        k_rope = k_rope[:, None, :, :]

        past_len = 0
        if past_key_value is not None:
            past_len = past_key_value[0].shape[1]

        cos, sin = self.rotary(seq_len=seq_len, device=device, dtype=dtype, offset=past_len)
        q_rope = apply_rope(q_rope, cos, sin)
        k_rope = apply_rope(k_rope, cos, sin)

        if past_key_value is not None:
            kv_latent = torch.cat([past_key_value[0], kv_latent], dim=1)
            k_rope = torch.cat([past_key_value[1], k_rope], dim=2)

        present = (kv_latent, k_rope) if use_cache else None
        key_len = kv_latent.shape[1]
        kv_expanded = self.kv_b_proj(kv_latent)
        kv_expanded = kv_expanded.view(
            batch,
            key_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        ).transpose(1, 2)
        k_nope, v = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_rope = k_rope.expand(-1, self.num_heads, -1, -1)

        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal = self._causal_mask(seq_len, key_len, past_len, device=device)
        attn_scores = attn_scores.masked_fill(~causal[None, None, :, :], torch.finfo(attn_scores.dtype).min)

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            if key_mask.shape[-1] != k.shape[2]:
                key_mask = key_mask[..., -k.shape[2] :]
            attn_scores = attn_scores.masked_fill(~key_mask, torch.finfo(attn_scores.dtype).min)

        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(dtype=dtype)
        attn_output = torch.matmul(attn_probs, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch,
            seq_len,
            self.num_heads * self.v_head_dim,
        )
        output = self.o_proj(attn_output)
        return output, present


MLAAttention = DeepSeekV3MLA
