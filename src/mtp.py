from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class DeepSeekV3MTPModule(nn.Module):
    """One DeepSeek-V3 multi-token prediction module.

    The module mirrors V3's MTP topology: normalize the future-token embedding
    and current hidden state, project their concatenation back to hidden size,
    run a full Transformer block, then share the main LM head for logits.
    """

    def __init__(
        self,
        hidden_size: int,
        block: nn.Module,
        norm_factory: Callable[[int], nn.Module],
    ) -> None:
        super().__init__()
        self.enorm = norm_factory(hidden_size)
        self.hnorm = norm_factory(hidden_size)
        self.eh_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.block = block

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        depth_idx: int,
        embed_tokens: nn.Embedding,
        lm_head: nn.Linear,
        attention_mask: torch.Tensor | None = None,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        shift = depth_idx + 1
        usable = hidden_states.shape[1] - shift
        if usable <= 0:
            zero = hidden_states.new_zeros(())
            return {
                "loss": zero,
                "hidden_states": hidden_states,
                "aux_loss": zero,
                "stats": {
                    "router_entropy": zero,
                    "expert_load_variance": zero,
                    "aux_loss": zero,
                },
            }

        h = hidden_states[:, :usable, :]
        future_ids = labels[:, shift - 1 : shift - 1 + usable].clamp_min(0)
        target = labels[:, shift : shift + usable]
        future_emb = embed_tokens(future_ids)

        fused = self.eh_proj(torch.cat([self.enorm(future_emb), self.hnorm(h)], dim=-1))
        block_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                block_mask = attention_mask[:, :, :usable, :usable]
            else:
                block_mask = attention_mask[:, :usable]
        block_position_ids = position_ids[:, :usable] if position_ids is not None else None
        block_position_embeddings = None
        if position_embeddings is not None:
            block_position_embeddings = {
                name: (cos[:, :usable], sin[:, :usable])
                for name, (cos, sin) in position_embeddings.items()
            }
        use_checkpoint = bool(getattr(self.block, "activation_checkpointing", False)) and self.training
        if use_checkpoint:
            mtp_hidden, aux_loss, stats, _ = checkpoint(
                lambda block_hidden: self.block(
                    block_hidden,
                    attention_mask=block_mask,
                    position_embeddings=block_position_embeddings,
                    position_ids=block_position_ids,
                    input_ids=future_ids,
                    past_key_values=None,
                    use_cache=False,
                ),
                fused,
                use_reentrant=False,
            )
        else:
            mtp_hidden, aux_loss, stats, _ = self.block(
                fused,
                attention_mask=block_mask,
                position_embeddings=block_position_embeddings,
                position_ids=block_position_ids,
                input_ids=future_ids,
                past_key_values=None,
                use_cache=False,
            )
        logits = lm_head(mtp_hidden)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
            ignore_index=-100,
        )
        return {
            "loss": loss,
            "hidden_states": mtp_hidden,
            "aux_loss": aux_loss,
            "stats": stats,
        }


MultiTokenPredictionHead = DeepSeekV3MTPModule
