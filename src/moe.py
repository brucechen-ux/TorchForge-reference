from __future__ import annotations

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float | None = None) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        if self.swiglu_limit is not None:
            gate = gate.clamp(max=self.swiglu_limit)
            up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        gated = F.silu(gate) * up
        return self.down_proj(gated)


class DeepSeekV4TopKRouter(nn.Module):
    """DeepSeek-V4 top-k router: score + correction bias selects experts, raw score weights routed tokens."""

    def __init__(
        self,
        hidden_size: int,
        num_routed_experts: int,
        top_k: int,
        route_scale: float = 1.5,
        score_function: str = "sqrtsoftplus",
        use_correction_bias: bool = True,
        balance_bias: bool = False,
        balance_bias_lr: float = 1.0e-3,
        balance_bias_clamp: float = 5.0,
    ) -> None:
        super().__init__()
        if score_function not in {"sigmoid", "softmax", "sqrtsoftplus"}:
            raise ValueError(f"Unsupported router score_function: {score_function}")
        self.hidden_size = hidden_size
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.route_scale = route_scale
        self.score_function = score_function
        self.balance_bias_enabled = balance_bias
        self.balance_bias_lr = balance_bias_lr
        self.balance_bias_clamp = balance_bias_clamp
        self._pending_balance_update: torch.Tensor | None = None

        self.weight = nn.Parameter(torch.empty(num_routed_experts, hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if use_correction_bias:
            self.register_buffer("e_score_correction_bias", torch.zeros(num_routed_experts), persistent=True)
        else:
            self.register_buffer("e_score_correction_bias", None)

    def _scores(self, logits: torch.Tensor) -> torch.Tensor:
        if self.score_function == "sigmoid":
            return logits.sigmoid()
        if self.score_function == "softmax":
            return logits.softmax(dim=-1)
        return torch.sqrt(F.softplus(logits))

    @property
    def balance_bias(self) -> torch.Tensor | None:
        return self.e_score_correction_bias

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_size)
        logits = F.linear(flat.float(), self.weight.float())
        scores = self._scores(logits)
        route_scores = scores
        if self.e_score_correction_bias is not None:
            route_scores = route_scores + self.e_score_correction_bias
        topk_idx = torch.topk(route_scores, self.top_k, dim=-1, sorted=False).indices
        topk_weight = scores.gather(1, topk_idx)
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1.0e-20)
        return topk_idx, topk_weight * self.route_scale, scores, logits

    @torch.no_grad()
    def update_balance_bias(self, expert_counts: torch.Tensor) -> None:
        if not self.balance_bias_enabled or self.e_score_correction_bias is None:
            return
        counts = expert_counts.detach().float()
        mean = counts.mean().clamp_min(1.0)
        adjustment = (mean - counts) / mean
        self.e_score_correction_bias.add_(self.balance_bias_lr * adjustment)
        self.e_score_correction_bias.clamp_(-self.balance_bias_clamp, self.balance_bias_clamp)

    @torch.no_grad()
    def stage_balance_bias_update(self, expert_counts: torch.Tensor) -> None:
        if not self.balance_bias_enabled or self.e_score_correction_bias is None:
            return
        self._pending_balance_update = expert_counts.detach().float()

    @torch.no_grad()
    def apply_pending_balance_bias_update(self) -> None:
        if self._pending_balance_update is None:
            return
        self.update_balance_bias(self._pending_balance_update)
        self._pending_balance_update = None


class DeepSeekV4HashRouter(nn.Module):
    """DeepSeek-V4 Hash-MoE router.

    Expert selection is fixed by token id through ``tid2eid``. The learned gate
    still scores the selected experts and supplies their combine weights.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        num_routed_experts: int,
        top_k: int,
        route_scale: float = 1.5,
        score_function: str = "sqrtsoftplus",
    ) -> None:
        super().__init__()
        if score_function not in {"sigmoid", "softmax", "sqrtsoftplus"}:
            raise ValueError(f"Unsupported router score_function: {score_function}")
        if top_k > num_routed_experts:
            raise ValueError("Hash-MoE top_k cannot exceed num_routed_experts.")
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.route_scale = route_scale
        self.score_function = score_function
        self.weight = nn.Parameter(torch.empty(num_routed_experts, hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        token_ids = torch.arange(vocab_size, dtype=torch.long).unsqueeze(1)
        offsets = torch.arange(top_k, dtype=torch.long).unsqueeze(0)
        tid2eid = (token_ids + offsets) % num_routed_experts
        self.register_buffer("tid2eid", tid2eid, persistent=True)

    def _scores(self, logits: torch.Tensor) -> torch.Tensor:
        if self.score_function == "sigmoid":
            return logits.sigmoid()
        if self.score_function == "softmax":
            return logits.softmax(dim=-1)
        return torch.sqrt(F.softplus(logits))

    @property
    def balance_bias(self) -> None:
        return None

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_size)
        flat_ids = input_ids.reshape(-1).clamp_min(0)
        if flat_ids.max() >= self.vocab_size:
            raise ValueError("Hash-MoE input_ids contain token ids outside moe.vocab_size.")
        logits = F.linear(flat.float(), self.weight.float())
        scores = self._scores(logits)
        topk_idx = self.tid2eid[flat_ids].to(device=hidden_states.device)
        topk_weight = scores.gather(1, topk_idx)
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1.0e-20)
        return topk_idx, topk_weight * self.route_scale, scores, logits

    @torch.no_grad()
    def stage_balance_bias_update(self, expert_counts: torch.Tensor) -> None:
        del expert_counts

    @torch.no_grad()
    def apply_pending_balance_bias_update(self) -> None:
        return


class DeepSeekV4Experts(nn.Module):
    """HF DeepSeek-V4 style packed expert weights: gate/up and down stored as 3D tensors."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        swiglu_limit: float = 10.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_size
        self.intermediate_dim = intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2 * intermediate_size, hidden_size))
        self.down_proj = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.limit = swiglu_limit
        nn.init.kaiming_uniform_(self.gate_up_proj, a=5**0.5)
        nn.init.kaiming_uniform_(self.down_proj, a=5**0.5)

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return F.silu(gate) * up

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        final = torch.zeros_like(hidden_states)
        expert_load = torch.zeros(self.num_experts, device=hidden_states.device, dtype=torch.float32)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx_tensor in hit:
            expert_idx = expert_idx_tensor[0]
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))
            current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, current.to(final.dtype))
            expert_load[expert_idx] = float(token_idx.numel())
        return final, expert_load


class DeepSeekV4ModuleListExperts(nn.Module):
    """Unpacked experts for ablations against the packed V4 expert parameter layout."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        swiglu_limit: float | None = None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [SwiGLUExpert(hidden_size, intermediate_size, swiglu_limit=swiglu_limit) for _ in range(num_experts)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        final = torch.zeros_like(hidden_states)
        expert_load = torch.zeros(self.num_experts, device=hidden_states.device, dtype=torch.float32)
        for expert_id, expert in enumerate(self.experts):
            token_mask = top_k_index == expert_id
            if not token_mask.any():
                final = final + expert(hidden_states[:1]).sum() * 0.0
                continue
            token_pos, route_pos = token_mask.nonzero(as_tuple=True)
            current = expert(hidden_states[token_pos]).to(final.dtype)
            current = current * top_k_weights[token_pos, route_pos, None].to(final.dtype)
            final.index_add_(0, token_pos, current)
            expert_load[expert_id] = float(token_pos.numel())
        return final, expert_load


class DeepSeekV3Gate(nn.Module):
    """DeepSeek grouped top-k router.

    The default path follows V3 routing: sigmoid scores, optional correction
    bias for auxiliary-loss-free balancing, group-limited expert selection, and
    a routed scaling factor. The sqrtsoftplus score function is available for
    V4-style affinity scoring in tiny hybrid configs.
    """

    def __init__(
        self,
        hidden_size: int,
        num_routed_experts: int,
        num_experts_per_token: int,
        num_expert_groups: int,
        num_limited_groups: int,
        route_scale: float,
        score_function: str = "sigmoid",
        normalize_topk_prob: bool = True,
        use_correction_bias: bool = True,
        balance_bias: bool = False,
        balance_bias_lr: float = 1.0e-3,
        balance_bias_clamp: float = 5.0,
    ) -> None:
        super().__init__()
        if num_routed_experts % num_expert_groups != 0:
            raise ValueError("num_routed_experts must be divisible by num_expert_groups.")
        group_size = num_routed_experts // num_expert_groups
        if num_limited_groups * group_size < num_experts_per_token:
            raise ValueError("num_limited_groups leaves fewer candidate experts than num_experts_per_token.")
        if score_function not in {"sigmoid", "softmax", "sqrtsoftplus"}:
            raise ValueError(f"Unsupported router score_function: {score_function}")

        self.hidden_size = hidden_size
        self.num_routed_experts = num_routed_experts
        self.num_experts_per_token = num_experts_per_token
        self.num_expert_groups = num_expert_groups
        self.num_limited_groups = num_limited_groups
        self.route_scale = route_scale
        self.score_function = score_function
        self.normalize_topk_prob = normalize_topk_prob
        self.use_correction_bias = use_correction_bias
        self.balance_bias_enabled = balance_bias
        self.balance_bias_lr = balance_bias_lr
        self.balance_bias_clamp = balance_bias_clamp
        self._pending_balance_update: torch.Tensor | None = None

        self.weight = nn.Parameter(torch.empty(num_routed_experts, hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if use_correction_bias:
            self.register_buffer("balance_bias", torch.zeros(num_routed_experts))
        else:
            self.register_buffer("balance_bias", None)

    @property
    def e_score_correction_bias(self) -> torch.Tensor | None:
        return self.balance_bias

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = F.linear(hidden_states.float(), self.weight.float())
        if self.score_function == "sigmoid":
            scores = logits.sigmoid()
        elif self.score_function == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            scores = torch.sqrt(F.softplus(logits))

        route_scores = scores
        if self.e_score_correction_bias is not None:
            route_scores = route_scores + self.e_score_correction_bias

        if self.num_expert_groups > 1:
            batch_tokens = route_scores.shape[0]
            grouped = route_scores.view(batch_tokens, self.num_expert_groups, -1)
            if self.e_score_correction_bias is not None:
                group_scores = grouped.topk(k=min(2, grouped.shape[-1]), dim=-1).values.sum(dim=-1)
            else:
                group_scores = grouped.max(dim=-1).values
            selected_groups = group_scores.topk(k=self.num_limited_groups, dim=-1).indices
            group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
            group_mask.scatter_(1, selected_groups, True)
            expert_mask = group_mask[:, :, None].expand_as(grouped).reshape_as(route_scores)
            route_scores = route_scores.masked_fill(~expert_mask, torch.finfo(route_scores.dtype).min)

        topk_idx = route_scores.topk(k=self.num_experts_per_token, dim=-1).indices
        topk_weight = scores.gather(1, topk_idx)
        if self.normalize_topk_prob and self.num_experts_per_token > 1:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        topk_weight = topk_weight * self.route_scale
        return topk_idx, topk_weight, scores, logits

    @torch.no_grad()
    def update_balance_bias(self, expert_counts: torch.Tensor) -> None:
        if not self.balance_bias_enabled or self.balance_bias is None:
            return
        counts = expert_counts.detach().float()
        mean = counts.mean().clamp_min(1.0)
        adjustment = (mean - counts) / mean
        self.balance_bias.add_(self.balance_bias_lr * adjustment)
        self.balance_bias.clamp_(-self.balance_bias_clamp, self.balance_bias_clamp)

    @torch.no_grad()
    def stage_balance_bias_update(self, expert_counts: torch.Tensor) -> None:
        if not self.balance_bias_enabled or self.balance_bias is None:
            return
        self._pending_balance_update = expert_counts.detach().float()

    @torch.no_grad()
    def apply_pending_balance_bias_update(self) -> None:
        if self._pending_balance_update is None:
            return
        self.update_balance_bias(self._pending_balance_update)
        self._pending_balance_update = None


class DeepSeekV3MoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expert_intermediate_size: int,
        num_routed_experts: int,
        num_shared_experts: int,
        top_k: int,
        aux_loss_weight: float,
        normalize_topk_prob: bool,
        num_expert_groups: int = 8,
        num_limited_groups: int = 4,
        route_scale: float = 2.5,
        score_function: str = "sigmoid",
        use_correction_bias: bool = True,
        balance_bias: bool = False,
        balance_bias_lr: float = 1.0e-3,
        balance_bias_clamp: float = 5.0,
        implementation: str = "torch",
        moe_ep_size: int = 1,
        moe_capacity_factor: float = 1.25,
        moe_eval_capacity_factor: float = 2.0,
        moe_min_capacity: int = 4,
        moe_drop_tokens: bool = False,
        moe_drop_policy: str = "probs",
        moe_use_rts: bool = True,
        moe_use_tutel: bool = False,
        swiglu_limit: float | None = 10.0,
        use_packed_experts: bool = True,
        router_type: str = "moe",
        vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.implementation = implementation.lower()
        self.router_type = router_type
        self.use_packed_experts = use_packed_experts
        if self.router_type not in {"hash_moe", "moe"}:
            raise ValueError(f"Unsupported V4 MoE router_type: {router_type}")
        if self.router_type == "hash_moe" and vocab_size is None:
            raise ValueError("Hash-MoE requires vocab_size.")

        if self.implementation == "deepspeed":
            if self.router_type == "hash_moe":
                raise ValueError("moe.implementation=deepspeed does not support V4 hash_moe in this fork.")
            self.impl = DeepSeekV3DeepSpeedMoE(
                hidden_size=hidden_size,
                expert_intermediate_size=expert_intermediate_size,
                num_routed_experts=num_routed_experts,
                num_shared_experts=num_shared_experts,
                top_k=top_k,
                aux_loss_weight=aux_loss_weight,
                normalize_topk_prob=normalize_topk_prob,
                num_expert_groups=num_expert_groups,
                num_limited_groups=num_limited_groups,
                route_scale=route_scale,
                score_function=score_function,
                use_correction_bias=use_correction_bias,
                balance_bias=balance_bias,
                balance_bias_lr=balance_bias_lr,
                balance_bias_clamp=balance_bias_clamp,
                moe_ep_size=moe_ep_size,
                moe_capacity_factor=moe_capacity_factor,
                moe_eval_capacity_factor=moe_eval_capacity_factor,
                moe_min_capacity=moe_min_capacity,
                moe_drop_tokens=moe_drop_tokens,
                moe_drop_policy=moe_drop_policy,
                moe_use_rts=moe_use_rts,
                moe_use_tutel=moe_use_tutel,
            )
            return
        if self.implementation != "torch":
            raise ValueError(f"Unsupported MoE implementation: {implementation}")
        self.impl = None

        if self.router_type == "hash_moe":
            self.gate = DeepSeekV4HashRouter(
                hidden_size=hidden_size,
                vocab_size=int(vocab_size),
                num_routed_experts=num_routed_experts,
                top_k=top_k,
                route_scale=route_scale,
                score_function=score_function,
            )
        else:
            self.gate = DeepSeekV4TopKRouter(
                hidden_size=hidden_size,
                num_routed_experts=num_routed_experts,
                top_k=top_k,
                route_scale=route_scale,
                score_function=score_function,
                use_correction_bias=use_correction_bias,
                balance_bias=balance_bias,
                balance_bias_lr=balance_bias_lr,
                balance_bias_clamp=balance_bias_clamp,
            )
        if use_packed_experts:
            if swiglu_limit is None:
                raise ValueError("Packed V4 experts require moe.swiglu_limit to be set.")
            self.experts = DeepSeekV4Experts(
                hidden_size=hidden_size,
                intermediate_size=expert_intermediate_size,
                num_experts=num_routed_experts,
                swiglu_limit=swiglu_limit,
            )
        else:
            self.experts = DeepSeekV4ModuleListExperts(
                hidden_size=hidden_size,
                intermediate_size=expert_intermediate_size,
                num_experts=num_routed_experts,
                swiglu_limit=swiglu_limit,
            )
        self.shared_experts = SwiGLUExpert(
            hidden_size,
            expert_intermediate_size * num_shared_experts,
            swiglu_limit=swiglu_limit,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.impl is not None:
            return self.impl(hidden_states)

        batch, seq_len, hidden_size = hidden_states.shape
        flat = hidden_states.reshape(batch * seq_len, hidden_size)

        if self.router_type == "hash_moe":
            if input_ids is None:
                raise ValueError("Hash-MoE forward requires input_ids.")
            topk_idx, topk_weight, router_scores, router_logits = self.gate(flat, input_ids)
        else:
            topk_idx, topk_weight, router_scores, router_logits = self.gate(flat)

        shared_out = self.shared_experts(flat).to(flat.dtype)
        routed_out, expert_load = self.experts(flat, topk_idx, topk_weight.to(flat.dtype))

        if self.training:
            self.gate.stage_balance_bias_update(expert_load)

        output = (shared_out + routed_out).reshape(batch, seq_len, hidden_size)

        load = expert_load / expert_load.sum().clamp_min(1.0)
        importance = router_scores / router_scores.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        importance_mean = importance.mean(dim=0)
        aux_loss = flat.new_zeros(())
        if self.aux_loss_weight > 0.0:
            aux_loss = self.aux_loss_weight * self.num_routed_experts * torch.sum(importance_mean * load.detach())
        router_entropy = (-importance.clamp_min(1e-9) * importance.clamp_min(1e-9).log()).sum(dim=-1).mean()
        load_variance = load.var(unbiased=False)

        stats = {
            "expert_load": load.detach(),
            "expert_load_variance": load_variance.detach(),
            "router_entropy": router_entropy.detach(),
            "gate_logits_mean": router_logits.detach().float().mean(),
            "gate_logits_std": router_logits.detach().float().std(unbiased=False),
            "aux_loss": aux_loss.detach(),
            "moe_routing": {
                "router_type": self.router_type,
                "expert_counts": expert_load.detach().float(),
                "topk_expert_counts": expert_load.detach().float(),
                "balance_bias": (
                    self.gate.balance_bias.detach().float()
                    if getattr(self.gate, "balance_bias", None) is not None
                    else torch.zeros(self.num_routed_experts, device=hidden_states.device)
                ),
            },
        }
        return output, aux_loss, stats


class DeepSeekV3MoEDispatchGate(nn.Module):
    """V3 grouped top-k router adapted to DeepSpeed MOELayer dispatch tensors."""

    def __init__(
        self,
        hidden_size: int,
        num_routed_experts: int,
        top_k: int,
        normalize_topk_prob: bool,
        num_expert_groups: int,
        num_limited_groups: int,
        route_scale: float,
        score_function: str,
        use_correction_bias: bool,
        balance_bias: bool,
        balance_bias_lr: float,
        balance_bias_clamp: float,
        capacity_factor: float,
        eval_capacity_factor: float,
        min_capacity: int,
        drop_tokens: bool,
        drop_policy: str,
    ) -> None:
        super().__init__()
        if drop_policy not in {"probs", "position"}:
            raise ValueError(f"Unsupported DeepSpeed MoE drop policy: {drop_policy}")
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.k = top_k
        self.capacity_factor = capacity_factor
        self.eval_capacity_factor = eval_capacity_factor
        self.min_capacity = min_capacity
        self.drop_tokens = drop_tokens
        self.drop_policy = drop_policy
        self.ep_group = None
        self.wall_clock_breakdown = False
        self.gate_time = 0.0
        self.last_router_entropy: torch.Tensor | None = None
        self.last_load_variance: torch.Tensor | None = None
        self.last_aux_loss: torch.Tensor | None = None
        self.last_stats: dict[str, torch.Tensor] | None = None
        self.router = DeepSeekV3Gate(
            hidden_size=hidden_size,
            num_routed_experts=num_routed_experts,
            num_experts_per_token=top_k,
            num_expert_groups=num_expert_groups,
            num_limited_groups=num_limited_groups,
            route_scale=route_scale,
            score_function=score_function,
            normalize_topk_prob=normalize_topk_prob,
            use_correction_bias=use_correction_bias,
            balance_bias=balance_bias,
            balance_bias_lr=balance_bias_lr,
            balance_bias_clamp=balance_bias_clamp,
        )

    def _set_ep_group(self, ep_group) -> None:
        if self.ep_group is not None:
            raise AssertionError("Attempting to override an existing ep_group")
        self.ep_group = ep_group

    def _capacity(self, num_tokens: int, device: torch.device) -> int:
        factor = self.capacity_factor if self.training else self.eval_capacity_factor
        capacity = math.ceil((num_tokens / self.num_routed_experts) * factor * self.top_k)
        capacity = max(capacity, self.min_capacity)
        return min(capacity, num_tokens)

    def forward(
        self,
        hidden_states: torch.Tensor,
        used_token: torch.Tensor | None = None,
        use_tutel: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del use_tutel
        num_tokens = hidden_states.shape[0]
        router_start = time.perf_counter()
        topk_idx, topk_weight, router_scores, router_logits = self.router(hidden_states)
        router_time_ms = (time.perf_counter() - router_start) * 1000.0

        dispatch_start = time.perf_counter()
        route_weights = torch.zeros_like(router_scores)
        route_weights.scatter_(1, topk_idx, topk_weight)
        route_mask = torch.zeros_like(router_scores, dtype=torch.bool)
        route_mask.scatter_(1, topk_idx, True)

        if used_token is not None:
            used = used_token.reshape(-1, 1).bool()
            route_mask = route_mask & used
            route_weights = route_weights * used.to(route_weights.dtype)

        pre_counts = route_mask.sum(dim=0).detach().to(hidden_states.device)
        top1_counts = torch.bincount(topk_idx[:, 0], minlength=self.num_routed_experts).detach().to(hidden_states.device)
        if self.training:
            self.router.stage_balance_bias_update(pre_counts)
        load = pre_counts.float() / pre_counts.sum().float().clamp_min(1.0)
        importance = router_scores / router_scores.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        importance_mean = importance.mean(dim=0)
        aux_loss = self.num_routed_experts * torch.sum(importance_mean * load.detach())

        if self.drop_tokens:
            capacity = self._capacity(num_tokens, hidden_states.device)
            if self.drop_policy == "probs":
                _, capacity_idx = torch.topk(route_weights, k=capacity, dim=0, sorted=False)
                capacity_mask = torch.zeros_like(route_mask)
                capacity_mask.scatter_(0, capacity_idx, True)
                route_mask = route_mask & capacity_mask
            else:
                locations = torch.cumsum(route_mask.to(torch.int64), dim=0) - 1
                route_mask = route_mask & (locations < capacity)
        else:
            capacity_tensor = pre_counts.max().to(hidden_states.device)
            if self.ep_group is not None:
                from deepspeed import comm as dist

                dist.all_reduce(capacity_tensor, op=dist.ReduceOp.MAX, group=self.ep_group)
            capacity = max(int(capacity_tensor.item()), 1)
        dispatch_time_ms = (time.perf_counter() - dispatch_start) * 1000.0

        combine_start = time.perf_counter()
        locations = torch.cumsum(route_mask.to(torch.int64), dim=0) - 1
        safe_locations = (locations.clamp_min(0) * route_mask.to(torch.int64)).to(torch.int64)
        locations_one_hot = F.one_hot(safe_locations, num_classes=capacity).to(route_weights.dtype)
        combine_weights = route_weights.unsqueeze(-1) * route_mask.unsqueeze(-1).to(route_weights.dtype) * locations_one_hot
        dispatch_mask = combine_weights.bool()
        combine_time_ms = (time.perf_counter() - combine_start) * 1000.0

        post_counts = route_mask.sum(dim=0).detach().to(hidden_states.device)
        dropped_per_expert = (pre_counts - post_counts).clamp_min(0)
        routed_assignments = pre_counts.sum().float()
        kept_assignments = post_counts.sum().float()
        dropped_assignments = dropped_per_expert.sum().float()
        capacity_slots = float(max(capacity * self.num_routed_experts, 1))
        post_mean = post_counts.float().mean().clamp_min(1.0)
        capacity_tensor = hidden_states.new_tensor(float(capacity))
        router_entropy = (-importance.clamp_min(1e-9) * importance.clamp_min(1e-9).log()).sum(dim=-1).mean()
        self.last_router_entropy = router_entropy.detach()
        self.last_load_variance = load.var(unbiased=False).detach()
        self.last_aux_loss = aux_loss.detach()
        self.last_stats = {
            "num_tokens": hidden_states.new_tensor(float(num_tokens)),
            "top_k": hidden_states.new_tensor(float(self.top_k)),
            "capacity": capacity_tensor,
            "capacity_factor": hidden_states.new_tensor(
                float(self.capacity_factor if self.training else self.eval_capacity_factor)
            ),
            "capacity_slots": hidden_states.new_tensor(capacity_slots),
            "routed_assignments": routed_assignments.detach(),
            "kept_assignments": kept_assignments.detach(),
            "dropped_assignments": dropped_assignments.detach(),
            "drop_rate": dropped_assignments.detach() / routed_assignments.detach().clamp_min(1.0),
            "capacity_utilization": kept_assignments.detach() / hidden_states.new_tensor(capacity_slots),
            "max_capacity_utilization": post_counts.float().max() / capacity_tensor.clamp_min(1.0),
            "pre_expert_counts": pre_counts.float(),
            "expert_counts": post_counts.float(),
            "dropped_per_expert": dropped_per_expert.float(),
            "top1_expert_counts": top1_counts.float(),
            "topk_expert_counts": pre_counts.float(),
            "gate_logits_mean": router_logits.detach().float().mean(),
            "gate_logits_std": router_logits.detach().float().std(unbiased=False),
            "router_entropy": router_entropy.detach().float(),
            "balance_bias": (
                self.router.e_score_correction_bias.detach().float()
                if self.router.e_score_correction_bias is not None
                else torch.zeros(self.num_routed_experts, device=hidden_states.device)
            ),
            "expert_load_mean": post_counts.float().mean(),
            "expert_load_max": post_counts.float().max(),
            "expert_load_p95": torch.quantile(post_counts.float(), 0.95),
            "expert_load_std": post_counts.float().std(unbiased=False),
            "max_load_over_mean": post_counts.float().max() / post_mean,
            "load_cv": post_counts.float().std(unbiased=False) / post_mean,
            "l_aux": aux_loss.detach().float(),
            "router_time_ms": hidden_states.new_tensor(router_time_ms),
            "moe_dispatch_time_ms": hidden_states.new_tensor(dispatch_time_ms),
            "moe_combine_time_ms": hidden_states.new_tensor(combine_time_ms),
        }
        return aux_loss, combine_weights, dispatch_mask, pre_counts

    @torch.no_grad()
    def apply_pending_balance_bias_update(self) -> None:
        self.router.apply_pending_balance_bias_update()


class DeepSeekV3DeepSpeedMoE(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expert_intermediate_size: int,
        num_routed_experts: int,
        num_shared_experts: int,
        top_k: int,
        aux_loss_weight: float,
        normalize_topk_prob: bool,
        num_expert_groups: int,
        num_limited_groups: int,
        route_scale: float,
        score_function: str,
        use_correction_bias: bool,
        balance_bias: bool,
        balance_bias_lr: float,
        balance_bias_clamp: float,
        moe_ep_size: int,
        moe_capacity_factor: float,
        moe_eval_capacity_factor: float,
        moe_min_capacity: int,
        moe_drop_tokens: bool,
        moe_drop_policy: str,
        moe_use_rts: bool,
        moe_use_tutel: bool,
    ) -> None:
        super().__init__()
        if num_routed_experts % moe_ep_size != 0:
            raise ValueError("num_routed_experts must be divisible by moe_ep_size for DeepSpeed MoE.")
        try:
            from deepspeed.moe.layer import MoE
        except ImportError as exc:
            raise RuntimeError("moe.implementation=deepspeed requires deepspeed to be importable.") from exc

        self.hidden_size = hidden_size
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.aux_loss_weight = aux_loss_weight
        self.shared_experts = SwiGLUExpert(
            hidden_size,
            expert_intermediate_size * num_shared_experts,
        )
        self.gate = DeepSeekV3MoEDispatchGate(
            hidden_size=hidden_size,
            num_routed_experts=num_routed_experts,
            top_k=top_k,
            normalize_topk_prob=normalize_topk_prob,
            num_expert_groups=num_expert_groups,
            num_limited_groups=num_limited_groups,
            route_scale=route_scale,
            score_function=score_function,
            use_correction_bias=use_correction_bias,
            balance_bias=balance_bias,
            balance_bias_lr=balance_bias_lr,
            balance_bias_clamp=balance_bias_clamp,
            capacity_factor=moe_capacity_factor,
            eval_capacity_factor=moe_eval_capacity_factor,
            min_capacity=moe_min_capacity,
            drop_tokens=moe_drop_tokens,
            drop_policy=moe_drop_policy,
        )
        self.routed_moe = MoE(
            hidden_size=hidden_size,
            expert=SwiGLUExpert(hidden_size, expert_intermediate_size),
            num_experts=num_routed_experts,
            ep_size=moe_ep_size,
            k=top_k,
            capacity_factor=moe_capacity_factor,
            eval_capacity_factor=moe_eval_capacity_factor,
            min_capacity=moe_min_capacity,
            drop_tokens=moe_drop_tokens,
            use_rts=moe_use_rts,
            use_tutel=moe_use_tutel,
        )
        self.routed_moe.deepspeed_moe.gate = self.gate
        self.routed_moe.deepspeed_moe.use_tutel = bool(getattr(self.routed_moe.deepspeed_moe, "use_tutel", False))

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch, seq_len, hidden_size = hidden_states.shape
        flat = hidden_states.reshape(batch * seq_len, hidden_size)

        shared_start = time.perf_counter()
        shared_out = self.shared_experts(flat).to(flat.dtype)
        shared_time_ms = (time.perf_counter() - shared_start) * 1000.0
        routed_start = time.perf_counter()
        routed_out, gate_aux_loss, exp_counts = self.routed_moe(flat)
        routed_time_ms = (time.perf_counter() - routed_start) * 1000.0
        routed_out = routed_out.to(flat.dtype)
        output = (shared_out + routed_out).reshape(batch, seq_len, hidden_size)

        aux_loss = gate_aux_loss.to(flat.dtype) * self.aux_loss_weight
        load = exp_counts.float() / exp_counts.float().sum().clamp_min(1.0)
        routing = self.gate.last_stats or {}
        router_ms = float(routing.get("router_time_ms", flat.new_zeros(())).detach().float().item()) if routing else 0.0
        dispatch_ms = float(routing.get("moe_dispatch_time_ms", flat.new_zeros(())).detach().float().item()) if routing else 0.0
        combine_ms = float(routing.get("moe_combine_time_ms", flat.new_zeros(())).detach().float().item()) if routing else 0.0
        a2a_ms = float(getattr(self.routed_moe.deepspeed_moe, "time_falltoall", 0.0)) + float(
            getattr(self.routed_moe.deepspeed_moe, "time_salltoall", 0.0)
        )
        expert_mlp_time_ms = max(routed_time_ms - router_ms - dispatch_ms - combine_ms - a2a_ms, 0.0)
        stats = {
            "expert_load": load.detach(),
            "expert_load_variance": (
                self.gate.last_load_variance
                if self.gate.last_load_variance is not None
                else load.var(unbiased=False).detach()
            ),
            "router_entropy": (
                self.gate.last_router_entropy
                if self.gate.last_router_entropy is not None
                else flat.new_zeros(())
            ),
            "aux_loss": aux_loss.detach(),
            "moe_routing": routing,
            "moe_time": flat.new_tensor(float(getattr(self.routed_moe.deepspeed_moe, "time_moe", 0.0))),
            "moe_first_alltoall_time": flat.new_tensor(
                float(getattr(self.routed_moe.deepspeed_moe, "time_falltoall", 0.0))
            ),
            "moe_second_alltoall_time": flat.new_tensor(
                float(getattr(self.routed_moe.deepspeed_moe, "time_salltoall", 0.0))
            ),
            "router_time": flat.new_tensor(router_ms),
            "moe_dispatch_time": flat.new_tensor(dispatch_ms),
            "expert_mlp_time": flat.new_tensor(expert_mlp_time_ms),
            "moe_combine_time": flat.new_tensor(combine_ms),
            "shared_mlp_time": flat.new_tensor(shared_time_ms),
        }
        return output, aux_loss, stats
