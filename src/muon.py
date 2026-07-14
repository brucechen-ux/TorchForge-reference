from __future__ import annotations

import math
import time
from typing import Any, Iterable

import torch


V4_QUINTIC_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
STABILIZING_QUINTIC_COEFFICIENTS = (2.0, -1.5, 0.5)


def _orthogonality_error(x: torch.Tensor) -> torch.Tensor:
    """Relative Frobenius error on the feasible side of a rectangular matrix."""
    work = x.float()
    if work.shape[0] <= work.shape[1]:
        gram = work @ work.mT
    else:
        gram = work.mT @ work
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return torch.linalg.vector_norm(gram - identity) / math.sqrt(max(gram.shape[0], 1))


@torch.no_grad()
def newton_schulz(
    matrix: torch.Tensor,
    *,
    method: str,
    iterations: int | None = None,
    first_stage_steps: int = 8,
    second_stage_steps: int = 2,
    first_stage_coefficients: tuple[float, float, float] = V4_QUINTIC_COEFFICIENTS,
    second_stage_coefficients: tuple[float, float, float] = STABILIZING_QUINTIC_COEFFICIENTS,
    standard_steps: int = 10,
    standard_coefficients: tuple[float, float, float] = STABILIZING_QUINTIC_COEFFICIENTS,
    eps: float = 1.0e-7,
    trace: bool = False,
) -> tuple[torch.Tensor, list[float]]:
    """Orthogonalize a 2-D matrix with report-aligned quintic Newton-Schulz steps."""
    if matrix.ndim != 2:
        raise ValueError(f"Newton-Schulz expects a 2-D tensor, got shape={tuple(matrix.shape)}")
    if method not in {"v4", "hybrid", "standard"}:
        raise ValueError(f"Unknown Newton-Schulz method: {method}")
    if not torch.isfinite(matrix).all():
        raise FloatingPointError("Newton-Schulz input contains NaN or Inf")

    work = matrix.float()
    max_abs = work.abs().max()
    frobenius_norm = (work / max_abs).norm() * max_abs if max_abs > 0 else max_abs
    x = matrix.to(dtype=torch.bfloat16 if matrix.is_cuda else torch.float32)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.mT
    x = x / frobenius_norm.clamp_min(eps).to(dtype=x.dtype)

    errors: list[float] = []
    if trace:
        errors.append(float(_orthogonality_error(x).item()))
    if method in {"v4", "hybrid"}:
        coefficients = [first_stage_coefficients] * first_stage_steps + [second_stage_coefficients] * second_stage_steps
    else:
        coefficients = [standard_coefficients] * standard_steps
    if iterations is not None:
        coefficients = coefficients[: int(iterations)]
    for a, b, c in coefficients:
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
        if trace:
            errors.append(float(_orthogonality_error(x).item()))
    if transposed:
        x = x.mT
    return x.to(dtype=matrix.dtype), errors


def _step_number(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.item())
    return int(value)


class InstrumentedAdamW(torch.optim.AdamW):
    """AdamW with low-overhead, interval-based update/parameter instrumentation."""

    def __init__(self, params: Iterable, *, metrics_interval: int = 100, **kwargs: Any) -> None:
        super().__init__(params, **kwargs)
        self.metrics_interval = max(int(metrics_interval), 1)
        self.instrumentation_step = 0
        self.last_step_metrics: dict[str, Any] = {}

    @torch.no_grad()
    def step(self, closure=None):
        self.instrumentation_step += 1
        sample = self.instrumentation_step == 1 or self.instrumentation_step % self.metrics_interval == 0
        grad_sq = torch.zeros((), device=self.param_groups[0]["params"][0].device, dtype=torch.float32) if sample else None
        if sample:
            for group in self.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        grad_sq += param.grad.detach().float().pow(2).sum()

        started = time.perf_counter()
        loss = super().step(closure)
        optimizer_time = time.perf_counter() - started

        if sample:
            update_sq = torch.zeros_like(grad_sq)
            parameter_sq = torch.zeros_like(grad_sq)
            for group in self.param_groups:
                lr = float(group["lr"])
                beta1, beta2 = group["betas"]
                eps = float(group["eps"])
                weight_decay = float(group["weight_decay"])
                decay = 1.0 - lr * weight_decay
                for param in group["params"]:
                    parameter_sq += param.detach().float().pow(2).sum()
                    state = self.state.get(param, {})
                    if not state or "exp_avg" not in state:
                        continue
                    step = _step_number(state["step"])
                    bias_correction1 = 1.0 - beta1**step
                    bias_correction2 = 1.0 - beta2**step
                    denom = state["exp_avg_sq"].float().sqrt() / math.sqrt(bias_correction2)
                    adaptive = (lr / bias_correction1) * state["exp_avg"].float() / (denom + eps)
                    # p_new = decay * p_old - adaptive
                    delta = (-adaptive - lr * weight_decay * param.detach().float()) / max(decay, 1.0e-12)
                    update_sq += delta.pow(2).sum()
            self.last_step_metrics = {
                "sample_step": self.instrumentation_step,
                "grad_norm": float(grad_sq.sqrt().item()),
                "update_norm": float(update_sq.sqrt().item()),
                "parameter_norm": float(parameter_sq.sqrt().item()),
                "optimizer_internal_time": optimizer_time,
                "newton_schulz_time": 0.0,
                "muon_matrix_count": 0,
            }
        return loss


class MuonWithAuxAdamW(torch.optim.Optimizer):
    """Muon for eligible matrices and AdamW for embeddings, vectors and scalars."""

    def __init__(
        self,
        params: Iterable,
        *,
        lr: float,
        weight_decay: float,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_method: str = "v4",
        ns_iterations: int = 5,
        ns_first_stage_steps: int = 8,
        ns_second_stage_steps: int = 2,
        ns_first_stage_coefficients: tuple[float, float, float] = V4_QUINTIC_COEFFICIENTS,
        ns_second_stage_coefficients: tuple[float, float, float] = STABILIZING_QUINTIC_COEFFICIENTS,
        ns_standard_steps: int = 10,
        ns_standard_coefficients: tuple[float, float, float] = STABILIZING_QUINTIC_COEFFICIENTS,
        update_rms_target: float = 0.18,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1.0e-8,
        metrics_interval: int = 100,
        diagnostics_interval: int = 500,
        diagnostics_max_matrices: int = 4,
        diagnostics_max_singular_values: int = 128,
    ) -> None:
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_method=ns_method,
            ns_iterations=ns_iterations,
            ns_first_stage_steps=ns_first_stage_steps,
            ns_second_stage_steps=ns_second_stage_steps,
            ns_first_stage_coefficients=ns_first_stage_coefficients,
            ns_second_stage_coefficients=ns_second_stage_coefficients,
            ns_standard_steps=ns_standard_steps,
            ns_standard_coefficients=ns_standard_coefficients,
            update_rms_target=update_rms_target,
            betas=adamw_betas,
            eps=adamw_eps,
        )
        super().__init__(params, defaults)
        self.metrics_interval = max(int(metrics_interval), 1)
        self.diagnostics_interval = max(int(diagnostics_interval), 1)
        self.diagnostics_max_matrices = max(int(diagnostics_max_matrices), 0)
        self.diagnostics_max_singular_values = max(int(diagnostics_max_singular_values), 1)
        self.instrumentation_step = 0
        self.last_step_metrics: dict[str, Any] = {}
        self.last_ns_diagnostics: list[dict[str, Any]] = []

    def _adamw_update(self, param: torch.Tensor, grad: torch.Tensor, group: dict[str, Any]) -> torch.Tensor:
        state = self.state[param]
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(param)
            state["exp_avg_sq"] = torch.zeros_like(param)
        state["step"] += 1
        beta1, beta2 = group["betas"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
        step = int(state["step"])
        denom = exp_avg_sq.sqrt().div_(math.sqrt(1.0 - beta2**step)).add_(group["eps"])
        return exp_avg.div(denom).mul(1.0 / (1.0 - beta1**step))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self.instrumentation_step += 1
        sample = self.instrumentation_step == 1 or self.instrumentation_step % self.metrics_interval == 0
        diagnose = self.instrumentation_step == 1 or self.instrumentation_step % self.diagnostics_interval == 0
        device = self.param_groups[0]["params"][0].device
        grad_sq = torch.zeros((), device=device, dtype=torch.float32)
        update_sq = torch.zeros_like(grad_sq)
        parameter_sq = torch.zeros_like(grad_sq)
        diagnostics: list[dict[str, Any]] = []
        ns_seconds = 0.0
        ns_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        muon_matrix_count = 0
        optimizer_started = time.perf_counter()

        for group in self.param_groups:
            for index, param in enumerate(group["params"]):
                if param.grad is None:
                    continue
                grad = param.grad.detach()
                if sample:
                    grad_sq += grad.float().pow(2).sum()
                use_muon = bool(group.get("use_muon", False)) and param.ndim in {2, 3}
                if use_muon:
                    muon_matrix_count += 1
                    state = self.state[param]
                    momentum_buffer = state.get("momentum_buffer")
                    if momentum_buffer is None:
                        momentum_buffer = state["momentum_buffer"] = torch.zeros_like(param)
                    momentum = float(group["momentum"])
                    momentum_buffer.mul_(momentum).add_(grad)
                    update_input = grad.add(momentum_buffer, alpha=momentum) if group["nesterov"] else momentum_buffer
                    trace_this = diagnose and len(diagnostics) < self.diagnostics_max_matrices
                    ns_started = time.perf_counter() if sample and device.type != "cuda" else None
                    event_pair = None
                    if sample and device.type == "cuda":
                        event_pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                        event_pair[0].record()
                    logical_matrices = update_input.unbind(0) if update_input.ndim == 3 else (update_input,)
                    logical_updates = []
                    errors = []
                    for logical_matrix in logical_matrices:
                        logical_update, logical_errors = newton_schulz(
                            logical_matrix,
                            method=str(group["ns_method"]),
                            iterations=None,
                            first_stage_steps=int(group["ns_first_stage_steps"]),
                            second_stage_steps=int(group["ns_second_stage_steps"]),
                            first_stage_coefficients=tuple(group["ns_first_stage_coefficients"]),
                            second_stage_coefficients=tuple(group["ns_second_stage_coefficients"]),
                            standard_steps=int(group["ns_standard_steps"]),
                            standard_coefficients=tuple(group["ns_standard_coefficients"]),
                            trace=trace_this and not errors,
                        )
                        rows, columns = logical_matrix.shape
                        logical_update.mul_(math.sqrt(max(rows, columns)) * float(group["update_rms_target"]))
                        logical_updates.append(logical_update)
                        if logical_errors:
                            errors = logical_errors
                    update = torch.stack(logical_updates) if update_input.ndim == 3 else logical_updates[0]
                    if event_pair is not None:
                        event_pair[1].record()
                        ns_events.append(event_pair)
                    elif ns_started is not None:
                        ns_seconds += time.perf_counter() - ns_started
                    if trace_this:
                        all_singular_values = torch.linalg.svdvals(grad.float())
                        positive = all_singular_values[all_singular_values > 1.0e-12]
                        condition = float((positive[0] / positive[-1]).item()) if positive.numel() else float("inf")
                        singular_values = all_singular_values[: self.diagnostics_max_singular_values]
                        diagnostics.append(
                            {
                                "step": self.instrumentation_step,
                                "parameter": getattr(self, "parameter_names", {}).get(
                                    id(param), f"{group.get('name', 'group')}.parameter_{index}"
                                ),
                                "shape": list(param.shape),
                                "method": str(group["ns_method"]),
                                "iterations": int(group["ns_iterations"]),
                                "singular_values": singular_values.cpu().tolist(),
                                "condition_number": condition,
                                "orthogonality_errors": errors,
                                "iterations_to_1e-3": next((i for i, value in enumerate(errors) if value < 1.0e-3), None),
                                "iterations_to_1e-5": next((i for i, value in enumerate(errors) if value < 1.0e-5), None),
                            }
                        )
                else:
                    update = self._adamw_update(param, grad, group)

                lr = float(group["lr"])
                weight_decay = float(group["weight_decay"])
                if weight_decay:
                    param.mul_(1.0 - lr * weight_decay)
                delta = update.mul(-lr)
                param.add_(delta)
                if sample:
                    decay = max(1.0 - lr * weight_decay, 1.0e-12)
                    old_parameter = (param.detach().float() - delta.float()) / decay
                    update_sq += (delta.float() - lr * weight_decay * old_parameter).pow(2).sum()
                    parameter_sq += param.detach().float().pow(2).sum()

        if sample:
            if ns_events:
                ns_events[-1][1].synchronize()
                ns_seconds = sum(start.elapsed_time(end) for start, end in ns_events) / 1000.0
            self.last_step_metrics = {
                "sample_step": self.instrumentation_step,
                "grad_norm": float(grad_sq.sqrt().item()),
                "update_norm": float(update_sq.sqrt().item()),
                "parameter_norm": float(parameter_sq.sqrt().item()),
                "optimizer_internal_time": time.perf_counter() - optimizer_started,
                "newton_schulz_time": ns_seconds,
                "muon_matrix_count": muon_matrix_count,
            }
        if diagnose:
            self.last_ns_diagnostics = diagnostics
        return loss
