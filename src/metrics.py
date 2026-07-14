from __future__ import annotations

from dataclasses import dataclass, field

import torch


def compute_grad_norm(parameters) -> float:
    total = torch.tensor(0.0)
    found = False
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        total = total + grad.float().pow(2).sum()
        found = True
    if not found:
        return 0.0
    return float(total.sqrt().item())


@dataclass
class MeanAccumulator:
    total: float = 0.0
    count: int = 0
    history: list[float] = field(default_factory=list)

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n
        self.history.append(float(value))

    @property
    def average(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count
