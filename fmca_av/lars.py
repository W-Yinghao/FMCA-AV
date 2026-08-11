"""Dependency-free LARS optimizer used by the official FroSSL recipe.

This is a compact adaptation of ``solo/utils/lars.py`` in the pinned FroSSL
repository (MIT package metadata; itself derived from Lightning Bolts).
"""

from typing import Any, Iterable, Optional

import torch
from torch import Tensor
from torch.optim import Optimizer


class LARS(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        eta: float = 0.001,
        eps: float = 1e-8,
        clip_lr: bool = False,
        exclude_bias_n_norm: bool = False,
    ) -> None:
        if lr < 0 or momentum < 0 or weight_decay < 0:
            raise ValueError("LARS lr, momentum, and weight_decay must be non-negative")
        super().__init__(params, {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "eta": eta,
            "eps": eps,
            "clip_lr": clip_lr,
            "exclude_bias_n_norm": exclude_bias_n_norm,
        })

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                update = parameter.grad
                if parameter.ndim != 1 or not group["exclude_bias_n_norm"]:
                    parameter_norm = torch.linalg.vector_norm(parameter)
                    gradient_norm = torch.linalg.vector_norm(update)
                    if parameter_norm > 0 and gradient_norm > 0:
                        trust = group["eta"] * parameter_norm / (
                            gradient_norm + group["weight_decay"] * parameter_norm + group["eps"]
                        )
                        if group["clip_lr"]:
                            trust = torch.minimum(trust / group["lr"], trust.new_ones(()))
                        update = update.add(parameter, alpha=group["weight_decay"]) * trust
                if group["momentum"]:
                    state = self.state[parameter]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = update.detach().clone()
                    else:
                        state["momentum_buffer"].mul_(group["momentum"]).add_(update)
                    update = state["momentum_buffer"]
                parameter.add_(update, alpha=-group["lr"])
        return loss
