import math
from collections.abc import Callable, Iterable

import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
__all__ = ["AdamW", "cross_entropy", "get_batch", "get_lr_cosine_schedule", "gradient_clipping", "load_checkpoint", "save_checkpoint"]



def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    stable_logits = logits - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(stable_logits), dim=-1))
    target_logits = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    losses = max_logits.squeeze(-1) + log_sum_exp - target_logits
    return losses.mean()


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        super().__init__(
            params,
            {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay},
        )

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                grad = parameter.grad.data
                state = self.state[parameter]
                if not state:
                    state["t"] = 1
                    state["m"] = torch.zeros_like(parameter.data)
                    state["v"] = torch.zeros_like(parameter.data)

                t = state["t"]
                m = state["m"]
                v = state["v"]
                parameter.data -= lr * weight_decay * parameter.data
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_corrected_lr = lr * math.sqrt(1 - beta2**t) / (1 - beta1**t)
                parameter.data -= bias_corrected_lr * m / (torch.sqrt(v) + eps)
                state["t"] = t + 1

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (
            max_learning_rate - min_learning_rate
        )
    return min_learning_rate


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    parameters_with_grad = [parameter for parameter in parameters if parameter.grad is not None]
    total_norm_squared = sum(torch.sum(parameter.grad**2) for parameter in parameters_with_grad)
    total_norm = torch.sqrt(total_norm_squared)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for parameter in parameters_with_grad:
            parameter.grad.mul_(scale)
