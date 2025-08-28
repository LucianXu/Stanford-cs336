from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float] = (0.9, 0.999), 
                 weight_decay: float = 1e-2, eps: float = 1e-8):
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Hyperparameters
                lr = group['lr']
                beta1, beta2 = group['betas']
                weight_decay = group['weight_decay']
                eps = group['eps']

                # Get state
                g = p.grad.data
                state = self.state[p]
                step = state.get("step", 0)
                exp_avg = state.get("exp_avg", torch.zeros_like(p.data))
                exp_avg_sq = state.get("exp_avg_sq", torch.zeros_like(p.data))

                # Update
                step += 1
                """
                exp_avg = beta1 * exp_avg + (1 - beta1) * g
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * g * g
                lr_t = lr * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)
                p.data -= lr_t * exp_avg / (torch.sqrt(exp_avg_sq) + eps)
                p.data -= lr * weight_decay * p
                """
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                lr_t = lr * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)
                p.data.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-lr_t)
                p.data.add_(p, alpha=-lr * weight_decay)

                # Save state
                state["step"] = step
                state["exp_avg"] = exp_avg
                state["exp_avg_sq"] = exp_avg_sq
        return loss

def LearningRateScheduler(t: int, lr_max: float, lr_min: float, T_w: int, T_c: int) -> float:
    """
    "Warm-up, Cosine annealing, Post-annealing" schedule.
    Why not "Warm-up, Stable, Decay"?
    """
    if t < T_w:
        return lr_max * t / T_w
    elif t <= T_c:
        return lr_min + 0.5 * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w))) * (lr_max - lr_min)
    else:
        return lr_min

def GraidentClipping(params: Iterable, Max_norm: float, eps: float = 1e-6):
    norm = 0.0
    for p in params:
        if p.grad is None:
            continue
        norm += p.grad.data.norm(2) ** 2
    norm = norm.sqrt()
    if norm > Max_norm:
        for p in params:
            if p.grad is None:
                continue
            p.grad.data.mul_(Max_norm / (norm + eps))