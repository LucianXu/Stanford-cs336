from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                p.data -= lr / math.sqrt(t + 1) * p.grad.data
                state["t"] = t + 1
        return loss

weights = 5 * torch.randn((10, 10))
for lr in [1e1, 1e2, 1e3]:
    weights_copy = torch.nn.Parameter(weights.clone())
    opt = SGD([weights_copy], lr=lr)
    losses = []
    for t in range(10):
        opt.zero_grad()
        loss = (weights_copy**2).mean()
        losses.append(loss.cpu().item())
        loss.backward()
        opt.step()
    print(f"lr = {lr}, losses = {losses}")