import torch
import torch.nn as nn
from einops import rearrange, einsum

def CrossEntropy(x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    x_max = x.max(dim=-1, keepdim=True).values
    x_exp = torch.exp(x - x_max).sum(dim=-1, keepdim=True)
    x_target = x.gather(dim = -1, index = targets.unsqueeze(-1))
    loss = (x_max - x_target + torch.log(x_exp)).squeeze(-1)
    return loss.mean()
