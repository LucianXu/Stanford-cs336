import os
import torch
from typing import BinaryIO, IO

def SaveCheckpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int,
                     out: str | os.PathLike | BinaryIO | IO[bytes]):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, out)

def LoadCheckpoint(src: str | os.PathLike | BinaryIO | IO[bytes],
                    model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration