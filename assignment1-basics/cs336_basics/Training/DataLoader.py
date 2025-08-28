import torch
import numpy.typing as npt

def GetBatch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    length = len(dataset)
    inputs = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    targets = torch.zeros((batch_size, context_length), dtype=torch.long, device=device)
    for i in range(batch_size):
        start_idx = torch.randint(0, length - context_length, (1,)).item()
        inputs[i] = torch.tensor(dataset[start_idx : start_idx + context_length], dtype=torch.long, device=device)
        targets[i] = torch.tensor(dataset[start_idx + 1 : start_idx + context_length + 1], dtype=torch.long, device=device)
    return inputs, targets