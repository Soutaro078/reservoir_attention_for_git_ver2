import torch
from typing import Tuple


def get_mask(shape: Tuple) -> torch.Tensor:
    return torch.triu(torch.ones(shape) * float('-inf'), diagonal=1)
