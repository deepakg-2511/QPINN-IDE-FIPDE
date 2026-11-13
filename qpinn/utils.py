"""Small helpers shared by the whole package."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """dy/dx for a column y(x); keeps the graph so it can be differentiated again."""
    return torch.autograd.grad(y.sum(), x, create_graph=True)[0]


def col(a, device, requires_grad=False) -> torch.Tensor:
    """numpy/list -> float32 column tensor (N, 1)."""
    t = torch.as_tensor(np.asarray(a, dtype=np.float32).reshape(-1, 1), device=device)
    return t.requires_grad_(requires_grad)


def mse(r: torch.Tensor) -> torch.Tensor:
    return torch.mean(r ** 2)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
