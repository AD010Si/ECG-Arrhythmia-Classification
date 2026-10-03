"""Class-imbalance helpers (Fusion beats make up well under 1% of the data)."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler


def inverse_frequency_weights(y: np.ndarray, device="cpu") -> torch.Tensor:
    counts = np.bincount(y)
    return torch.tensor(len(y) / (len(counts) * counts), dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) with optional per-class weights ``alpha``."""

    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def balanced_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Oversample rare classes so each batch is roughly class-balanced."""
    w = inverse_frequency_weights(y).cpu().numpy()[y]
    return WeightedRandomSampler(weights=w, num_samples=len(w), replacement=True)
