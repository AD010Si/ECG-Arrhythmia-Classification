"""Data loading, preprocessing and PyTorch datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

N_TIMESTEPS = 250
RR_COLS = ["pre_rr", "post_rr", "rr_ratio"]
SIG_COLS = [f"sig_{i}" for i in range(N_TIMESTEPS)]
FEATURE_COLS = SIG_COLS + RR_COLS
CLASS_NAMES = {0: "Normal", 1: "Supraventricular", 2: "Ventricular", 3: "Fusion"}


def load_data(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read ``train.csv`` and ``test.csv`` from ``data_dir``."""
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "test.csv")


class ECGDataset(Dataset):
    """Tensor dataset; yields ``(x, y)`` or just ``x`` when labels are absent."""

    def __init__(self, X: np.ndarray, y: np.ndarray | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]


class Preprocessor:
    """Median-impute (RR features have NaNs) then standardise.

    Fit on training data only to avoid leaking validation/test statistics.
    """

    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(self.imputer.fit_transform(X))

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(self.imputer.transform(X))


def make_loader(X, y=None, batch_size=128, shuffle=False, sampler=None) -> DataLoader:
    return DataLoader(ECGDataset(X, y), batch_size=batch_size,
                      shuffle=shuffle if sampler is None else False, sampler=sampler)
