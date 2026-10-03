"""Training / evaluation loops."""
from __future__ import annotations

import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict(model, loader, device: str = "cpu"):
    """Return ``(y_pred, y_true)`` for a labelled loader."""
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        preds.extend(model(x.to(device)).argmax(1).cpu().numpy())
        trues.extend(y.numpy())
    return np.array(preds), np.array(trues)


@torch.no_grad()
def predict_proba(model, X: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Softmax probabilities for an unlabelled feature matrix."""
    model.eval()
    logits = model(torch.tensor(X, dtype=torch.float32, device=device))
    return torch.softmax(logits, dim=1).cpu().numpy()


def evaluate(model, loader, device: str = "cpu") -> float:
    y_pred, y_true = predict(model, loader, device)
    return f1_score(y_true, y_pred, average="macro")


def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion,
                epochs: int, device: str = "cpu", verbose: bool = True) -> list[float]:
    """Train and return per-epoch validation Macro F1.

    Note: the *best* epoch on a validation split is an optimistic estimate.
    Report the final-epoch score or an out-of-fold score for an unbiased number.
    """
    history, t0 = [], time.time()
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += loss.item() * xb.size(0)
        f1 = evaluate(model, val_loader, device)
        history.append(f1)
        if verbose:
            print(f"epoch {epoch + 1:2d}/{epochs} | train loss {total / len(train_loader.dataset):.4f} | val macro-F1 {f1:.4f}")
    if verbose:
        print(f"done in {time.time() - t0:.0f}s | best {max(history):.4f} | final {history[-1]:.4f}")
    return history
