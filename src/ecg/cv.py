"""Stratified K-fold training with soft-vote test ensembling."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from .data import Preprocessor, make_loader
from .losses import inverse_frequency_weights
from .models import build_model
from .train import predict_proba, set_seed, train_model


def cross_validate(X, y, X_test, model_name="fusion", n_folds=5, epochs=15,
                   lr=1e-3, batch_size=128, seed=42, device="cpu"):
    """Returns ``(oof_pred, test_proba_mean, fold_scores)``.

    The preprocessing pipeline is fit inside each fold on that fold's training
    rows only, so no validation statistics leak into training.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(X), dtype=int)
    test_prob = np.zeros((len(X_test), len(np.unique(y))))
    scores = []

    for fold, (tr, va) in enumerate(skf.split(X, y), start=1):
        print(f"\n--- Fold {fold}/{n_folds} ---")
        set_seed(seed + fold)
        pre = Preprocessor()
        X_tr, X_va, X_te = pre.fit_transform(X[tr]), pre.transform(X[va]), pre.transform(X_test)

        tr_loader = make_loader(X_tr, y[tr], batch_size, shuffle=True)
        va_loader = make_loader(X_va, y[va], 256)
        model = build_model(model_name, in_dim=X.shape[1]).to(device)
        criterion = nn.CrossEntropyLoss(weight=inverse_frequency_weights(y[tr], device))
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(tr_loader))

        train_model(model, tr_loader, va_loader, opt, sched, criterion, epochs, device)

        oof[va] = predict_proba(model, X_va, device).argmax(1)
        test_prob += predict_proba(model, X_te, device) / n_folds
        scores.append(f1_score(y[va], oof[va], average="macro"))
        print(f"Fold {fold} macro-F1: {scores[-1]:.4f}")

    print(f"\nper-fold: {np.round(scores, 4)} | mean {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print(f"out-of-fold macro-F1: {f1_score(y, oof, average='macro'):.4f}")
    return oof, test_prob, scores
