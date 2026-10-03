"""Train a single model on an 80/20 stratified split.

    python scripts/train.py --model fusion --epochs 20 --data-dir data
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch.nn as nn  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import classification_report  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from ecg.data import CLASS_NAMES, FEATURE_COLS, SIG_COLS, Preprocessor, load_data, make_loader  # noqa: E402
from ecg.losses import FocalLoss, balanced_sampler, inverse_frequency_weights  # noqa: E402
from ecg.models import build_model  # noqa: E402
from ecg.train import get_device, predict, set_seed, train_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "cnn", "fusion"], default="fusion")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--imbalance", choices=["weighted_ce", "sampler_focal"], default="weighted_ce")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    train_df, _ = load_data(args.data_dir)
    X, y = train_df[FEATURE_COLS].values, train_df["label"].values
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)

    pre = Preprocessor()
    X_tr, X_va = pre.fit_transform(X_tr), pre.transform(X_va)
    if args.model == "cnn":                       # signal-only model ignores RR columns
        X_tr, X_va = X_tr[:, :len(SIG_COLS)], X_va[:, :len(SIG_COLS)]

    cw = inverse_frequency_weights(y_tr, device)
    if args.imbalance == "sampler_focal":
        tr_loader = make_loader(X_tr, y_tr, args.batch_size, sampler=balanced_sampler(y_tr))
        criterion = FocalLoss(alpha=cw, gamma=2.0)
    else:
        tr_loader = make_loader(X_tr, y_tr, args.batch_size, shuffle=True)
        criterion = nn.CrossEntropyLoss(weight=cw)
    va_loader = make_loader(X_va, y_va, 256)

    model = build_model(args.model, in_dim=X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(tr_loader))
    train_model(model, tr_loader, va_loader, opt, sched, criterion, args.epochs, device)

    y_pred, y_true = predict(model, va_loader, device)
    print(classification_report(y_true, y_pred, target_names=list(CLASS_NAMES.values()), digits=3))


if __name__ == "__main__":
    main()
