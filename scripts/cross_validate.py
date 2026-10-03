"""5-fold CV + soft-vote ensemble; writes ``submission.csv``.

    python scripts/cross_validate.py --model fusion --data-dir data --out results/submission.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ecg.cv import cross_validate  # noqa: E402
from ecg.data import FEATURE_COLS, load_data  # noqa: E402
from ecg.train import get_device, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["mlp", "fusion"], default="fusion")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/submission.csv")
    args = ap.parse_args()

    set_seed(args.seed)
    train_df, test_df = load_data(args.data_dir)
    _, test_prob, _ = cross_validate(
        train_df[FEATURE_COLS].values, train_df["label"].values, test_df[FEATURE_COLS].values,
        model_name=args.model, n_folds=args.folds, epochs=args.epochs, seed=args.seed, device=get_device())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": test_df["id"], "label": test_prob.argmax(1)}).to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
