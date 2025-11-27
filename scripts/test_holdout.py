import argparse
from pathlib import Path

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from dataclasses import asdict

from model import get_model
from evaluate import (
    compute_regression_metrics,
    simulate_trading_strategy,
    plot_strategy_curves,
    to_native_dict,
)

FEATURE_DIR = Path("output/features")
MODEL_DIR = Path("output/models")
REPORT_DIR = Path("output/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


class HoldoutDataset(Dataset):
    def __init__(self, X, seq_len=1):
        self.seq_len = seq_len
        if seq_len > 1:
            self.X = self.build_sequences(X, seq_len)
        else:
            self.X = X

    @staticmethod
    def build_sequences(X, seq_len):
        seq_data = []
        for i in range(seq_len - 1, len(X)):
            seq_data.append(X[i - seq_len + 1 : i + 1])
        return np.array(seq_data)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32)


def evaluate_holdout(args):
    cutoff = pd.to_datetime(args.cutoff_date)

    X = np.load(FEATURE_DIR / "X.npy")
    y = np.load(FEATURE_DIR / "y.npy")
    meta_path = FEATURE_DIR / "dataset.parquet"
    if not meta_path.exists():
        raise FileNotFoundError("dataset.parquet not found under output/features/")
    meta = pd.read_parquet(meta_path)
    hours = pd.to_datetime(meta["hour"]).dt.tz_localize(None).to_numpy()
    if len(hours) != len(X):
        raise ValueError("dataset metadata rows do not match feature rows")

    holdout_mask = hours >= cutoff
    if not holdout_mask.any():
        raise ValueError("No holdout samples found for the specified cutoff.")

    X_holdout = X[holdout_mask]
    y_holdout = y[holdout_mask]
    timestamps = hours[holdout_mask]

    if args.seq_len > 1:
        dataset = HoldoutDataset(X_holdout, seq_len=args.seq_len)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        timestamps = timestamps[args.seq_len - 1 :]
    else:
        dataset = HoldoutDataset(X_holdout, seq_len=1)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    input_dim = X.shape[1]
    config_path = MODEL_DIR / f"{args.model}_args.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing training config: {config_path}")
    cfg = argparse.Namespace()
    with open(config_path, "r") as f:
        cfg.__dict__.update(json.load(f))

    model = get_model(args.model, input_dim=input_dim, args=cfg)
    checkpoint = MODEL_DIR / f"{args.model}_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.extend(model(batch).cpu().numpy().tolist())
    preds = np.array(preds)

    if len(preds) != len(y_holdout[: len(preds)]):
        raise ValueError("Prediction length mismatch on holdout set.")

    y_eval = y_holdout[: len(preds)]
    timestamps = timestamps[: len(preds)]

    train_returns = y[~holdout_mask]
    if not len(train_returns):
        raise ValueError("No training returns available to compute threshold.")
    threshold = np.percentile(np.abs(train_returns), args.signal_percentile)

    regression = compute_regression_metrics(y_eval, preds)
    strategy = simulate_trading_strategy(y_eval, preds, threshold)

    positions = np.where(
        preds >= threshold,
        1,
        np.where(preds <= -threshold, -1, 0),
    )
    returns = positions * y_eval
    plot_strategy_curves("holdout_model", positions, returns, timestamps, REPORT_DIR)

    print("Holdout metrics:")
    print(to_native_dict(asdict(regression)))
    print(to_native_dict(asdict(strategy)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the holdout set.")
    parser.add_argument("--model", type=str, default="lr",
                        choices=["lr", "mlp", "lstm", "transformer", "gru"])
    parser.add_argument("--seq_len", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cutoff_date", type=str, default="2024-10-01")
    parser.add_argument("--signal_percentile", type=float, default=60.0)
    args = parser.parse_args()

    evaluate_holdout(args)
