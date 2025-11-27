import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import get_model
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
import types

FEATURE_DIR = Path("output/features")
MODEL_DIR = Path("output/models")
PRED_DIR = Path("output/predictions")
PRED_DIR.mkdir(exist_ok=True)


# -------------------------------------------------------
# Dataset (same as train)
# -------------------------------------------------------
class CryptoDataset(Dataset):
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


# -------------------------------------------------------
# Prediction Loop
# -------------------------------------------------------
def predict(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for X in tqdm(loader, desc="Predicting"):
            X = X.to(device)
            y_hat = model(X)
            preds.extend(y_hat.cpu().numpy().tolist())

    return np.array(preds)


# -------------------------------------------------------
# Main Function
# -------------------------------------------------------
def run_predict(args):
    model_name   = args.model
    seq_len      = args.seq_len
    batch_size   = args.batch_size
    seed         = args.seed
    cutoff_date  = pd.to_datetime(args.cutoff_date)
    
    print("\n=== Loading dataset for inference ===")

    X = np.load(FEATURE_DIR / "X.npy")
    y = np.load(FEATURE_DIR / "y.npy")  # Only for evaluation, not required
    meta_path = FEATURE_DIR / "dataset.parquet"
    if not meta_path.exists():
        raise FileNotFoundError("dataset.parquet not found under output/features/")
    meta = pd.read_parquet(meta_path)
    timestamps = pd.to_datetime(meta["hour"]).dt.tz_localize(None).to_numpy()
    if len(timestamps) != len(X):
        raise ValueError("dataset metadata rows do not match feature rows")

    # Handle sequence data
    if seq_len > 1:
        print(f"Using sequence length = {seq_len}")
        y = y[seq_len - 1 :]   # align y with last element in each sequence
        timestamps = timestamps[seq_len - 1 :]

    dataset = CryptoDataset(X, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    input_dim = X.shape[1]

    # === Load training config ===
    config_path = MODEL_DIR / f"{model_name}_args.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing training config: {config_path}")

    with open(config_path, "r") as f:
        saved_cfg = json.load(f)

    # turn dict → object with attributes
    class Struct: pass
    model_args = Struct()
    for k, v in saved_cfg.items():
        setattr(model_args, k, v)

    model = get_model(model_name, input_dim=input_dim, args=model_args)

    model_path = MODEL_DIR / f"{model_name}_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    print(f"\n=== Predicting with {model_name.upper()} on {device} ===")
    # === If EMA was used in training, load EMA weights ===
    ema_path = MODEL_DIR / f"{model_name}_ema.pt"
    if ema_path.exists():
      print("Loading EMA shadow weights...")
      ema_state = torch.load(ema_path, map_location=device)
      for name, param in model.named_parameters():
          if name in ema_state:
              param.data.copy_(ema_state[name])

    preds = predict(model, loader, device)

    valid_len = len(preds)
    timestamps = timestamps[:valid_len]
    subset = np.where(
        timestamps < cutoff_date,
        "train",
        "holdout",
    )

    print("\n=== Building results table ===")
    df = pd.DataFrame({
        "pred": preds,
        "target": y[:len(preds)],
        "subset": subset,
        "timestamp": timestamps
    })

    out_fp = PRED_DIR / f"predictions_{model_name}.csv"
    df.to_csv(out_fp, index=False)

    print(f"Saved predictions → {out_fp}")
    print(f"Done! {len(df)} predictions generated.")

    return df


# -------------------------------------------------------
# CLI
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ---- basic ----
    parser.add_argument("--model", type=str, default="lr",
                        choices=["lr", "mlp", "lstm", "transformer", "gru"])
    parser.add_argument("--seq_len", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoff_date", type=str, default="2024-10-01",
                        help="Dates before this belong to the training subset; later dates form the holdout.")

    args = parser.parse_args()

    run_predict(args)
