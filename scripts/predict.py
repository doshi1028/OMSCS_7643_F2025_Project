import argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from model import get_model
from pathlib import Path
import pandas as pd
from tqdm import tqdm

FEATURE_DIR = Path("output/features")
MODEL_DIR = Path("output/models")
PRED_DIR = Path("output/predictions")
PRED_DIR.mkdir(exist_ok=True)


# -------------------------------------------------------
# Dataset (same as train)
# -------------------------------------------------------
class CryptoDataset(Dataset):
    def __init__(self, X, seq_len=1):
        self.X = X
        self.seq_len = seq_len

        if seq_len > 1:
            self.X = self.build_sequences(X, seq_len)

    def build_sequences(self, X, seq_len):
        seq_data = []
        for i in range(len(X) - seq_len + 1):
            seq_data.append(X[i : i + seq_len])
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
def run_predict(model_name, seq_len=1, batch_size=64):
    print("\n=== Loading dataset for inference ===")

    X = np.load(FEATURE_DIR / "X.npy")
    y = np.load(FEATURE_DIR / "y.npy")  # Only for evaluation, not required

    # Handle sequence data
    if seq_len > 1:
        print(f"Using sequence length = {seq_len}")
        effective_len = len(X) - seq_len + 1
        y = y[seq_len - 1:]   # align y with last element in each sequence

    dataset = CryptoDataset(X, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    input_dim = X.shape[1]
    model = get_model(model_name, input_dim=input_dim)

    model_path = MODEL_DIR / f"{model_name}_best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    print(f"\n=== Predicting with {model_name.upper()} on {device} ===")

    preds = predict(model, loader, device)

    print("\n=== Building results table ===")
    df = pd.DataFrame({
        "pred": preds,
        "target": y[:len(preds)]
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
    parser.add_argument("--model", type=str, default="mlp",
                        choices=["mlp", "lstm", "transformer"])
    parser.add_argument("--seq_len", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    run_predict(
        model_name=args.model,
        seq_len=args.seq_len,
        batch_size=args.batch_size
    )
