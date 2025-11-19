import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import get_model
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


FEATURE_DIR = Path("output/features")
MODEL_DIR = Path("output/models")
MODEL_DIR.mkdir(exist_ok=True)


# -------------------------------------------------------
#  Dataset
# -------------------------------------------------------
class CryptoDataset(Dataset):
    def __init__(self, X, y, seq_len=1):
        """
        X: (N, input_dim)  OR sequence data (N, seq_len, input_dim)
        y: (N,)
        seq_len: if > 1, reshape X into sequences for LSTM/Transformer
        """
        self.X = X
        self.y = y
        self.seq_len = seq_len

        if seq_len > 1:
            self.X = self.build_sequences(X, seq_len)

    def build_sequences(self, X, seq_len):
        """
        Build overlapping sequences for time series models.
        X shape: (N, input_dim)
        Output: (N - seq_len + 1, seq_len, input_dim)
        """
        seq_data = []
        for i in range(len(X) - seq_len + 1):
            seq_data.append(X[i : i + seq_len])
        return np.array(seq_data)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


# -------------------------------------------------------
#  Training Loop
# -------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(X)

        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


# -------------------------------------------------------
#  Main Train Function
# -------------------------------------------------------
def train(model_name, seq_len=1, batch_size=64, lr=1e-4, epochs=30):
    print("\n=== 🚀 Loading Dataset ===")

    X = np.load(FEATURE_DIR / "X.npy")
    y = np.load(FEATURE_DIR / "y.npy")

    input_dim = X.shape[1]

    # Train-val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, shuffle=False  # time series → do NOT shuffle
    )

    train_ds = CryptoDataset(X_train, y_train, seq_len=seq_len)
    val_ds = CryptoDataset(X_val, y_val, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    print(f"✔ Train: {len(train_ds)} samples")
    print(f"✔ Val: {len(val_ds)} samples")

    # Load model
    model = get_model(model_name, input_dim=input_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"\n=== 🧠 Training {model_name.upper()} on {device} ===")
    best_val = float("inf")
    patience, patience_counter = 5, 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0

            torch.save(
                model.state_dict(),
                MODEL_DIR / f"{model_name}_best.pt"
            )
            print("   💾 Saved new best model!")

        else:
            patience_counter += 1

            if patience_counter >= patience:
                print("   ⏹ Early stopping: no improvement.")
                break

    print(f"\n🎉 Training finished. Best val loss = {best_val:.6f}")


# -------------------------------------------------------
#  CLI
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mlp",
                        choices=["mlp", "lstm", "transformer"])
    parser.add_argument("--seq_len", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)

    args = parser.parse_args()

    train(
        model_name=args.model,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs
    )
