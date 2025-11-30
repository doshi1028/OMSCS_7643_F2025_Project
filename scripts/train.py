import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import get_model
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import json
import math

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
        self.seq_len = seq_len

        if seq_len > 1:
            self.X, self.y = self.build_sequences(X, y, seq_len)
        else:
            self.X, self.y = X, y

    @staticmethod
    def build_sequences(X, y, seq_len):
        """
        Build overlapping sequences for time series models and align labels with
        the last timestep in each window to avoid leakage.
        """
        seq_data = []
        targets = []
        for i in range(seq_len - 1, len(X)):
            seq_data.append(X[i - seq_len + 1 : i + 1])
            targets.append(y[i])
        return np.array(seq_data), np.array(targets)

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
def train_one_epoch(model, loader, optimizer, criterion, device, 
    grad_clip=None, ema=None, scheduler=None):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        preds = model(X)

        loss = criterion(preds, y)
        loss.backward()

        ## ---- Gradient clipping (optional)
        if grad_clip is not None:       
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()

        ## Step scheduler
        if scheduler is not None:
            scheduler.step()

        ## ---- EMA update
        if ema is not None:
            ema.update(model)


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
def train(args):
    model_name   = args.model
    seq_len      = args.seq_len
    batch_size   = args.batch_size
    lr           = args.lr
    epochs       = args.epochs
    weight_decay = args.weight_decay
    grad_clip    = args.grad_clip
    scheduler_name = args.scheduler
    warmup_pct   = args.warmup_pct
    ema_decay    = args.ema_decay
    seed         = args.seed
    print("\n=== 🚀 Loading Dataset ===")
    set_seed(seed) 
    X = np.load(FEATURE_DIR / "X.npy")
    y = np.load(FEATURE_DIR / "y.npy")
    meta_path = FEATURE_DIR / "dataset.parquet"
    if not meta_path.exists():
        raise FileNotFoundError("dataset.parquet not found under output/features/")
    meta = pd.read_parquet(meta_path)
    hours = pd.to_datetime(meta["hour"]).dt.tz_localize(None).to_numpy()
    if len(hours) != len(X):
        raise ValueError("dataset metadata rows do not match feature rows")
    cutoff = pd.to_datetime(args.train_end_date)
    mask = hours < cutoff
    if not mask.any():
        raise ValueError(f"No samples before cutoff {args.train_end_date}")
    X = X[mask]
    y = y[mask]
    hours = hours[mask]

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
    model = get_model(model_name, input_dim=input_dim, args=args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    ## ---------- EMA ----------
    ema = EMA(model, decay=ema_decay) if ema_decay else None

    ## ---------- Scheduler ----------
    total_steps = epochs * len(train_loader)
    warmup_steps = int(warmup_pct * total_steps)

    if scheduler_name == "cosine_warmup":
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
    else:
        scheduler = None

    print(f"\n=== 🧠 Training {model_name.upper()} on {device} ===")
    best_val = float("inf")
    patience, patience_counter = 5, 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, grad_clip=grad_clip, ema=ema, scheduler=scheduler
        )


        ## ---- Evaluate using EMA shadow weights
        if ema is not None:
            ema.apply_shadow(model)

        val_loss = eval_one_epoch(model, val_loader, criterion, device)

        if ema is not None:
            ema.restore(model)

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
            
            if ema is not None:
              # save EMA shadow weights
              torch.save(ema.shadow, MODEL_DIR / f"{model_name}_ema.pt")
            
            with open(MODEL_DIR / f"{model_name}_args.json", "w") as f:
                json.dump(vars(args), f, indent=4)
            torch.save(model.state_dict(), MODEL_DIR / f"{model_name}_best.pt")
        else:
            patience_counter += 1

            if patience_counter >= patience:
                print("   ⏹ Early stopping: no improvement.")
                break

    print(f"\n🎉 Training finished. Best val loss = {best_val:.6f}")

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.original[name]

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # linear warmup
            scale = step / self.warmup_steps
        else:
            # cosine decay
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.min_lr + scale * (base_lr - self.min_lr)
            for base_lr in self.base_lrs
        ]

def set_seed(seed):   
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# -------------------------------------------------------
#  CLI
# -------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ---- base ----
    parser.add_argument("--model", type=str, default="lr",
                        choices=["lr", "mlp", "lstm", "transformer","gru"])
    parser.add_argument("--seq_len", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)

    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--scheduler", type=str, default=None,
                        choices=[None, "cosine_warmup"])
    parser.add_argument("--warmup_pct", type=float, default=0.1)
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_end_date", type=str, default="2024-10-01",
                        help="Use samples strictly before this date for training/validation (YYYY-MM-DD).")
    parser.add_argument("--num_layers", type=int, default=2)

    # ---- model-specific arguments ----
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    # LSTM
    parser.add_argument("--lstm_proj_dim", type=int, default=128)
    parser.add_argument("--lstm_use_layernorm", type=int, default=1)
    parser.add_argument("--lstm_use_attention", type=int, default=1)

    # Transformer
    parser.add_argument("--tf_d_model", type=int, default=128)
    parser.add_argument("--tf_heads", type=int, default=4)
    parser.add_argument("--tf_layers", type=int, default=4)
    parser.add_argument("--tf_ff_dim", type=int, default=256)
    parser.add_argument("--tf_dropout", type=float, default=0.1)
    parser.add_argument("--tf_learnable_pos", type=int, default=1)
    parser.add_argument("--tf_use_cls_token", type=int, default=1)
    parser.add_argument("--tf_pool", type=str, default="attention",
                        choices=["attention", "cls", "mean", "last"])
    parser.add_argument("--tf_embed_scale", type=float, default=1.0)

    args = parser.parse_args()

    ## Call train function
    train(args)
