import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, timestamps=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.timestamps = timestamps

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.timestamps is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.timestamps[idx]


##Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PriceTransformer(nn.Module):
    def __init__(
        self,
        feature_dim,
        embed_scale=0.01,
        d_model=128,
        nhead=8,
        num_layers=3,
        dim_ff=256,
        dropout=0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.embed_scale = embed_scale

        # Pre-norm stabilizes mult-modal scale mismatch
        self.pre_norm = nn.LayerNorm(feature_dim)

        # Map input feature_dim → d_model
        self.input_proj = nn.Linear(feature_dim, d_model)

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer Encoder with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    ## Generate causal (future-masked) attention mask
    def _generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask.to(device)

    ## Forward pass
    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        B, T, _ = x.size()
        device = x.device

        # Scale embedding part inside x (only emb cols)
        x = x * self.embed_scale

        # Pre-normalization for stability
        x = self.pre_norm(x)

        # ---- Embedding Token ----
        # latest time step / last token as embedding token
        emb_token = x[:, -1:, :]        # (B, 1, feature_dim)

        # sequence = [t-12 ... t-1, emb-token]
        x = torch.cat([x, emb_token], dim=1)   # (B, T+1, feature_dim)
        T_new = T + 1

        # Input projection
        x = self.input_proj(x)   # (B, T+1, d_model)

        # Positional encoding
        x = self.pos_encoder(x)

        # Causal mask
        mask = self._generate_causal_mask(T_new, device)

        # Transformer encode
        h = self.transformer(x, mask=mask)

        # Only use last token for prediction
        out = self.head(h[:, -1, :])  # (B, 1)
        return out.squeeze(1)


## Training
def train_transformer(
    X, y, timestamps=None,
    batch_size=32,
    lr=1e-5,
    epochs=20
):

    feature_dim = X.shape[-1]
    model = PriceTransformer(feature_dim=feature_dim).to(device)

    ## 80-20 split
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    ts_train = timestamps[:split_idx] if timestamps is not None else None
    ts_val   = timestamps[split_idx:] if timestamps is not None else None

    train_loader = DataLoader(
        TimeSeriesDataset(X_train, y_train, ts_train),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        TimeSeriesDataset(X_val, y_val, ts_val),
        batch_size=batch_size
    )

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        ## Training
        model.train()
        train_loss = 0

        for batch in train_loader:
            Xb, yb = batch[:2]
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        ## Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                Xb, yb = batch[:2]
                Xb = Xb.to(device)
                yb = yb.to(device)

                pred = model(Xb)
                val_loss += loss_fn(pred, yb).item()

        print(f"Epoch {epoch:02d} | Train Loss {train_loss/len(train_loader):.6f} | "
              f"Val Loss {val_loss/len(val_loader):.6f}")

        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))

    return model, train_losses, val_losses


## Prediction
def predict(model, X):
    model.eval()
    X = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(X).cpu().numpy()
