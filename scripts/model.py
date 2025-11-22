import torch
import torch.nn as nn


# ------------------------------------------------------
# 0. LINEAR REGRESSION BASELINE
# ------------------------------------------------------
class LinearRegressor(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        if x.dim() > 2:
            # Flatten sequences before the linear head
            x = x[:, -1, :]
        return self.linear(x).squeeze(-1)


# ------------------------------------------------------
# 1. MLP BASELINE
# ------------------------------------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        """
        x: (batch, input_dim)
        """
        return self.net(x).squeeze(-1)



# ------------------------------------------------------
# 2. LSTM MODEL (supports sequences)
# ------------------------------------------------------
class LSTMRegressor(nn.Module):
    def __init__(self,
                 input_dim=768,
                 hidden_dim=256,
                 num_layers=2,
                 dropout=0.1,
                 bidirectional=False):
        """
        If used without sequence input, input will be reshaped to (batch, 1, input_dim)
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.regressor = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, 1)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim) or (batch, input_dim)
        """
        if x.dim() == 2:
            # If no sequence, treat as seq_len=1
            x = x.unsqueeze(1)

        out, (h_n, _) = self.lstm(x)

        # h_n: (num_layers * num_directions, batch, hidden)
        last_hidden = h_n[-1]  # last layer, last direction

        return self.regressor(last_hidden).squeeze(-1)



# ------------------------------------------------------
# 3. TRANSFORMER ENCODER MODEL
# ------------------------------------------------------
class TransformerRegressor(nn.Module):
    def __init__(self,
                 input_dim=768,
                 n_heads=8,
                 hidden_dim=256,
                 num_layers=2,
                 dropout=0.1):
        super().__init__()

        # Project input dim to model dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
            dim_feedforward=hidden_dim * 4
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        """
        x: (batch, seq_len, input_dim) or (batch, input_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_proj(x)  # (batch, seq_len, hidden_dim)
        out = self.transformer(x)  # (batch, seq_len, hidden_dim)

        # Use last token output
        last_token = out[:, -1, :]  # (batch, hidden_dim)

        return self.regressor(last_token).squeeze(-1)



# ------------------------------------------------------
#  Helper function to load a model by name
# ------------------------------------------------------
def get_model(model_name, input_dim=768):
    model_name = model_name.lower()

    if model_name == "lr":
        return LinearRegressor(input_dim=input_dim)

    if model_name == "mlp":
        return MLPRegressor(input_dim=input_dim)

    elif model_name == "lstm":
        return LSTMRegressor(input_dim=input_dim)

    elif model_name == "transformer":
        return TransformerRegressor(input_dim=input_dim)

    else:
        raise ValueError(f"Unknown model name: {model_name}")
