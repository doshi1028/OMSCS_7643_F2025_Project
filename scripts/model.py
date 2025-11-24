import torch
import torch.nn as nn
import math

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
    def __init__(
        self,
        input_dim=768,
        hidden_dim=256,
        proj_dim=128,         
        num_layers=2,
        dropout=0.1,
        use_layernorm=True,
        use_attention=True,   
    ):
        super().__init__()

        self.use_layernorm = use_layernorm
        self.use_attention = use_attention

        ## Input dropout
        self.dropout_in = nn.Dropout(dropout)

        ## Multi-layer LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        ## LayerNorm (optional)
        if use_layernorm:
            self.ln = nn.LayerNorm(hidden_dim)

        ## Optional Attention pooling
        if use_attention:
            self.att_w = nn.Linear(hidden_dim, 1)

        ## Projection layer: hidden_dim → proj_dim
        self.proj = nn.Linear(hidden_dim, proj_dim)

        ## MLP head
        self.regressor = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim // 2, 1)
        )

        ## Orthogonal init
        self.reset_parameters()

    def reset_parameters(self):
      for name, param in self.lstm.named_parameters():
          if "weight_hh" in name:  # recurrent weights
              nn.init.orthogonal_(param)

    def forward(self, x):
        # Allow input shape (batch, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Input dropout
        x = self.dropout_in(x)

        # LSTM forward
        out, (h_n, _) = self.lstm(x)

        # last hidden of last layer
        last = h_n[-1]

        # LayerNorm
        if self.use_layernorm:
            last = self.ln(last)

        # Attention Pooling (optional)
        if self.use_attention:
            # out: (batch, seq, hidden)
            att_score = torch.softmax(self.att_w(out).squeeze(-1), dim=1)
            att_vec   = torch.sum(att_score.unsqueeze(-1) * out, dim=1)
            last = last + att_vec    # residual add

        # projection + head
        z = self.proj(last)
        return self.regressor(z).squeeze(-1)



# ------------------------------------------------------
# 3. TRANSFORMER ENCODER MODEL
# ------------------------------------------------------
class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        max_len=512,

        use_learnable_pos=False,
        use_cls_token=True,
        pool_type="attention",   # <=== default attention pooling

        embed_scale=1.0
    ):
        super().__init__()

        self.pool_type = pool_type
        self.use_cls_token = use_cls_token
        self.embed_scale = embed_scale
        self.d_model = d_model

        ## Pre-normalization for stability
        self.pre_norm = nn.LayerNorm(input_dim)

        ## Linear projection
        self.input_proj = nn.Linear(input_dim, d_model)

        ## Positional encodings
        if use_learnable_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)
        else:
            self.register_buffer("pos_embedding", self._build_sinusoid(max_len, d_model))

        ## Optional CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.01)

        ## Transformer Encoder (PreNorm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        ##  Attention Pooling Block
        if pool_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh(),
                nn.Linear(d_model, 1)     # score per timestep
            )

        ## Output MLP head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        ## Weight init
        self._init_weights()


    # sinusoid
    def _build_sinusoid(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "weight" in name:
                nn.init.xavier_uniform_(p)

    # --------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------
    def forward(self, x):
        # x: (B, T, C)
        B, T, _ = x.shape
        device = x.device

        # PreNorm
        x = self.pre_norm(x)

        # scale embeddings
        x = x * self.embed_scale

        # project to d_model
        x = self.input_proj(x)

        # CLS token
        if self.use_cls_token:
            cls = self.cls_token.expand(B, 1, self.d_model)
            x = torch.cat([cls, x], dim=1)
            pos = self.pos_embedding[:, :T+1, :]
        else:
            pos = self.pos_embedding[:, :T, :]

        # add position encoding
        x = x + pos.to(device)

        # Transformer encoding
        h = self.transformer(x)  

        ## Attention Pooling
        if self.pool_type == "attention":
            # attn_score: (B, L, 1)
            attn_score = self.attention(h)
            attn_weights = torch.softmax(attn_score, dim=1)  # causal safe, no future look back
            pooled = (attn_weights * h).sum(dim=1)

        elif self.pool_type == "cls":
            pooled = h[:, 0, :]

        elif self.pool_type == "mean":
            pooled = h.mean(dim=1)

        else:  # "last"
            pooled = h[:, -1, :]

        return self.head(pooled).squeeze(-1)

# ------------------------------------------------------
# 4. GRU Regressor MODEL
# ------------------------------------------------------
class GRURegressor(nn.Module):
    def __init__(self,
                 input_dim=768,
                 hidden_dim=256,
                 num_layers=2,
                 dropout=0.2):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # attention pooling (causal-safe)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.apply(self._init_weights)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # normalize input
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # recurrent encoder
        out, h_n = self.gru(x)   # out: (batch, seq, hidden)

        # attention pooling (weight only past timesteps)
        attn_scores = self.attention(out)    # (batch, seq, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = (out * attn_weights).sum(dim=1)   # weighted average

        return self.head(pooled).squeeze(-1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)



# ------------------------------------------------------
#  Helper function to load a model by name
# ------------------------------------------------------
def get_model(model_name, input_dim, args):
    model_name = model_name.lower()

    # ---------------------------
    # Linear Regression
    # ---------------------------
    if model_name == "lr":
        return LinearRegressor(input_dim=input_dim)

    # ---------------------------
    # MLP
    # ---------------------------
    if model_name == "mlp":
        return MLPRegressor(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )

    # ---------------------------
    # LSTM
    # ---------------------------
    if model_name == "lstm":
        return LSTMRegressor(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            proj_dim=args.lstm_proj_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            use_layernorm=bool(args.lstm_use_layernorm),
            use_attention=bool(args.lstm_use_attention),
        )

    # ---------------------------
    # GRU
    # ---------------------------
    if model_name == "gru":
        return GRURegressor(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )

    # ---------------------------
    # Transformer
    # ---------------------------
    if model_name == "transformer":
        return TransformerRegressor(
            input_dim=input_dim,
            d_model=args.tf_d_model,
            nhead=args.tf_heads,
            num_layers=args.tf_layers,
            dim_feedforward=args.tf_ff_dim,
            dropout=args.tf_dropout,
            max_len=512,    # or args.seq_len*2

            use_learnable_pos=bool(args.tf_learnable_pos),
            use_cls_token=bool(args.tf_use_cls_token),
            pool_type=args.tf_pool,
            embed_scale=args.tf_embed_scale,
        )

    raise ValueError(f"Unknown model name: {model_name}")
