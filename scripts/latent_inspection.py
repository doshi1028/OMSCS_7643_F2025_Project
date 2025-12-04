"""Latent factor inspection using BERTopic, PCA, and optional transformer modeling."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from bertopic import BERTopic
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import r2_score
from umap import UMAP

EMB_DIR = Path("output/embeddings")
MARKET_FILE = Path("output/data/clean_market.parquet")
ANALYSIS_DIR = Path("output/analysis")
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

BITCOIN_STOPWORDS = {
    "bitcoin",
    "btc",
    "satoshi",
    "satoshis",
    "btcusd",
    "btc/usd",
    "btc-usd",
    "crypto",
    "coin",
}


class FullSequenceTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        max_len=5000,
    ):
        super().__init__()
        self.d_model = d_model
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.register_buffer(
            "pos_embedding",
            self._build_pos_encoding(max_len, d_model),
            persistent=False,
        )

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
            num_layers=num_layers,
        )
        self.head = nn.Linear(d_model, 1)

    def _build_pos_encoding(self, length: int, dim: int):
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-np.log(10000.0) / dim)
        )
        pe = torch.zeros(1, length, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def _ensure_positional_length(self, target_len: int):
        if target_len > self.pos_embedding.size(1):
            new_len = max(target_len, int(self.pos_embedding.size(1) * 1.5))
            self.pos_embedding = self._build_pos_encoding(new_len, self.d_model).to(
                self.pos_embedding.device
            )

    def forward(self, x):
        # x: (B, T, K)
        B, T, _ = x.shape
        self._ensure_positional_length(T)
        x = self.input_norm(x)
        x = self.input_proj(x)
        pos = self.pos_embedding[:, :T, :]
        x = x + pos
        h = self.transformer(x)
        return self.head(h).squeeze(-1)


def load_symbol_embeddings(symbol: str) -> pd.DataFrame:
    path = EMB_DIR / f"{symbol}_embeddings.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    df = pd.read_parquet(path)
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x, dtype=np.float32))
    df["hour"] = pd.to_datetime(df["hour"]).dt.tz_localize(None)
    df["newsTimestamp"] = pd.to_datetime(df["newsTimestamp"]).dt.tz_localize(None)
    return df.sort_values("newsTimestamp").reset_index(drop=True)


def run_topic_model(df: pd.DataFrame, n_topics: int, top_words: int, random_state: int):
    texts = df["text"].tolist()
    matrix = np.stack(df["embedding"].values)
    stop_words = list(ENGLISH_STOP_WORDS.union(BITCOIN_STOPWORDS))
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        token_pattern=r"(?u)\b\w\w+\b",
        lowercase=True,
        strip_accents="unicode",
    )
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=random_state,
    )
    topic_model = BERTopic(
        nr_topics=n_topics,
        verbose=True,
        vectorizer_model=vectorizer,
        umap_model=umap_model,
    )
    topics, _ = topic_model.fit_transform(texts, matrix)
    df["topic"] = topics

    topic_info = topic_model.get_topic_info()
    topic_summaries = {}
    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]
        if topic_id == -1:
            continue
        top_terms = [word for word, _ in topic_model.get_topic(topic_id)[:top_words]]
        topic_summaries[topic_id] = top_terms
    return topic_model, topic_summaries


def run_pca(df: pd.DataFrame, n_components: int):
    matrix = np.stack(df["embedding"].values)
    pca = PCA(n_components=n_components, random_state=42)
    scores = pca.fit_transform(matrix)
    return pca, scores


def summarize_pcs(df, scores, topic_labels, top_pcs: int, k: int):
    summaries = []
    for comp_idx in range(min(top_pcs, scores.shape[1])):
        component_scores = scores[:, comp_idx]
        order = np.argsort(component_scores)
        top_idx = order[-k:][::-1]
        bottom_idx = order[:k]
        summaries.append(
            {
                "component": comp_idx + 1,
                "top_positive": [
                    {
                        "headline": df.iloc[i]["text"],
                        "timestamp": df.iloc[i]["newsTimestamp"].isoformat(),
                        "topic": int(topic_labels[i]),
                        "score": float(component_scores[i]),
                    }
                    for i in top_idx
                ],
                "top_negative": [
                    {
                        "headline": df.iloc[i]["text"],
                        "timestamp": df.iloc[i]["newsTimestamp"].isoformat(),
                        "topic": int(topic_labels[i]),
                        "score": float(component_scores[i]),
                    }
                    for i in bottom_idx
                ],
            }
        )
    return summaries


def compute_returns(df: pd.DataFrame, horizon: int):
    df = df.sort_values("hour").copy()
    df["future_close"] = df["close"].shift(-horizon)
    df["return"] = (df["future_close"] - df["close"]) / df["close"]
    df = df.iloc[:-horizon]
    return df[["hour", "return"]]


def train_transformer_full_sequence(
    symbol,
    scores,
    df,
    horizon,
    d_model,
    nhead,
    num_layers,
    lr,
    epochs,
):
    pca_cols = [f"PC{i+1}" for i in range(scores.shape[1])]
    pca_df = pd.DataFrame(scores, columns=pca_cols)
    pca_df["hour"] = df["hour"].values
    hourly = pca_df.groupby("hour").mean().reset_index()

    market = pd.read_parquet(MARKET_FILE)
    market["hour"] = pd.to_datetime(market["hour"]).dt.tz_localize(None)
    df_symbol = market[market["symbol"] == symbol].copy()
    returns = compute_returns(df_symbol, horizon=horizon)

    merged = hourly.merge(returns, on="hour", how="inner")
    if merged.empty:
        raise ValueError("No overlapping hours between PCA features and returns.")

    X = merged[pca_cols].values
    y = merged["return"].values

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FullSequenceTransformer(
        input_dim=X.shape[1],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
    ).to(device)

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device).unsqueeze(0)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_tensor).squeeze(0)
        loss = criterion(preds, y_tensor)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        preds = model(X_tensor).squeeze(0)
        mse = criterion(preds, y_tensor).item()
        rmse = float(np.sqrt(mse))
        mae = float(torch.mean(torch.abs(preds - y_tensor)).item())
        r2 = float(
            r2_score(
                y_tensor.cpu().numpy(),
                preds.cpu().numpy(),
            )
        )

    # Gradient-based importance
    model.zero_grad()
    X_var = X_tensor.clone().detach().requires_grad_(True)
    preds_for_grad = model(X_var).squeeze(0)
    grad_loss = criterion(preds_for_grad, y_tensor)
    grad_loss.backward()
    grad_importance = (
        X_var.grad.detach().abs().mean(dim=1).squeeze(0).cpu().numpy()
    )
    pc_importance = {
        pc: float(val) for pc, val in zip(pca_cols, grad_importance)
    }

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "importance": pc_importance,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect latent factors with BERTopic and PCA.")
    parser.add_argument("--symbol", type=str, default="BTC")
    parser.add_argument("--topics", type=int, default=30)
    parser.add_argument("--topic-words", type=int, default=10)
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--pc-top-k", type=int, default=5)
    parser.add_argument("--run-transformer", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--transformer-lr", type=float, default=1e-4)
    parser.add_argument("--transformer-epochs", type=int, default=50)
    parser.add_argument("--transformer-d-model", type=int, default=128)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    args = parser.parse_args()

    df = load_symbol_embeddings(args.symbol)
    print(f"Loaded {len(df)} rows for {args.symbol}")

    topic_model, topic_summary = run_topic_model(
        df,
        n_topics=args.topics,
        top_words=args.topic_words,
        random_state=args.random_state,
    )
    pca, scores = run_pca(df, args.pca_components)

    pc_summary = summarize_pcs(df, scores, df["topic"].values, args.pca_components, args.pc_top_k)

    result = {
        "symbol": args.symbol,
        "topics": topic_summary,
        "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "pc_summary": pc_summary,
    }

    if args.run_transformer:
        tf_result = train_transformer_full_sequence(
            args.symbol,
            scores,
            df,
            horizon=args.horizon,
            d_model=args.transformer_d_model,
            nhead=args.transformer_heads,
            num_layers=args.transformer_layers,
            lr=args.transformer_lr,
            epochs=args.transformer_epochs,
        )
        result["transformer_analysis"] = tf_result

    out_path = ANALYSIS_DIR / f"latent_summary_{args.symbol}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved summary → {out_path}")

    topic_model.save(ANALYSIS_DIR / f"bertopic_{args.symbol}")


if __name__ == "__main__":
    main()
