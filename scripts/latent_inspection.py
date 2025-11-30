"""Latent factor inspection using BERTopic, PCA, and optional transformer modeling."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from torch.utils.data import Dataset, DataLoader

from model import get_model

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


class PCADataset(Dataset):
    def __init__(self, X, y, seq_len=1):
        self.seq_len = seq_len
        if seq_len > 1:
            self.X, self.y = self._build_sequences(X, y, seq_len)
        else:
            self.X, self.y = X, y

    @staticmethod
    def _build_sequences(X, y, seq_len):
        feats, targets = [], []
        for i in range(seq_len - 1, len(X)):
            feats.append(X[i - seq_len + 1 : i + 1])
            targets.append(y[i])
        return np.array(feats), np.array(targets)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


def load_symbol_embeddings(symbol: str) -> pd.DataFrame:
    path = EMB_DIR / f"{symbol}_embeddings.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    df = pd.read_parquet(path)
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x, dtype=np.float32))
    df["hour"] = pd.to_datetime(df["hour"]).dt.tz_localize(None)
    df["newsTimestamp"] = pd.to_datetime(df["newsTimestamp"]).dt.tz_localize(None)
    return df.sort_values("newsTimestamp").reset_index(drop=True)


def run_topic_model(df: pd.DataFrame, n_topics: int, top_words: int):
    texts = df["text"].tolist()
    matrix = np.stack(df["embedding"].values)
    stop_words = list(ENGLISH_STOP_WORDS.union(BITCOIN_STOPWORDS))
    vectorizer = CountVectorizer(
        stop_words=stop_words,
        token_pattern=r"(?u)\b\w\w+\b",
        lowercase=True,
        strip_accents="unicode",
    )
    topic_model = BERTopic(nr_topics=n_topics, verbose=True, vectorizer_model=vectorizer)
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


def train_transformer_on_pca(symbol, scores, df, seq_len, horizon, lr, epochs):
    print("\n=== Transformer on PCA Features ===")
    pca_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
    pca_df["hour"] = df["hour"].values
    hourly = pca_df.groupby("hour").mean().reset_index()

    market = pd.read_parquet(MARKET_FILE)
    market["hour"] = pd.to_datetime(market["hour"]).dt.tz_localize(None)
    df_symbol = market[market["symbol"] == symbol].copy()
    returns = compute_returns(df_symbol, horizon=horizon)

    merged = hourly.merge(returns, on="hour", how="inner")
    if merged.empty:
        raise ValueError("No overlapping hours between PCA features and returns.")

    X = merged[[col for col in merged.columns if col.startswith("PC")]].values
    y = merged["return"].values

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_ds = PCADataset(X_train, y_train, seq_len=seq_len)
    val_ds = PCADataset(X_val, y_val, seq_len=seq_len)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)

    args = argparse.Namespace(
        tf_d_model=128,
        tf_heads=4,
        tf_layers=2,
        tf_ff_dim=256,
        tf_dropout=0.1,
        tf_pool="attention",
        tf_learnable_pos=1,
        tf_use_cls_token=1,
        tf_embed_scale=1.0,
        hidden_dim=256,
        dropout=0.1,
        lstm_proj_dim=128,
        lstm_use_layernorm=1,
        lstm_use_attention=1,
        num_layers=2,
    )

    model = get_model("transformer", input_dim=X.shape[1], args=args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * xb.size(0)
        train_loss = total / len(train_loader.dataset)

        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                total += loss.item() * xb.size(0)
        val_loss = total / len(val_loader.dataset)
        best_val = min(best_val, val_loss)
        print(f"Epoch {epoch+1}: train {train_loss:.6f}, val {val_loss:.6f}")

    return best_val


def main():
    parser = argparse.ArgumentParser(description="Inspect latent factors with BERTopic and PCA.")
    parser.add_argument("--symbol", type=str, default="BTC")
    parser.add_argument("--topics", type=int, default=30)
    parser.add_argument("--topic-words", type=int, default=10)
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--pc-top-k", type=int, default=5)
    parser.add_argument("--run-transformer", action="store_true")
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    df = load_symbol_embeddings(args.symbol)
    print(f"Loaded {len(df)} rows for {args.symbol}")

    topic_model, topic_summary = run_topic_model(
        df, n_topics=args.topics, top_words=args.topic_words
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
        best_val = train_transformer_on_pca(
            args.symbol,
            scores,
            df,
            seq_len=args.seq_len,
            horizon=args.horizon,
            lr=args.lr,
            epochs=args.epochs,
        )
        result["transformer_val_loss"] = best_val

    out_path = ANALYSIS_DIR / f"latent_summary_{args.symbol}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved summary → {out_path}")

    topic_model.save(ANALYSIS_DIR / f"bertopic_{args.symbol}")


if __name__ == "__main__":
    main()
BITCOIN_STOPWORDS = {
    "bitcoin",
    "btc",
    "satoshi",
    "satoshis",
    "btc/usd",
    "btc-usd",
    "btcusd",
}
