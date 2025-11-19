import os
import numpy as np
import pandas as pd
from pathlib import Path

MERGED_FILE = Path("output/merged_dataset.parquet")
EMB_DIR = Path("output/embeddings")
FEATURE_DIR = Path("output/features")
FEATURE_DIR.mkdir(exist_ok=True)

EMBED_DIM = 768   # FinBERT / FinGPT default factor (for bert-base)


def load_embeddings_for_symbol(symbol):
    """
    Load embedding parquet for a symbol.
    """
    fp = EMB_DIR / f"{symbol}_embeddings.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Embedding file not found: {fp}")

    df = pd.read_parquet(fp)
    return df[["hour", "symbol", "embedding"]]


def compute_returns(df):
    """
    Compute next-hour return:
      return_t = (close_{t+1} - close_t) / close_t
    """
    df = df.sort_values("hour").copy()
    df["close_next"] = df["close"].shift(-1)
    df["return"] = (df["close_next"] - df["close"]) / df["close"]

    # Remove last row (no next-hour close)
    df = df[:-1]
    return df


def build_features_for_symbol(df_merged, df_emb):
    """
    Merge embeddings with market data and compute features + targets.
    """
    df = df_merged.merge(df_emb, on=["hour", "symbol"], how="left")

    # Embedding: ensure ndarray
    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(x, dtype=np.float32)
                  if isinstance(x, (list, np.ndarray))
                  else np.zeros(EMBED_DIM, dtype=np.float32)
    )

    # Compute next-hour returns
    df = compute_returns(df)

    # Features X: currently embedding only (add more features like SMA, EMA, RSI...)
    X = np.stack(df["embedding"].values)      # shape: [N, 768]
    y = df["return"].values.astype(np.float32)

    return df, X, y


def main():
    print("\n=== Building ML features ===")

    # Load merged dataset
    merged = pd.read_parquet(MERGED_FILE)
    print(f"✔ Loaded merged dataset: {merged.shape}")

    symbols = sorted(merged["symbol"].unique())
    print(f"✔ Found symbols: {symbols}")

    all_X, all_y = [], []
    records = []

    for symbol in symbols:
        print(f"\n--- Processing {symbol} ---")

        df_symbol = merged[merged["symbol"] == symbol].copy()
        df_emb = load_embeddings_for_symbol(symbol)

        df_feat, X, y = build_features_for_symbol(df_symbol, df_emb)

        # Collect
        all_X.append(X)
        all_y.append(y)
        records.append(df_feat)

        print(f"   {symbol}: X shape = {X.shape}, y shape = {y.shape}")

    # Concatenate all symbols
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    df_all = pd.concat(records, ignore_index=True)

    print("\n=== Final Dataset Shapes ===")
    print(f"X_all: {X_all.shape}")
    print(f"y_all: {y_all.shape}")
    print(f"records: {df_all.shape}")

    # Save outputs
    np.save(FEATURE_DIR / "X.npy", X_all)
    np.save(FEATURE_DIR / "y.npy", y_all)
    df_all.to_parquet(FEATURE_DIR / "dataset.parquet")

    print("\n Saved:")
    print(" - features/X.npy")
    print(" - features/y.npy")
    print(" - features/dataset.parquet")
    print("\n Feature generation completed!")


if __name__ == "__main__":
    main()
