import numpy as np
import pandas as pd
from pathlib import Path

MARKET_FILE = Path("output/clean_market.parquet")
EMB_FILE = Path("output/embeddings/hourly_embeddings.parquet")
FEATURE_DIR = Path("output/features")
FEATURE_DIR.mkdir(exist_ok=True)

EMBED_DIM = 768
LOOKBACK_HOURS = 12


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


def _average_lookback_windows(vectors: np.ndarray, lookback: int) -> np.ndarray:
    """
    Turn [T, D] vectors into lookback-averaged features of shape [T - lookback + 1, D].
    """
    if lookback <= 1:
        return vectors

    if len(vectors) < lookback:
        return np.empty((0, vectors.shape[1]), dtype=vectors.dtype)

    windows = []
    for idx in range(lookback, len(vectors) + 1):
        window = vectors[idx - lookback : idx]
        windows.append(window.mean(axis=0))

    return np.stack(windows, axis=0)


def _align_targets(y: np.ndarray, lookback: int) -> np.ndarray:
    if lookback <= 1:
        return y
    if len(y) < lookback:
        return np.empty(0, dtype=y.dtype)
    return y[lookback - 1 :]


def build_features_for_symbol(
    df_symbol: pd.DataFrame,
    hourly_emb: pd.DataFrame,
    lookback: int = LOOKBACK_HOURS,
):
    """
    Merge embeddings with market data and compute features + targets.

    TODO: Once the modeling finalizes the architectures (e.g., sequence
    encoder vs. flattened MLP), expose hooks to return either reshaped sequences
    with shape [N, lookback, D] or custom aggregations. For now we average
    embeddings across the lookback window to produce one vector per hour.
    """
    df = df_symbol.merge(hourly_emb, on="hour", how="left")

    df["embedding"] = df["embedding"].apply(
        lambda x: np.array(x, dtype=np.float32)
        if isinstance(x, (list, np.ndarray))
        else np.zeros(EMBED_DIM, dtype=np.float32)
    )

    # Compute next-hour returns
    df = compute_returns(df)

    embeddings = np.stack(df["embedding"].values)
    y = df["return"].values.astype(np.float32)

    X = _average_lookback_windows(embeddings, lookback)
    y_aligned = _align_targets(y, lookback)

    df_trimmed = df.iloc[lookback - 1 :].reset_index(drop=True) if lookback > 1 else df

    return df_trimmed, X, y_aligned


def main():
    print("\n=== Building ML features ===")

    if not MARKET_FILE.exists():
        raise FileNotFoundError(f"Clean market parquet not found: {MARKET_FILE}")
    if not EMB_FILE.exists():
        raise FileNotFoundError(f"Hourly embeddings not found: {EMB_FILE}")

    market = pd.read_parquet(MARKET_FILE)
    hourly_emb = pd.read_parquet(EMB_FILE)

    symbols = sorted(market["symbol"].unique())
    print(f"✔ Found symbols: {symbols}")

    all_X, all_y = [], []
    records = []

    for symbol in symbols:
        print(f"\n--- Processing {symbol} ---")

        df_symbol = market[market["symbol"] == symbol].copy()

        df_feat, X, y = build_features_for_symbol(
            df_symbol,
            hourly_emb,
            lookback=LOOKBACK_HOURS,
        )

        if len(X) == 0 or len(y) == 0:
            print(f"   {symbol}: insufficient history for lookback={LOOKBACK_HOURS}, skipping.")
            continue

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
