import numpy as np
import pandas as pd
from pathlib import Path
import argparse 

MARKET_FILE = Path("output/data/clean_market.parquet")
EMB_DIR = Path("output/embeddings")
FEATURE_DIR = Path("output/features")
FEATURE_DIR.mkdir(exist_ok=True)

EMBED_DIM = 768
LOOKBACK_HOURS = 12


def load_embeddings_for_symbol(symbol: str) -> pd.DataFrame:
    emb_path = EMB_DIR / f"{symbol}_embeddings.parquet"
    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding file not found for {symbol}: {emb_path}")

    df = pd.read_parquet(emb_path)

    df["embedding"] = df["embedding"].apply(lambda val: np.array(val, dtype=np.float32))
    df["pos_count"] = df["positive"]
    df["neg_count"] = df["negative"]

    df["hour"] = pd.to_datetime(df["hour"]).dt.tz_localize(None)
    df["newsTimestamp"] = pd.to_datetime(df["newsTimestamp"]).dt.tz_localize(None)

    return df[["hour", "embedding", "pos_count", "neg_count", "newsTimestamp"]]



def compute_returns(df, horizon):
    """
    Compute next-hour return:
      return_t = (close_{t+h} - close_t) / close_t
    """
    df = df.sort_values("hour").copy()
    df[f"close_t+{horizon}"] = df["close"].shift(-horizon)
    df["return"] = (df[f"close_t+{horizon}"] - df["close"]) / df["close"]

    # Remove unavailable rows(no next-h-hours' close)
    df = df[: -horizon]
    return df


def _average_lookback_windows(
        vectors, hours, lookback,
        mode="mean",
        hour_volume_map=None,
        decay_alpha=0.8,
):
    """
    Supported modes:
        - mean        : simple average
        - volume      : volume-weighted average
        - exp_decay   : exponential decay weighting
        - max         : max pooling across lookback window
    """

    unique_hours = np.unique(hours)
    if len(unique_hours) < lookback:
        return np.empty((0, vectors.shape[1]), dtype=vectors.dtype)

    # ====== Step 1: group vectors by hour ======
    hour_to_vectors = {}
    for vec, hr in zip(vectors, hours):
        hour_to_vectors.setdefault(hr, []).append(vec)

    ordered_hours = sorted(hour_to_vectors.keys())
    hour_embeddings = np.array([np.mean(hour_to_vectors[h], axis=0)
                                for h in ordered_hours])

    windows = []
    output_hours = []

    for idx in range(lookback, len(hour_embeddings) + 1):
        h_emb = hour_embeddings[idx - lookback: idx]
        hrs = ordered_hours[idx - lookback: idx]

        # ------------ mean ------------
        if mode == "mean":
            pooled = h_emb.mean(axis=0)

        # ------------ max pooling ------------
        elif mode == "max":
            pooled = h_emb.max(axis=0)

        # ------------ volume weighted ------------
        elif mode == "volume":
            vols = np.array([hour_volume_map.get(pd.Timestamp(h), 1.0) for h in hrs], dtype=np.float32)

            # replace NaN or NEGATIVE volume
            vols = np.nan_to_num(vols, nan=1.0)
            vols[vols < 0] = 1.0

            if vols.sum() == 0:
                vols = np.ones_like(vols)

            vols = vols / vols.sum()
            pooled = (vols[:, None] * h_emb).sum(axis=0)


        # ------------ exponential decay ------------
        elif mode == "exp_decay":
            # weights: recent hour weight highest
            w = decay_alpha ** np.arange(lookback)[::-1]     # e.g. lookback=4 → [α^3, α^2, α^1, α^0]
            w = w.astype(np.float32)
            w = w / w.sum()
            pooled = (w[:, None] * h_emb).sum(axis=0)

        # ------------ attention pooling ------------
        elif mode == "attn":
            # Attention = softmax(norm(x_t))
            score = np.linalg.norm(h_emb, axis=1)
            score = score - score.max()
            score = np.exp(np.clip(score, -50, 50))
            score = score / score.sum()

            pooled = (score[:, None] * h_emb).sum(axis=0)

        else:
            raise ValueError(f"Unknown lookback_mode={mode}")

        windows.append(pooled)
        output_hours.append(hrs[-1])

    return np.array(windows), np.array(output_hours)


def _align_targets(y: np.ndarray, lookback: int) -> np.ndarray:
    if lookback <= 1:
        return y
    if len(y) < lookback:
        return np.empty(0, dtype=y.dtype)
    return y[lookback - 1 :]


def build_features_for_symbol(
    df_symbol,
    news_rows,
    horizon,
    lookback,
    perf_foresight,
    lookback_mode="mean",
    hour_volume_map=None,
):

    """
    Build embedding features from row-level news, then align with returns.
    """
    df = news_rows.sort_values("newsTimestamp").reset_index(drop=True).copy()
  
    if lookback_mode is None:
      lookback_mode = "mean"
  
    if perf_foresight:
        df = df[(df["pos_count"] > 0) | (df["neg_count"] > 0)].copy()
        if df.empty:
            return df, np.empty((0, EMBED_DIM)), np.empty(0)
        print(f"perf_foresight={perf_foresight}: kept {len(df)} rows with positive/negative signals")

    embeddings = np.stack(df["embedding"].values)
    hours = df["hour"].values
    timestamps = df["newsTimestamp"].values

    X, lookback_hours = _average_lookback_windows(
        embeddings,
        hours,
        lookback,
        mode=lookback_mode,
        hour_volume_map=hour_volume_map
    )
    aligned_ts = []
    for hr in lookback_hours:
        idx = np.where(hours == hr)[0]
        aligned_ts.append(timestamps[idx[-1]])
    aligned_ts = np.array(aligned_ts)

    price_df = df_symbol.sort_values("hour").copy()
    price_df = compute_returns(price_df, horizon)
    price_df["hour"] = pd.to_datetime(price_df["hour"]).dt.tz_localize(None)
    price_df = price_df[["hour", "return"]]

    merged = pd.DataFrame({"hour": lookback_hours, "newsTimestamp": aligned_ts})
    merged = merged.merge(price_df, on="hour", how="inner")

    if merged.empty:
        return merged, np.empty((0, EMBED_DIM)), np.empty(0)

    y = merged["return"].values.astype(np.float32)

    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]
    merged = merged.iloc[:min_len].reset_index(drop=True)

    return merged, X, y


def main():
    print("\n=== Building ML features ===")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="How many hours ahead to predict the return (e.g. --horizon 3)"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=LOOKBACK_HOURS,
        help="Number of past hours to average embeddings over (default 12)."
    )
    parser.add_argument(
        "--perfect-foresight",
        action="store_true",
        help="If set, keep only rows with non-zero sentiment (pos/neg).",
    )
    parser.add_argument(
    "--lookback-mode",
    type=str,
    default="mean",
    choices=["mean", "volume", "exp_decay", "max", "attn"]
)

 
    args = parser.parse_args()
    horizon = args.horizon
    lookback = args.lookback
    perfect_foresight = args.perfect_foresight
    lookback_mode = args.lookback_mode

    if not MARKET_FILE.exists():
        raise FileNotFoundError(f"Clean market parquet not found: {MARKET_FILE}")

    market = pd.read_parquet(MARKET_FILE)
    market["hour"] = pd.to_datetime(market["hour"]).dt.tz_localize(None)

    symbols = sorted(market["symbol"].unique())
    print(f"✔ Found symbols: {symbols}")

    all_X, all_y = [], []
    records = []

    for symbol in symbols:
        print(f"\n--- Processing {symbol} ---")


        df_symbol = market[market["symbol"] == symbol].copy()
        hour_volume_map = (
              df_symbol[["hour", "volume"]]
              .set_index("hour")["volume"]
              .to_dict()
          )
        try:
            news_rows = load_embeddings_for_symbol(symbol)
        except FileNotFoundError:
            print(f"   {symbol}: no embedding file found, skipping.")
            continue

        df_feat, X, y = build_features_for_symbol(
            df_symbol,
            news_rows,
            horizon,
            lookback=lookback,
            perf_foresight=perfect_foresight,
            lookback_mode=lookback_mode,
            hour_volume_map=hour_volume_map
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
