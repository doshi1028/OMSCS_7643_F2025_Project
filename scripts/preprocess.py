import os
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
MARKET_DIR = DATA_DIR / "crypto_data_hourly"
NEWS_FILE = DATA_DIR / "cryptopanic_news.csv"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

def clean_news(df):
    """Clean CryptoPanic news: handle NaN, combine title/description, normalize datetime."""
    print("🔹 Cleaning news data...")

    # Normalize text columns
    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")

    # Merge title + description
    df["text"] = (df["title"].str.strip() + ". " + df["description"].str.strip()).str.strip()
    df["text"] = df["text"].str.replace("..", ".", regex=False)

    # Drop rows with no usable text
    df = df[df["text"].str.len() > 0].copy()

    # Standardize datetime
    df["newsDatetime"] = pd.to_datetime(df["newsDatetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["newsDatetime"])

    # Hour-level timestamp for alignment
    df["hour"] = df["newsDatetime"].dt.floor("H")

    print(f"✔ News cleaned: {len(df)} rows remain.")
    return df[["hour", "text", "positive", "negative"]]


def load_and_clean_market():
    """Load all parquet files in crypto_data_hourly and clean."""
    print("Loading market data files...")

    market_dfs = []

    for fname in os.listdir(MARKET_DIR):
        if fname.endswith(".parquet"):
            symbol = fname.split("-")[0]  # e.g., BTC-USD_hourly.parquet → BTC
            fpath = MARKET_DIR / fname

            df = pd.read_parquet(fpath)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            df["hour"] = df["timestamp"].dt.floor("H")
            df["symbol"] = symbol

            market_dfs.append(df)

    market = pd.concat(market_dfs, ignore_index=True)

    print(f"Loaded {len(market_dfs)} market files, total rows = {len(market)}")
    return market


def align_news_to_market(market, news):
    """For each symbol-hour, aggregate all news texts."""
    print("🔹 Aligning news with market data (hourly)...")

    # Group news texts per hour
    agg_news = (
        news.groupby("hour")
            .agg({
                "text": list,
                "positive": "sum",
                "negative": "sum"
            })
            .rename(columns={"text": "news_texts",
                             "positive": "pos_count",
                             "negative": "neg_count"})
            .reset_index()
    )

    print(f"Aggregated news to {len(agg_news)} hourly buckets.")

    # Merge per symbol (outer merge keeps all market hours)
    merged = market.merge(agg_news, on="hour", how="left")

    # Where no news: fill empty list
    merged["news_texts"] = merged["news_texts"].apply(lambda x: x if isinstance(x, list) else [])
    merged["pos_count"] = merged["pos_count"].fillna(0).astype(int)
    merged["neg_count"] = merged["neg_count"].fillna(0).astype(int)

    print(f"Final merged shape: {merged.shape}")
    return merged


def main():
    print("\n=== Running Preprocess Pipeline ===")

    # 1. Clean News
    news_raw = pd.read_csv(NEWS_FILE)
    news_clean = clean_news(news_raw)
    news_clean.to_parquet(OUTPUT_DIR / "clean_news.parquet")
    print("Saved clean_news.parquet\n")

    # 2. Load + Clean Market Data
    market = load_and_clean_market()
    market.to_parquet(OUTPUT_DIR / "clean_market.parquet")
    print("Saved clean_market.parquet\n")

    # 3. Align News to Each Market Symbol/Hour
    merged = align_news_to_market(market, news_clean)
    merged.to_parquet(OUTPUT_DIR / "merged_dataset.parquet")
    print("Saved merged_dataset.parquet")

    print("\n Preprocess pipeline complete!")


if __name__ == "__main__":
    main()
