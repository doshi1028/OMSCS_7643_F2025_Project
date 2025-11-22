import os
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
# TODO: point MARKET_SOURCE to the finalized multi-asset directory once spaCy tagging is ready.
MARKET_SOURCE = DATA_DIR / "BTC_USD_hourly.parquet"  # directory or single parquet file
NEWS_FILE = DATA_DIR / "cryptopanic_news.csv"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
CLEAN_DIR = OUTPUT_DIR / "data"
CLEAN_DIR.mkdir(exist_ok=True)

def clean_news(df):
    """Clean CryptoPanic news: handle NaN, combine title/description, normalize datetime."""
    print("🔹 Cleaning news data...")
    # TODO: incorporate stronger text normalization (e.g., remove special characters/emojis).

    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["text"] = (df["title"].str.strip() + ". " + df["description"].str.strip()).str.strip()
    df["text"] = df["text"].str.replace("..", ".", regex=False)
    df = df[df["text"].str.len() > 0].copy()

    df["newsDatetime"] = pd.to_datetime(df["newsDatetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["newsDatetime"])
    df["hour"] = df["newsDatetime"].dt.floor("H")

    print(f"✔ News cleaned: {len(df)} rows remain.")
    return df[["hour", "text", "positive", "negative"]]


def _iter_market_files():
    """Yield (symbol, path) pairs from either a directory or a single parquet file."""
    if MARKET_SOURCE.is_dir():
        for fname in os.listdir(MARKET_SOURCE):
            if fname.endswith(".parquet"):
                symbol = fname.split("-")[0]
                yield symbol, MARKET_SOURCE / fname
    elif MARKET_SOURCE.suffix == ".parquet":
        symbol = MARKET_SOURCE.stem.split("_")[0].upper()
        yield symbol, MARKET_SOURCE
    else:
        raise FileNotFoundError(f"Unsupported MARKET_SOURCE: {MARKET_SOURCE}")


def load_and_clean_market():
    """Load parquet market data (single file or directory) and clean."""
    print("Loading market data files...")

    market_dfs = []

    for symbol, fpath in _iter_market_files():
        df = pd.read_parquet(fpath)
        # Normalize column names
        ts_col = "timestamp" if "timestamp" in df.columns else "datetime"
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        df = df.dropna(subset=[ts_col])

        df["hour"] = df[ts_col].dt.floor("H")
        df["symbol"] = symbol

        market_dfs.append(df)

    if not market_dfs:
        raise ValueError(f"No market parquet files found in {MARKET_SOURCE}")

    market = pd.concat(market_dfs, ignore_index=True)

    print(f"Loaded {len(market_dfs)} market files, total rows = {len(market)}")
    return market


def align_news_to_market(
    market: pd.DataFrame,
    news: pd.DataFrame,
    fallback_symbol: str = "BTC",
) -> pd.DataFrame:
    """
    Temporary alignment that assumes *all* news corresponds to the fallback symbol.

    TODO: Replace this with spaCy-powered multi-asset tagging as described earlier.
    Once tagging exists, the function should explode the news dataframe into
    (symbol, hour) pairs and aggregate sentiment per asset before merging.
    """

    btc_market = market[market["symbol"] == fallback_symbol].copy()
    if btc_market.empty:
        raise ValueError(
            f"No market rows found for fallback symbol '{fallback_symbol}'. "
            "Update the assumption or preprocess real tags."
        )

    agg_news = (
        news.groupby("hour")
        .agg(
            {"text": list, "positive": "sum", "negative": "sum"}
        )
        .rename(
            columns={
                "text": "news_texts",
                "positive": "pos_count",
                "negative": "neg_count",
            }
        )
        .reset_index()
    )

    agg_news["news_texts"] = agg_news["news_texts"].apply(
        lambda texts: " ".join(texts) if isinstance(texts, list) else ""
    )

    merged = btc_market.merge(agg_news, on="hour", how="left")
    merged["news_texts"] = merged["news_texts"].fillna("")
    merged["pos_count"] = merged["pos_count"].fillna(0).astype(int)
    merged["neg_count"] = merged["neg_count"].fillna(0).astype(int)

    return merged


def main():
    print("\n=== Running Preprocess Pipeline ===")

    news_raw = pd.read_csv(NEWS_FILE)
    news_clean = clean_news(news_raw)
    news_clean.to_parquet(CLEAN_DIR / "clean_news.parquet")
    print("Saved clean_news.parquet\n")

    market = load_and_clean_market()
    market.to_parquet(CLEAN_DIR / "clean_market.parquet")
    print("Saved clean_market.parquet\n")

    print("\n Preprocess pipeline complete!")


if __name__ == "__main__":
    main()
