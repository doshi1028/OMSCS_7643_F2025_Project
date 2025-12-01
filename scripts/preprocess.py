import os
import pandas as pd
import numpy as np
from pathlib import Path
import re

DATA_DIR = Path("data")

MARKET_SOURCE = DATA_DIR / "BTC_USD_hourly.parquet"  # directory or single parquet file
NEWS_FILE = DATA_DIR / "cryptopanic_news.csv"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)
CLEAN_DIR = OUTPUT_DIR / "data"
CLEAN_DIR.mkdir(exist_ok=True)

# TODO: point MARKET_SOURCE to the finalized multi-asset directory once spaCy tagging is ready. -->  spaCy is too complicated?
CRYPTO_KEYWORDS = {
    "BTC": ["bitcoin", "btc", "satoshis", "satoshi"],
    "ETH": ["ethereum", "eth", "ether"],
    "SOL": ["solana", "sol"],
    "ADA": ["cardano", "ada"],
    "LTC": ["litecoin", "ltc"],
    "OP": ["optimism", "op"],
    "XRP": ["xrp", "ripple"],
    "DOGE": ["dogecoin", "doge"],
    "BNB": ["bnb", "binance coin", "binance"],
    "USDT": ["tether", "usdt"],
    "USDC": ["usd coin", "usdc"],
    "DOT": ["polkadot", "dot"],
    "AVAX": ["avalanche", "avax"],
    "MATIC": ["polygon", "matic"],
    "ATOM": ["cosmos", "atom"],
    "LINK": ["chainlink", "link"],
    "XLM": ["stellar", "xlm"],
    "TRX": ["tron", "trx"],
    "SHIB": ["shiba", "shiba inu", "shib"],
}

def detect_related_symbols(text):
    """Return list of crypto symbols mentioned in the news text."""
    text_low = text.lower()
    matched = []

    for symbol, keywords in CRYPTO_KEYWORDS.items():
        for kw in keywords:
            if kw in text_low:
                matched.append(symbol)
                break

    return matched if matched else ["OTHERS"]
    

def normalize_text(text):
    """Stronger cleaning: remove URLs, emojis, HTML tags, excessive spaces."""
    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove Twitter style RT @username:
    text = re.sub(r"\bRT\s+@\w+:\s*", "", text)
    
    # Remove standalone @username (with or without punctuation)
    text = re.sub(r"@\w+", "", text)

    # Remove emojis and other non-text symbols
    text = text.encode("ascii", "ignore").decode()

    # Remove multiple slashes like "4/" or "1/"
    # Optional — depends on whether you want to keep list numbering
    text = re.sub(r"\b\d+\/\b", "", text)

    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def clean_news(df):
    """Clean CryptoPanic news: handle NaN, combine title/description, normalize datetime."""
    print(" Cleaning news data...")

    df["title"] = df["title"].fillna("")
    df["description"] = df["description"].fillna("")
    df["text"] = (df["title"].str.strip() + ". " + df["description"].str.strip()).str.strip()
    df["text"] = df["text"].str.replace("..", ".", regex=False)
    
    # TODO: incorporate stronger text normalization (e.g., remove special characters/emojis)-->complete
    df["text"] = df["text"].apply(normalize_text)
    
    df = df[df["text"].str.len() > 0].copy()

    df["newsDatetime"] = pd.to_datetime(df["newsDatetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["newsDatetime"])
    df["hour"] = df["newsDatetime"].dt.floor("H")
    df["newsTimestamp"] = df["newsDatetime"]
    df["symbol_tags"] = df["text"].apply(detect_related_symbols)
    df["primary_symbol"] = df["symbol_tags"].apply(lambda tags: tags[0] if isinstance(tags, list) and len(tags) > 0 else "BTC")

    print(f"News cleaned: {len(df)} rows remain.")
    return df[["newsTimestamp", "hour", "text", "positive", "negative", "symbol_tags", "primary_symbol"]]


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
