import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np

INPUT_FILE = Path("output/data/clean_news.parquet")
MARKET_FILE = Path("output/data/clean_market.parquet")
OUTPUT_DIR = Path("output/embeddings")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "ProsusAI/finbert"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print(f"Loading model: {MODEL_NAME} on {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def get_sentence_embedding(text, tokenizer, model):
    normalized = (text or "").strip()
    if not normalized:
        return np.zeros(model.config.hidden_size, dtype=np.float32)

    inputs = tokenizer(
        normalized,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # [1, T, H]
        cls_embeddings = hidden[:, 0, :]  # [1, H]

    embedding = cls_embeddings.squeeze(0).cpu().numpy()
    return embedding.astype(np.float32)


def explode_news(news_df: pd.DataFrame) -> pd.DataFrame:
    df = news_df.explode("symbol_tags").rename(columns={"symbol_tags": "symbol"})
    df = df.dropna(subset=["symbol"])
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df


def main():
    print("\n=== Generating row-level embeddings ===")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Clean news parquet not found: {INPUT_FILE}")
    if not MARKET_FILE.exists():
        raise FileNotFoundError(f"Clean market parquet not found: {MARKET_FILE}")

    news = pd.read_parquet(INPUT_FILE)
    market = pd.read_parquet(MARKET_FILE)
    market_symbols = sorted(market["symbol"].unique())

    news_expanded = explode_news(news)
    news_expanded = news_expanded[news_expanded["symbol"].isin(market_symbols)]
    print(f"Found {len(market_symbols)} symbols with market data.")

    tokenizer, model = load_model()

    for symbol in market_symbols:
        symbol_rows = news_expanded[news_expanded["symbol"] == symbol].copy()
        if symbol_rows.empty:
            print(f"Skipping {symbol}: no tagged news.")
            continue

        symbol_rows = symbol_rows.sort_values("newsTimestamp").reset_index(drop=True)
        embeddings = []
        for _, row in tqdm(
            symbol_rows.iterrows(), total=len(symbol_rows), desc=f"{symbol} embeddings"
        ):
            emb = get_sentence_embedding(row["text"], tokenizer, model)
            embeddings.append(emb.tolist())

        out_df = symbol_rows[
            ["newsTimestamp", "hour", "text", "positive", "negative", "symbol"]
        ].copy()
        out_df["embedding"] = embeddings

        out_path = OUTPUT_DIR / f"{symbol}_embeddings.parquet"
        out_df.to_parquet(out_path)
        print(f"Saved {symbol} embeddings → {out_path}")

    print("\nEmbedding pipeline finished!")


if __name__ == "__main__":
    main()
