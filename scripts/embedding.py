import os
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np

INPUT_FILE = Path("output/merged_dataset.parquet")
OUTPUT_DIR = Path("output/embeddings")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_NAME = "ProsusAI/finbert"     # can be replaced with "FinGPT/fingpt-sentiment"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print(f"Loading model: {MODEL_NAME} on {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def get_sentence_embedding(texts, tokenizer, model):
    """
    lists of news (list[str]) -->  one hourly embedding (mean pooling).
    if no news， then return all zero embedding.
    """
    if len(texts) == 0:
        return np.zeros(model.config.hidden_size, dtype=np.float32)

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # [B, T, H]
        # CLS token embedding per sentence
        cls_embeddings = hidden[:, 0, :]     # [B, H]

    # cluster all news CLS embedding → hourly embedding
    embedding = cls_embeddings.mean(dim=0).cpu().numpy()
    return embedding.astype(np.float32)


def process_symbol(df_symbol, tokenizer, model, symbol):
    """
    Process one symbol (e.g., BTC) and compute embeddings for each hour.
    """
    print(f"\n🔹 Processing {symbol}, rows = {len(df_symbol)}")

    embeddings = []

    for _, row in tqdm(df_symbol.iterrows(), total=len(df_symbol)):
        news_list = row["news_texts"]
        emb = get_sentence_embedding(news_list, tokenizer, model)
        embeddings.append(emb)

    df_symbol["embedding"] = embeddings

    # Save parquet
    out_fp = OUTPUT_DIR / f"{symbol}_embeddings.parquet"

    df_save = df_symbol[["hour", "symbol", "embedding"]].copy()
    df_save.to_parquet(out_fp)

    print(f"Saved {symbol} embeddings → {out_fp}")
    return df_save


def main():
    print("\n=== Generating Embeddings ===")

    # 1. Load merged dataset
    df = pd.read_parquet(INPUT_FILE)
    print(f"✔ Loaded merged dataset: {df.shape}")

    # 2. Load FinBERT / FinGPT
    tokenizer, model = load_model()

    # 3. Symbol list
    symbols = sorted(df["symbol"].unique())
    print(f"Symbols found: {symbols}")

    # 4. Process each symbol independently
    for symbol in symbols:
        out_fp = OUTPUT_DIR / f"{symbol}_embeddings.parquet"
        if out_fp.exists():
            print(f"⏭ Skipping {symbol}, cached file found.")
            continue

        df_symbol = df[df["symbol"] == symbol].copy()
        process_symbol(df_symbol, tokenizer, model, symbol)

    print("\n Embedding pipeline finished!")


if __name__ == "__main__":
    main()

