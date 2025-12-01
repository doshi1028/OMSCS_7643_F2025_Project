import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
import torch.nn.functional as F

INPUT_FILE = Path("output/data/clean_news.parquet")
MARKET_FILE = Path("output/data/clean_market.parquet")
OUTPUT_DIR = Path("output/embeddings")
OUTPUT_DIR.mkdir(exist_ok=True)

TOKENIZER_NAME = "ProsusAI/finbert"
EMB_MODEL_NAME = "ProsusAI/finbert"
SENT_MODEL_NAME = "ProsusAI/finbert-sentiment"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    print(f"Loading models: {EMB_MODEL_NAME} and {SENT_MODEL_NAME} on {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    # Embedding model
    model_emb = AutoModel.from_pretrained(EMB_MODEL_NAME).to(DEVICE)
    model_emb.eval()

    # Sentiment model
    model_sent = AutoModelForSequenceClassification.from_pretrained(SENT_MODEL_NAME).to(DEVICE)
    model_sent.eval()

    return tokenizer, model_emb, model_sent


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

def get_sentiment_scores(text, tokenizer, model_sent):
    if not text.strip():
        return np.array([0.33, 0.33, 0.33], dtype=np.float32), 0.0
    
    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        logits = model_sent(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

    neg, neu, pos = probs
    bullish_score = pos - neg

    return probs.astype(np.float32), float(bullish_score)


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

    #tokenizer, model = load_model()
    tokenizer, model_emb, model_sent = load_model()

    for symbol in market_symbols:
        symbol_rows = news_expanded[news_expanded["symbol"] == symbol].copy()
        if symbol_rows.empty:
            print(f"Skipping {symbol}: no tagged news.")
            continue

        symbol_rows = symbol_rows.sort_values("newsTimestamp").reset_index(drop=True)
        embeddings = []
        sent_probs_list = []
        bullish_list = []
        for _, row in tqdm(
            symbol_rows.iterrows(), total=len(symbol_rows), desc=f"{symbol} embeddings and sentiment"
        ):
            text = row["text"]
            emb = get_sentence_embedding(text, tokenizer, model_emb)
            sent_probs, bullish = get_sentiment_scores(text, tokenizer, model_sent)

            embeddings.append(emb.tolist())
            sent_probs_list.append(sent_probs.tolist())
            bullish_list.append(bullish)

        out_df = symbol_rows[
            ["newsTimestamp", "hour", "text", "positive", "negative", "symbol"]
        ].copy()
        out_df["embedding"] = embeddings
        out_df["embedding"] = embeddings            # 768D
        out_df["sentiment_probs"] = sent_probs_list # 3D
        out_df["bullish_score"] = bullish_list      # 1D

        out_path = OUTPUT_DIR / f"{symbol}_embeddings.parquet"
        out_df.to_parquet(out_path)
        print(f"Saved {symbol} embeddings → {out_path}")

        # parquet structure
    #| hour  | text | embedding | sentiment_probs | bullish_score | positive | negative | symbol |
#| ----- | ---- | --------- | --------------- | ------------- | -------- | -------- | ------ |
#| 01:00 | ...  | [768]     | [neg, neu, pos] | pos-neg       | 0        | 1        | BTC    |


    print("\nEmbedding pipeline finished!")


if __name__ == "__main__":
    main()
