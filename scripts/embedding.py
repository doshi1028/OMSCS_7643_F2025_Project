import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np

INPUT_FILE = Path("output/clean_news.parquet")
OUTPUT_DIR = Path("output/embeddings")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "hourly_embeddings.parquet"

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
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # [1, T, H]
        cls_embeddings = hidden[:, 0, :]     # [1, H]

    embedding = cls_embeddings.squeeze(0).cpu().numpy()
    return embedding.astype(np.float32)


def aggregate_news(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("hour")
        .agg({
            "text": lambda s: " ".join(s),
            "positive": "sum",
            "negative": "sum",
        })
        .rename(columns={"text": "news_text"})
        .reset_index()
    )
    return agg


def main():
    print("\n=== Generating Embeddings ===")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Clean news parquet not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)
    print(f"✔ Loaded clean news: {df.shape}")
    agg_news = aggregate_news(df)
    print(f"✔ Aggregated into {len(agg_news)} hourly rows")

    tokenizer, model = load_model()

    embeddings = []
    for _, row in tqdm(agg_news.iterrows(), total=len(agg_news)):
        emb = get_sentence_embedding(row["news_text"], tokenizer, model)
        embeddings.append(emb.tolist())

    agg_news["embedding"] = embeddings
    agg_news.to_parquet(OUTPUT_FILE)
    print(f"Saved hourly embeddings → {OUTPUT_FILE}")
    print("\nEmbedding pipeline finished!")


if __name__ == "__main__":
    main()

