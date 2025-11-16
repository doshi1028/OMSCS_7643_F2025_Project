"""Proof-of-concept script to turn news text into FinBERT embeddings."""

from __future__ import annotations

import argparse
from typing import List

import torch
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "ProsusAI/finbert"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a piece of news text into a dense embedding using FinBERT. "
            "The script prints the pooled embedding to stdout."
        )
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="News text snippet. Pass multiple tokens or enclose the full text in quotes.",
    )
    parser.add_argument(
        "--model-name",
        default=MODEL_NAME,
        help="Hugging Face model identifier to use for embeddings.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device, defaults to CUDA when available.",
    )
    return parser.parse_args()


def mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Compute attention-mask-aware mean pooling."""
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    summed = torch.sum(hidden_states * mask, dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def build_embedding(
    text: str,
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str = "cpu",
) -> torch.Tensor:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoded)

    pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
    return pooled.squeeze(0).cpu()


def main() -> None:
    args = parse_args()
    text = " ".join(args.text).strip()
    if not text:
        raise SystemExit("Please supply a news snippet, e.g. python scripts/finbert_embed_demo.py \"BTC surges...\"")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.to(args.device)
    model.eval()

    embedding = build_embedding(text, model, tokenizer, args.device)
    print("Input text:", text)
    print("Embedding shape:", list(embedding.shape))
    print(embedding)


if __name__ == "__main__":
    main()
