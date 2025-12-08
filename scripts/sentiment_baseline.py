"""Generate FinBERT sentiment scores and fit a simple LR SOTA baseline with trading stats."""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

NEWS_FILE = Path("output/data/clean_news.parquet")
MARKET_FILE = Path("output/data/clean_market.parquet")
SENTIMENT_DIR = Path("output/sentiment")
REPORT_DIR = Path("output/reports")
SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

HOURS_PER_YEAR = 24 * 365


@dataclass
class RegressionMetrics:
    mse: float
    rmse: float
    mae: float
    mape_pct: float
    r2: float
    directional_accuracy: float
    pearson_ic: float
    spearman_ic: float
    precision_up: float
    recall_up: float
    precision_down: float
    recall_down: float
    pearson_ic: float
    spearman_ic: float


@dataclass
class StrategyMetrics:
    threshold: float
    avg_hourly_return: float
    cumulative_return: float
    sharpe: float
    hit_rate: float
    long_ratio: float
    short_ratio: float
    flat_ratio: float


def to_native_dict(mapping: Dict) -> Dict:
    """Convert numpy scalar values in a dict to native Python scalars."""
    native = {}
    for key, value in mapping.items():
        if isinstance(value, np.generic):
            native[key] = value.item()
        else:
            native[key] = value
    return native


def load_data(symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    news = pd.read_parquet(NEWS_FILE)
    market = pd.read_parquet(MARKET_FILE)

    news = news.copy()
    news["newsTimestamp"] = pd.to_datetime(news["newsTimestamp"], utc=True, errors="coerce")
    news["hour"] = news["newsTimestamp"].dt.ceil("H").dt.tz_convert(None)
    news["primary_symbol"] = news["primary_symbol"].fillna("BTC")

    market = market.copy()
    market["hour"] = pd.to_datetime(market["hour"], utc=True, errors="coerce").dt.tz_convert(None)
    market = market[market["symbol"] == symbol].sort_values("hour")

    return news[news["primary_symbol"] == symbol].reset_index(drop=True), market.reset_index(drop=True)


def score_sentiment(news_df: pd.DataFrame, model_name: str, device: str, batch_size: int) -> pd.DataFrame:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    texts = news_df["text"].tolist()
    outputs = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        outputs.append(probs)

    if not outputs:
        return news_df.assign(
            prob_positive=np.nan,
            prob_neutral=np.nan,
            prob_negative=np.nan,
            sentiment_score=np.nan,
            sentiment_label=np.nan,
        )

    probs = np.vstack(outputs)
    # FinBERT order: [negative, neutral, positive]
    prob_negative = probs[:, 0]
    prob_neutral = probs[:, 1]
    prob_positive = probs[:, 2]
    sentiment_score = prob_positive - prob_negative
    labels = np.argmax(probs, axis=1)
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    sentiment_label = [label_map[idx] for idx in labels]

    out = news_df.copy()
    out["prob_positive"] = prob_positive
    out["prob_neutral"] = prob_neutral
    out["prob_negative"] = prob_negative
    out["sentiment_score"] = sentiment_score
    out["sentiment_label"] = sentiment_label
    return out


def aggregate_hourly(sent_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        sent_df.groupby(["primary_symbol", "hour"], as_index=False)
        .agg(
            sentiment_score=("sentiment_score", "mean"),
            prob_positive=("prob_positive", "mean"),
            prob_neutral=("prob_neutral", "mean"),
            prob_negative=("prob_negative", "mean"),
            news_count=("sentiment_score", "size"),
        )
        .sort_values("hour")
    )
    return grouped


def compute_returns(market_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = market_df.sort_values("hour").copy()
    df[f"close_t+{horizon}"] = df["close"].shift(-horizon)
    df["return"] = (df[f"close_t+{horizon}"] - df["close"]) / df["close"]
    return df.iloc[:-horizon][["hour", "return"]]


def split_by_cutoff(df: pd.DataFrame, cutoff: pd.Timestamp, pretest_fraction: float):
    pre = df[df["hour"] < cutoff].sort_values("hour")
    holdout = df[df["hour"] >= cutoff].sort_values("hour")

    if pre.empty:
        return df.iloc[0:0], df.iloc[0:0], holdout

    split_idx = int(len(pre) * pretest_fraction)
    train = pre.iloc[:split_idx]
    test = pre.iloc[split_idx:]
    return train, test, holdout


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0


def _directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = (np.abs(y_true) > 1e-8) | (np.abs(y_pred) > 1e-8)
    if not np.any(mask):
        return float("nan")
    return np.mean(np.sign(y_true[mask]) == np.sign(y_pred[mask]))


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, ...]:
    up_true = (y_true > 0).astype(int)
    up_pred = (y_pred > 0).astype(int)
    down_true = (y_true < 0).astype(int)
    down_pred = (y_pred < 0).astype(int)

    precision_up, recall_up, _, _ = precision_recall_fscore_support(
        up_true, up_pred, average="binary", zero_division=0
    )
    precision_down, recall_down, _, _ = precision_recall_fscore_support(
        down_true, down_pred, average="binary", zero_division=0
    )
    return precision_up, recall_up, precision_down, recall_down


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = _safe_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    dir_acc = _directional_accuracy(y_true, y_pred)
    pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    spearman = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))
    precision_up, recall_up, precision_down, recall_down = _classification_metrics(
        y_true, y_pred
    )
    return RegressionMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        mape_pct=mape,
        r2=r2,
        directional_accuracy=dir_acc,
        pearson_ic=pearson,
        spearman_ic=spearman,
        precision_up=precision_up,
        recall_up=recall_up,
        precision_down=precision_down,
        recall_down=recall_down,
    )


def compute_signal_threshold(y_train: np.ndarray, percentile: float) -> float:
    if not 0 < percentile < 100:
        raise ValueError("percentile must be between 0 and 100")
    return float(np.percentile(np.abs(y_train), percentile))


def simulate_trading_strategy(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float
) -> StrategyMetrics:
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    positions = np.where(
        y_pred >= threshold,
        1,
        np.where(y_pred <= -threshold, -1, 0),
    )
    returns = positions * y_true
    avg_hourly_return = float(np.mean(returns))
    cumulative_return = float(np.prod(1 + returns) - 1)
    sharpe = float(
        np.sqrt(HOURS_PER_YEAR) * avg_hourly_return / (np.std(returns) + 1e-8)
    )
    hit_rate = float(
        np.mean(np.sign(returns[positions != 0]) == np.sign(y_true[positions != 0]))
    ) if np.any(positions != 0) else float("nan")
    long_ratio = float(np.mean(positions == 1))
    short_ratio = float(np.mean(positions == -1))
    flat_ratio = float(np.mean(positions == 0))
    return StrategyMetrics(
        threshold=threshold,
        avg_hourly_return=avg_hourly_return,
        cumulative_return=cumulative_return,
        sharpe=sharpe,
        hit_rate=hit_rate,
        long_ratio=long_ratio,
        short_ratio=short_ratio,
        flat_ratio=flat_ratio,
    )


def plot_strategy_curves(subset_name, positions, returns, timestamps, output_dir):
    if len(positions) == 0:
        return
    x = pd.to_datetime(timestamps)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].step(x, positions, where="post")
    axes[0].set_ylabel("Position")
    axes[0].set_title(f"{subset_name} positions")
    cum_returns = np.cumsum(returns)
    axes[1].plot(x, cum_returns)
    axes[1].set_ylabel("Cumulative Return")
    axes[1].set_xlabel("Time")
    axes[1].set_title(f"{subset_name} cumulative return")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig_path = output_dir / f"strategy_{subset_name}.png"
    fig.savefig(fig_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="FinBERT sentiment → LR baseline for next-hour returns.")
    parser.add_argument("--symbol", type=str, default="BTC")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--cutoff-date", type=str, default="2024-10-01")
    parser.add_argument("--pretest-fraction", type=float, default=0.8)
    parser.add_argument("--signal-percentile", type=float, default=50.0)
    parser.add_argument("--include-holdout", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--model-name", type=str, default="ProsusAI/finbert")
    args = parser.parse_args()

    print(f"=== Sentiment baseline for {args.symbol} ===")
    news_df, market_df = load_data(args.symbol)
    print(f"Loaded news rows: {len(news_df)}, market rows: {len(market_df)}")
    # scored = score_sentiment(news_df, model_name=args.model_name, device=args.device, batch_size=args.batch_size)
    # hourly = aggregate_hourly(scored)
    # hourly.to_parquet(SENTIMENT_DIR / f"{args.symbol}_hourly_sentiment.parquet", index=False)
    # print(f"Saved hourly sentiment → {SENTIMENT_DIR / f'{args.symbol}_hourly_sentiment.parquet'}")
    hourly = pd.read_parquet(SENTIMENT_DIR / f"{args.symbol}_hourly_sentiment.parquet")

    returns = compute_returns(market_df, horizon=args.horizon)
    merged = hourly.merge(returns, on="hour", how="inner").dropna(subset=["sentiment_score", "return"])
    if merged.empty:
        raise ValueError("No overlapping hours between sentiment and returns.")

    cutoff = pd.to_datetime(args.cutoff_date)
    train_df, test_df, holdout_df = split_by_cutoff(merged, cutoff=cutoff, pretest_fraction=args.pretest_fraction)

    if train_df.empty or test_df.empty:
        raise ValueError("Not enough data before cutoff to form train/test splits.")

    lr = LinearRegression()
    lr.fit(train_df[["sentiment_score"]].values, train_df["return"].values)

    threshold = compute_signal_threshold(train_df["return"].values, args.signal_percentile)

    def _eval_subset(name, subset_df):
        preds = lr.predict(subset_df[["sentiment_score"]].values)
        y_true = subset_df["return"].values
        regression = compute_regression_metrics(y_true, preds)
        strategy = simulate_trading_strategy(y_true, preds, threshold=threshold)
        positions = np.where(
            preds >= threshold,
            1,
            np.where(preds <= -threshold, -1, 0),
        )
        returns = positions * y_true
        plot_strategy_curves(
            f"sentiment_{name}",
            positions,
            returns,
            subset_df["hour"].values[: len(positions)],
            REPORT_DIR,
        )
        return {
            "regression_metrics": to_native_dict(asdict(regression)),
            "strategy_metrics": to_native_dict(asdict(strategy)),
        }

    metrics = {
        "cutoff_date": args.cutoff_date,
        "horizon": args.horizon,
        "pretest_fraction": args.pretest_fraction,
        "signal_percentile": args.signal_percentile,
        "threshold": threshold,
        "samples": {
            "train": int(len(train_df)),
            "test": int(len(test_df)),
            "holdout": int(len(holdout_df)),
        },
        "train": _eval_subset("train", train_df),
        "test": _eval_subset("test", test_df),
    }

    if args.include_holdout and not holdout_df.empty:
        metrics["holdout"] = _eval_subset("holdout", holdout_df)

    out_path = REPORT_DIR / f"sentiment_lr_{args.symbol}.json"
    with out_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved sentiment LR metrics → {out_path}")
    # Quick console summary so trading metrics are visible without opening JSON.
    def _print_block(name, block):
        reg = block["regression_metrics"]
        strat = block["strategy_metrics"]
        print(
            f"[{name}] R2={reg['r2']:.4f} RMSE={reg['rmse']:.4e} "
            f"DA={reg['directional_accuracy']:.3f} "
            f"Sharpe={strat['sharpe']:.3f} CumRet={strat['cumulative_return']:.4f} "
            f"Long/Short/Flat=({strat['long_ratio']:.2f}/{strat['short_ratio']:.2f}/{strat['flat_ratio']:.2f})"
        )

    _print_block("train", metrics["train"])
    _print_block("test", metrics["test"])
    if "holdout" in metrics:
        _print_block("holdout", metrics["holdout"])


if __name__ == "__main__":
    main()
