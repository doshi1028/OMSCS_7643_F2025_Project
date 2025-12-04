This is the repo for the CS7643 Fall 2025 team project – FinSignalX.

# Overview

We study whether transformer-based financial news embeddings help forecast short-term cryptocurrency returns. The pipeline integrates:

- Hourly crypto market data
- CryptoPanic news headlines and descriptions
- FinBERT/FinGPT sentence embeddings
- Deep regressors (MLP, LSTM, Transformer Encoder)

Pipeline steps:

1. Clean and align news + market data.
2. Generate sentence embeddings per hourly bucket.
3. Aggregate embeddings into lookback features.
4. Train supervised models on next-hour returns.
5. Run inference to produce prediction CSVs.
6. Evaluate regression metrics and a simple trading strategy.

# 📂 Project Structure
```bash
project/
│
├── data/
│ ├── crypto_data_hourly/ # Hourly parquet files for BTC, ETH, etc.
│ ├── cryptopanic_news.csv # Raw CryptoPanic news dataset
│
├── output/
│ ├── clean_news.parquet # Cleaned news data
│ ├── clean_market.parquet # Cleaned market data
│ ├── merged_dataset.parquet # News aligned with market hours
│ ├── embeddings/ # Per-symbol FinBERT embeddings
│ ├── features/ # Final ML dataset (X.npy, y.npy)
│ ├── models/ # Saved models (best.pt)
│ ├── predictions/ # Model prediction results CSV
│
├── script/
│ ├── preprocess.py # Clean + align news & market data
│ ├── embedding.py # Generate FinBERT/FinGPT embeddings
│ ├── build_features.py # Build feature matrix X and labels y
│ ├── model.py # MLP, LSTM, Transformer models
│ ├── train.py # Training loop with early stopping
│ ├── predict.py # Generate predictions using best model
│
├── run_all.sh # One-click full pipeline execution
├── README.md
└── requirements.txt
```


Use `scripts/run_all.sh <model> <seq_len>` to execute all stages, or follow the commands below to run each step individually.

# Environment setup

```
conda env create -f environment.yaml
conda activate finbert-embeddings
```

# Step-by-step workflow

## 1. Data preprocessing

```
python scripts/preprocess.py
```

Inputs: `data/BTC_USD_hourly.parquet` (or directory of parquet files) and `data/cryptopanic_news.csv`.  
Outputs (stored under `output/data/`):

- `output/data/clean_news.parquet` – cleaned per-row headlines with original timestamps + symbol tags  
- `output/data/clean_market.parquet` – normalized OHLCV per symbol

> ⚠️ Temporary assumption: symbol tagging is keyword-based; refine it later if you add a dedicated NER model.

## 2. Embedding generation

```
python scripts/embedding.py
```

Reads `output/data/clean_news.parquet` and `output/data/clean_market.parquet`, aligns headlines to detected symbols, and only generates embeddings for symbol/hour pairs that actually have tagged news. Each symbol receives its own `output/embeddings/<SYMBOL>_embeddings.parquet` file containing one embedding vector per hour plus sentiment counts.

## 3. Feature building

```
python scripts/build_features.py
```

Loads `output/data/clean_market.parquet` and the per-symbol embedding files from `output/embeddings/`, merges them on `(symbol, hour)`, computes next-hour returns, and averages embeddings over a configurable lookback window (default 12 hours, configurable via `--lookback`). Outputs:

- `output/features/X.npy` – averaged embedding features
- `output/features/y.npy` – aligned next-hour returns
- `output/features/dataset.parquet` – audit dataframe

## 4. Model training

```
python scripts/train.py --model lr --seq_len 1 --epochs 30
```

- `--model`: `lr`, `mlp`, `lstm`, `gru`, or `transformer` (see `scripts/model.py`). The LR option is the explicit baseline; the other deep models complement it.
- `--seq_len`: if >1, overlapping sequences of that length are fed into the model, with labels aligned to the final timestep.
- `--train_end_date`: only samples strictly before this date are used for training/validation so the final months remain untouched for testing. Default is 2024-10-01

Saves the best checkpoint to `output/models/<model>_best.pt`.

## 5. Prediction

```
python scripts/predict.py --model lr --seq_len 1 --cutoff_date 2024-10-01 \
    --pretest_fraction 0.8
```

Loads the trained checkpoint and produces `output/predictions/predictions_<model>.csv` containing `pred` (forecasted next-hour return), `target` (realized return), the associated `timestamp`, and `subset` labels (`train`/`test` for pre-cutoff samples according to `--pretest_fraction`, `holdout` for post-cutoff samples). Sequence lengths >1 reuse the same label alignment logic as training.

## 6. Performance evaluation

```
python scripts/evaluate.py --cutoff-date 2024-10-01 \
    --pretest-fraction 0.2 \
    --predictions output/predictions/predictions_<model>.csv
```

The evaluator:

1. Fits the **linear regression baseline** on data strictly before the cutoff date, splitting that pre-cutoff segment chronologically (default 80/20) so the baseline test metrics reflect unseen-but-pre-cutoff data. The post-cutoff window remains untouched for final holdout evaluation unless `--include-holdout` is passed.
2. If `--predictions` is provided, scores that CSV (any downstream model). When a `subset` column exists, only the `train`/`test` metrics are reported unless `--include-holdout` is specified.
3. Reports regression metrics (MSE, RMSE, MAE, MAPE, R², directional accuracy, Pearson/Spearman information coefficients, up/down precision & recall).
4. Runs a naive long/flat/short backtest by thresholding predicted returns (threshold learned from the baseline’s training split).

Add `--include-holdout` to include post-cutoff performance in both the baseline and prediction metrics. Results are saved to `output/reports/performance_report.json`, and timestamped strategy plots are written alongside the JSON.

### Holdout-only validation

After settling on hyperparameters, evaluate the final checkpoint exclusively on the untouched holdout window:

```
python scripts/test_holdout.py --model lr --seq_len 1 --cutoff_date 2024-10-01
```

This script reloads the saved model, scores all post-cutoff samples, prints regression/strategy metrics, and stores the holdout strategy plots under `output/reports/`.

### Sentiment-only SOTA baseline (FinBERT)

Compute FinBERT sentiment, aggregate hourly, and fit a linear regression on next-hour returns:

```
python scripts/sentiment_baseline.py \
    --symbol BTC \
    --cutoff-date 2024-10-01 \
    --pretest-fraction 0.8 \
    --horizon 1 \
    --batch-size 16
```

Outputs:
- `output/sentiment/<SYMBOL>_hourly_sentiment.parquet` – hourly averaged sentiment scores (ceil-rounded to avoid look-ahead).
- `output/reports/sentiment_lr_<SYMBOL>.json` – train/test metrics (holdout rows remain untouched for final evaluation).

### Latent factor inspection

```
python scripts/latent_inspection.py \
    --symbol BTC \
    --topics 25 \
    --topic-words 12 \
    --pca-components 10 \
    --pc-top-k 5 \
    --run-transformer \
    --horizon 1
```

The latent inspection tool lets you peek inside the learned representation without training a brand-new forecasting model:

- **BERTopic (c‑TF‑IDF):** runs on the raw, per-row embeddings/text (with Bitcoin-specific stop-words and stripped Twitter handles/numerics) to expose coherent headline themes and their representative keywords.
- **PCA spotlight:** highlights the five most/least loaded headlines for the top PCs so you can see which topics are driving each latent direction.
- **Transformer attribution (optional):** if `--run-transformer` is passed, an existing transformer encoder is trained on the hourly PCA series to predict next-hour returns, and gradient-based feature importances are computed for every PC (saved under `transformer_analysis.importance`). This reveals which latent factors move the loss the most, even if the model is not the final production forecaster.

All summaries plus BERTopic artifacts land in `output/analysis/`.

# FinBERT embedding proof of concept

```
python scripts/finbert_embed_demo.py "Bitcoin rallies after ETF rumors surface"
```

Prints the pooled embedding vector and dimensionality. Override defaults with `--device cpu` or `--model-name <hf_id>`.
