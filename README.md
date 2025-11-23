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

- `output/data/clean_news.parquet` – cleaned headlines with hourly timestamps  
- `output/data/clean_market.parquet` – normalized OHLCV per symbol

> ⚠️ Temporary assumption: every headline is attributed to BTC while the spaCy-based asset-tagging module is under development. Update `align_news_to_market` once multi-asset tagging is ready so each `(symbol, hour)` receives only its own news texts.

## 2. Embedding generation

```
python scripts/embedding.py
```

Reads `output/data/clean_news.parquet`, aggregates headlines per hour, and runs FinBERT (configurable via `MODEL_NAME`). Saves `output/embeddings/hourly_embeddings.parquet`, which holds one embedding vector per hour (plus sentiment counts).

## 3. Feature building

```
python scripts/build_features.py
```

Merges `output/data/clean_market.parquet` with `output/embeddings/hourly_embeddings.parquet`, computes next-hour returns, and averages embeddings over a configurable lookback window (default 12 hours). Outputs:

- `output/features/X.npy` – averaged embedding features
- `output/features/y.npy` – aligned next-hour returns
- `output/features/dataset.parquet` – audit dataframe

## 4. Model training

```
python scripts/train.py --model lr --seq_len 1 --epochs 30
```

- `--model`: `lr`, `mlp`, `lstm`, or `transformer` (see `scripts/model.py`). The LR option is the explicit baseline; the other deep models complement it.
- `--seq_len`: if >1, overlapping sequences of that length are fed into the model, with labels aligned to the final timestep.

Saves the best checkpoint to `output/models/<model>_best.pt`.

## 5. Prediction

```
python scripts/predict.py --model lr --seq_len 1
```

Loads the trained checkpoint and produces `output/predictions/predictions_<model>.csv` containing `pred` (forecasted next-hour return), `target` (realized return), and `subset` labels (`train` vs `test`, based on the chronological split defined by `--train_split`). Sequence lengths >1 reuse the same label alignment logic as training.

## 6. Performance evaluation

```
python scripts/evaluate.py --predictions output/predictions/predictions_<model>.csv
```

The evaluator:

1. Fits the **linear regression baseline** on `output/features/X.npy` vs. `y.npy` (chronological split).
2. If `--predictions` is provided, scores that CSV (any downstream model). When a `subset` column exists, only the `train`/`test` metrics are reported for that file (no full-sample aggregate).
3. Reports regression metrics (MSE, RMSE, MAE, MAPE, R², directional accuracy, Pearson/Spearman information coefficients, up/down precision & recall).
4. Runs a naive long/flat/short backtest by thresholding predicted returns (threshold learned from the baseline’s training split).

Results are saved to `output/reports/performance_report.json`.

# FinBERT embedding proof of concept

```
python scripts/finbert_embed_demo.py "Bitcoin rallies after ETF rumors surface"
```

Prints the pooled embedding vector and dimensionality. Override defaults with `--device cpu` or `--model-name <hf_id>`.
