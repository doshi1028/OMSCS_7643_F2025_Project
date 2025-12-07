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
│ ├── model.py # MLP, GRU， LSTM, Transformer models
│ ├── train.py # Training loop with early stopping
│ ├── predict.py # Generate predictions using best model
│
│ ├── hypersearch.py        # First round board based parameter tuning (random)
│ ├── hypersearch2.py       # Second round model key parameters tuning (parameter by parameter) 
│ ├── model_select.py       # Select best model from tuning runs (based on IC, SR (gap between test/holdout), and flat ratio
│
│   ├── sentiment_baseline.py # Simple FinBERT sentiment score baseline
│   ├── finbert_embed_demo.py # Demo to explore FinBERT embeddings
│   ├── latent_inspection.py  # Latent factor to examine the model explanations
│   ├── test_holdout.py       # Holdout evaluation script
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
python scripts/predict.py --model lr --seq_len 1 --cutoff_date 2024-10-01
```

Loads the trained checkpoint and produces `output/predictions/predictions_<model>.csv` containing `pred` (forecasted next-hour return), `target` (realized return), the associated `timestamp`, and `subset` labels (`train` vs `holdout`, based on the cutoff date). Sequence lengths >1 reuse the same label alignment logic as training.

## 6. Performance evaluation

```
python scripts/evaluate.py --cutoff-date 2024-10-01 \
    --pretest-fraction 0.2 \
    --predictions output/predictions/predictions_<model>.csv
```

## 7. Parameter Tuning and Best Model Selection
hypersearch.py
Broad first-round hyperparameter sweep using random search across each model’s configuration space. Helps identify promising parameter ranges before more targeted tuning.
```
#In jupyter 
from scripts.hypersearch import HyperSearch
hs = HyperSearch(max_runs=30000, search_mode="full")
hs.run()
```
hypersearch2.py
Refined second-stage tuning that varies key parameters one at a time. Builds on the first-round search to narrow in on model-specific optimal settings.
```
python scripts/hypersearch2.py --mode modelwise --max-runs 2000
```
model_select.py
Utility for ranking and comparing all tuning runs. Selects the best model based on information coefficient (IC), Sharpe ratio, and test–holdout stability metrics.
```
python scripts/model_select.py
```
The evaluator:

1. Fits the **linear regression baseline** on data strictly before the cutoff date, splitting that pre-cutoff segment chronologically (default 80/20) so the baseline test metrics reflect unseen-but-pre-cutoff data. The post-cutoff window remains untouched for final holdout evaluation.
2. If `--predictions` is provided, scores that CSV (any downstream model). When a `subset` column exists, only the `train`/`holdout` metrics are reported for that file (no full-sample aggregate).
3. Reports regression metrics (MSE, RMSE, MAE, MAPE, R², directional accuracy, Pearson/Spearman information coefficients, up/down precision & recall).
4. Runs a naive long/flat/short backtest by thresholding predicted returns (threshold learned from the baseline’s training split).

Results are saved to `output/reports/performance_report.json`, and timestamped strategy plots are written alongside the JSON.

### Holdout-only validation

After settling on hyperparameters, evaluate the final checkpoint exclusively on the untouched holdout window:

```
python scripts/test_holdout.py --model lr --seq_len 1 --cutoff_date 2024-10-01
```

This script reloads the saved model, scores all post-cutoff samples, prints regression/strategy metrics, and stores the holdout strategy plots under `output/reports/`.

# FinBERT embedding proof of concept

```
python scripts/finbert_embed_demo.py "Bitcoin rallies after ETF rumors surface"
```

Prints the pooled embedding vector and dimensionality. Override defaults with `--device cpu` or `--model-name <hf_id>`.
