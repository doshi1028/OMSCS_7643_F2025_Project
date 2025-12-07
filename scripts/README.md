# OMSCS 7463: Deep Learning  
## Final Project — Financial News Embeddings for Crypto Return Prediction

### FinSignalX Team Members  
- Zhenning Liu 
- YongCheng Li
- Yang Jiao

---

# 📌 Overview

This project investigates whether **financial news sentiment and textual embeddings** can predict **short-term cryptocurrency returns**.  
We integrate:

- Hourly crypto market data  
- CryptoPanic financial news  
- Transformer-based financial language models (FinBERT / FinGPT)  
- Deep learning regressors (MLP, LSTM, Transformer Encoder)

Our pipeline:

1. Cleans and aligns news + market data  
2. Generates sentence-level embeddings using FinBERT  
3. Aggregates embeddings into hourly features  
4. Builds a supervised dataset (X, y)  
5. Trains ML models to predict **next-hour return**  
6. Evaluates prediction performance  
7. Tuning and model Selection

All steps are automated via `run_all.sh`.

# 🏃‍♂️ How to Run 

suports **2 run methods**——one-click (recommend) or step by step.

---

## ✅ **Option 1: One-click Full Pipeline**

### **default mode（MLP，single hour）**
./run_all.sh

### **run LSTM with sequence length 12 (fox example)**
./run_all.sh transformer 12


The whole procedure：

- clean data
- generate FinBERT embeddings
- clustering hourly features
- construct X, y
- model training
- output predicted results

results saved at: 
- output/models/
- output/predictions/


---

# ✅ **Option 2: Step-by-Step Execution**

Run below script step by step in debugging mode: 

---

## **1. Preprocess raw data**
python src/preprocess.py

- `output/clean_news.parquet`
- `output/clean_market.parquet`
- `output/merged_dataset.parquet`

---

## **2. Generate FinBERT embeddings**
python src/embedding.py

- `output/embeddings/BTC_embeddings.parquet`

---

## **3. Build ML dataset**
python src/build_features.py --horizon 1

- `Horizon defines the hour length of future return`
- `output/features/X.npy`
- `output/features/y.npy`

---

## **4. Train model**
python scripts/train.py --model lstm --seq_len 12 --hidden_dim 256 --num_layers 2     --lstm_proj_dim 128 --lstm_use_attention 1

- `output/models/<model_name>_best.pt`
  
---
## **5. Predict**
python src/predict.py --model lstm --seq_len 12
- `output/predictions/predictions_lstm.csv`


---

## 6. Performance evaluation

```
python scripts/evaluate.py --cutoff-date 2024-10-01 \
    --pretest-fraction 0.2 \
    --predictions output/predictions/predictions_<model>.csv
```
---

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


# 📊 Models

This project provides several regression architectures for forecasting future crypto returns based on news embeddings. Each model supports variable `seq_len`, adjustable prediction horizon (`--horizon`), EMA, learning rate scheduling, and standardized MSE regression.

## 1. MLP Baseline
A simple feedforward network using only the current-hour embedding.

**Arguments**
- `--hidden_dim`: hidden layer width (default 256)
- `--dropout`: dropout rate (default 0.1)

**Example**
```bash
python scripts/train.py --model mlp --hidden_dim 512 --dropout 0.2
```

## 2. LSTM Regressor
Sequence model over past `seq_len` hours. Supports LayerNorm and Attention pooling.

**Arguments**
- `--seq_len`: sequence length
- `--hidden_dim`: LSTM hidden dimension (default 256)
- `--num_layers`: number of layers (default 2)
- `--dropout`: dropout rate (default 0.1)
- `--lstm_proj_dim`: projection dimension after LSTM (default 128)
- `--lstm_use_layernorm`: enable LayerNorm (0/1)
- `--lstm_use_attention`: enable attention pooling (0/1)

**Example**
```bash
python scripts/train.py --model lstm --seq_len 12 --hidden_dim 256 --num_layers 2     --lstm_proj_dim 128 --lstm_use_attention 1
```

## 3. Transformer Encoder
Multi-head attention encoder for longer sequences. Supports CLS token, learnable/sinusoidal positions, and multiple pooling methods.

**Arguments**
- `--seq_len`: input sequence length
- `--tf_d_model`: transformer hidden dimension (default 128)
- `--tf_heads`: number of attention heads (default 4)
- `--tf_layers`: number of encoder blocks (default 4)
- `--tf_ff_dim`: feedforward dimension (default 256)
- `--tf_dropout`: dropout rate (default 0.1)
- `--tf_learnable_pos`: learnable positional embeddings (0/1)
- `--tf_use_cls_token`: add CLS token (0/1)
- `--tf_pool`: pooling strategy (`attention`, `cls`, `mean`, `last`)
- `--tf_embed_scale`: embedding scaling factor (default 1.0)

**Example**
```bash
python scripts/train.py --model transformer --seq_len 12     --tf_d_model 128 --tf_heads 4 --tf_layers 4     --tf_pool attention --tf_dropout 0.1
```

## 4. GRU Regressor
Lightweight alternative to LSTM with attention pooling.

**Arguments**
- `--seq_len`: input sequence length
- `--hidden_dim`: GRU hidden dimension (default 256)
- `--num_layers`: number of layers (default 2)
- `--dropout`: dropout rate (default 0.2)

**Example**
```bash
python scripts/train.py --model gru --seq_len 12 --hidden_dim 256 --num_layers 2
```

## 5. Linear Regression (LR)
Simple linear baseline using only the flattened embedding.

**Example**
```bash
python scripts/train.py --model lr
```

## Prediction Usage
Prediction automatically loads model configuration:
```bash
python scripts/predict.py --model transformer --seq_len 12
```


# 📈 Dataset

### **Features (X)**  
- 768-dim FinBERT embedding  
- expantable：  
  - number of news
  - pos/neg sentimantal count
  - index（SMA、RSI、MACD）  

### **Labels (y)**  
return_t = (close[t+1] - close[t]) / close[t]


---

# 🔧 Installation
pip install -r requirements.txt


---

# 🔒 Ethical Notes

- Data are all public data 
- results cannot be used for trading 
- news sentiment may have bias  

---

# 🧠 Reproducibility

./run_all.sh transformer 12












