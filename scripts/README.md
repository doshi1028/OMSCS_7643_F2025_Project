# OMSCS 7463: Deep Learning  
## Final Project — Financial News Embeddings for Crypto Return Prediction

### Team Members
- Student A  
- Student B  

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

All steps are automated via `run_all.sh`.

---

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
├── src/
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


