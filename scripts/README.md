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

