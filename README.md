This is the repo for the CS7643 Fall 2025 team project. Team members:

Zhening Liu zliu303@gatech.edu

Yang Jiao yjiao3@gatech.edu   

Yongcheng Li yli3584@gatech.edu

## FinBERT embedding proof of concept

Set up the conda environment defined in `environment.yaml`:

```
conda env create -f environment.yaml
conda activate finbert-embeddings
```

Run the demo script with any news snippet:

```
python scripts/finbert_embed_demo.py "Bitcoin rallies after ETF rumors surface"
```

The script prints the pooled embedding vector along with its dimensionality. Set `--device cpu` to force CPU inference or `--model-name` to test other pretrained checkpoints. If you prefer pip-only setup, install `torch` and `transformers` with matching versions to those listed in `environment.yaml`.
