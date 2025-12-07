#!/bin/bash

# ============================================
#  Run the entire pipeline end-to-end
#  Example provided for a gru model
#

# ============================================

MODEL=${1:-lr}
SEQ_LEN=${2:-1}

echo "=============================================="
echo "OMSCS 7463 PROJECT: FULL PIPELINE LAUNCH"
echo "Model:     $MODEL"
echo "Seq Len:   $SEQ_LEN"
echo "=============================================="

echo ""
echo "=== Step 1: Preprocess data ==="
python scripts/preprocess.py
if [ $? -ne 0 ]; then
    echo "preprocess.py failed"
    exit 1
fi

echo ""
echo "=== Step 2: Generate embeddings ==="
python scripts/embedding.py
if [ $? -ne 0 ]; then
    echo "embedding.py failed"
    exit 1
fi

echo ""
echo "=== Step 3: Build ML features ==="
python scripts/build_features.py --horizon 1 --lookback 6 --lookback-mode mean
if [ $? -ne 0 ]; then
    echo "build_features.py failed"
    exit 1
fi

echo ""
echo "=== Step 4: Train model ($MODEL) ==="
python scripts/train.py --model gru --seq_len 12 --batch_size 128 --lr 0.001 --epochs 15 --weight_decay 1e-05 --scheduler cosine_warmup --warmup_pct 0.05 --hidden_dim 768 --lstm_proj_dim 32 --num_layers 3 --tf_d_model 384 --tf_heads 8 --tf_layers 4 --tf_ff_dim 256 --tf_pool cls --tf_dropout 0.1
if [ $? -ne 0 ]; then
    echo "train.py failed"
    exit 1
fi

echo ""
echo "=== Step 5: Predict using best model ==="
python scripts/predict.py --model gru --seq_len 12 --batch_size 128 --cutoff_date 2024-10-01
if [ $? -ne 0 ]; then
    echo "predict.py failed"
    exit 1
fi

echo ""
echo "=============================================="
echo "Pipeline completed successfully!"
echo "Model:      $MODEL"
echo "Seq Len:    $SEQ_LEN"
echo "Results:"
echo " - output/models/${MODEL}_best.pt"
echo " - output/predictions/predictions_${MODEL}.csv"
echo "=============================================="

echo ""
echo "=== Step 6: Evaluation ==="
python scripts/evaluate.py --include-holdout --predictions output/predictions/predictions_gru.csvff_date 2024-10-01
if [ $? -ne 0 ]; then
    echo "predict.py failed"
    exit 1
fi

echo ""
echo "=== Step 7: Hyperparameter Tuning ==="
python scripts/hypersearch2.py --mode modelwise --max-runs 2000
if [ $? -ne 0 ]; then
    echo "predict.py failed"
    exit 1
fi

echo ""
echo "=== Step 8: Best Model Selection ==="
python scripts/model_select.py
if [ $? -ne 0 ]; then
    echo "predict.py failed"
    exit 1
fi
