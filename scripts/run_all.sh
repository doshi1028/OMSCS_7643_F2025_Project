#!/bin/bash

# ============================================
#  Run the entire pipeline end-to-end
#  Usage:
#      ./run_all.sh mlp 1
#      ./run_all.sh lstm 12
#      ./run_all.sh transformer 12
#
#  If no args given -> default model=mlp, seq_len=1
# ============================================

MODEL=${1:-mlp}
SEQ_LEN=${2:-1}

echo "=============================================="
echo "OMSCS 7463 PROJECT: FULL PIPELINE LAUNCH"
echo "Model:     $MODEL"
echo "Seq Len:   $SEQ_LEN"
echo "=============================================="

echo ""
echo "=== Step 1: Preprocess data ==="
python src/preprocess.py
if [ $? -ne 0 ]; then
    echo "preprocess.py failed"
    exit 1
fi

echo ""
echo "=== Step 2: Generate embeddings ==="
python src/embedding.py
if [ $? -ne 0 ]; then
    echo "embedding.py failed"
    exit 1
fi

echo ""
echo "=== Step 3: Build ML features ==="
python src/build_features.py
if [ $? -ne 0 ]; then
    echo "build_features.py failed"
    exit 1
fi

echo ""
echo "=== Step 4: Train model ($MODEL) ==="
python src/train.py --model $MODEL --seq_len $SEQ_LEN
if [ $? -ne 0 ]; then
    echo "train.py failed"
    exit 1
fi

echo ""
echo "=== Step 5: Predict using best model ==="
python src/predict.py --model $MODEL --seq_len $SEQ_LEN
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
