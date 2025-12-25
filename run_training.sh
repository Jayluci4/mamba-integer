#!/bin/bash
# Training launch script for Mamba-Integer
# Run with: nohup bash run_training.sh > training.log 2>&1 &

set -e

cd /home/jayantlohia16/experiment/mamba-integer

echo "=========================================="
echo "Mamba-Integer Training Pipeline"
echo "Started at: $(date)"
echo "=========================================="

# Step 1: Check/create tokenized data
if [ ! -f "tinystories_train.bin" ]; then
    echo ""
    echo "[Step 1/2] Tokenizing TinyStories dataset..."
    python -u scripts/tokenize_dataset_offline.py
else
    echo ""
    echo "[Step 1/2] Tokenized data already exists (tinystories_train.bin)"
    ls -lh tinystories_train.bin
fi

# Step 2: Run training
echo ""
echo "[Step 2/2] Starting training..."
echo "=========================================="
python -u scripts/train_mamba_integer.py

echo ""
echo "=========================================="
echo "Training completed at: $(date)"
echo "=========================================="
