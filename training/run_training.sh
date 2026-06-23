#!/bin/bash
# Nusantara LLM — Training Launcher for NVIDIA H200
# Usage: bash run_training.sh

set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────
MODEL_NAME="meta-llama/Llama-3.1-70B"
DATASET_PATH="./data/processed"
OUTPUT_DIR="./checkpoints"
RUN_NAME="nusantara-llm-70b-v1"

# H200-optimized parameters
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=16   # Effective batch: 2 × 16 = 32
LEARNING_RATE=1e-5
MAX_STEPS=10000
WARMUP_STEPS=200
SAVE_STEPS=500
LOG_STEPS=10
SEED=42

# ─── Environment ────────────────────────────────────────────────────
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="0"
export OMP_NUM_THREADS=16
export NCCL_DEBUG="INFO"
export NCCL_IB_DISABLE="0"
export NCCL_P2P_DISABLE="0"
export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export WANDB_PROJECT="nusantara-llm"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# Flash Attention 2
export TORCH_CUDNN_V8_API_ENABLED="1"

# ─── Setup ──────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Nusantara LLM — H200 Training Launcher             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Model:     $MODEL_NAME"
echo "Dataset:   $DATASET_PATH"
echo "Batch:     $PER_DEVICE_BATCH_SIZE (eff: $((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)))"
echo "LR:        $LEARNING_RATE"
echo "Steps:     $MAX_STEPS"
echo "Precision: BF16 + Flash Attention 2"
echo "GPU:       NVIDIA H200 (141 GB HBM3e)"
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo "Detected GPU: $GPU_NAME ($GPU_MEM)"
    echo ""
fi

# ─── Prepare dataset ────────────────────────────────────────────────
if [ ! -d "$DATASET_PATH" ]; then
    echo "[1/3] Preparing dataset..."
    python data/prepare_dataset.py \
        --output "$DATASET_PATH" \
        --tokenizer "$MODEL_NAME" \
        --max-length 32768 \
        --max-samples 500000
    echo "✓ Dataset ready"
else
    echo "→ Dataset found: $DATASET_PATH"
fi

# ─── Dataset stats ──────────────────────────────────────────────────
echo "[2/3] Computing dataset stats..."
python data/dataset_stats.py --dataset "$DATASET_PATH"

# ─── Launch training ────────────────────────────────────────────────
echo "[3/3] Launching FSDP training..."
echo ""

# Use accelerate for FSDP config
accelerate launch \
    --config_file ./config/accelerate_config.yaml \
    training/run_fsdp.py \
    --model "$MODEL_NAME" \
    --dataset "$DATASET_PATH" \
    --output "$OUTPUT_DIR" \
    --batch-size "$PER_DEVICE_BATCH_SIZE" \
    --grad-accum "$GRADIENT_ACCUMULATION_STEPS" \
    --lr "$LEARNING_RATE" \
    --max-steps "$MAX_STEPS" \
    --warmup "$WARMUP_STEPS" \
    --save-steps "$SAVE_STEPS" \
    --run-name "$RUN_NAME"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          Training Complete!                                 ║"
echo "║          Model saved to: $OUTPUT_DIR                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
