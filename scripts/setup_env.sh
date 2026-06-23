#!/bin/bash
# Environment setup for Nusantara LLM

set -euo pipefail

echo "Setting up Nusantara LLM environment..."

# System dependencies
sudo apt-get update -qq && sudo apt-get install -y -qq \
    build-essential \
    cmake \
    curl \
    wget \
    git-lfs \
    libopenmpi-dev \
    openmpi-bin \
    > /dev/null 2>&1 || echo "Warning: Some system packages could not be installed"

# Create Python venv
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel > /dev/null

# Install PyTorch (CUDA 12.1+)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install \
    transformers>=4.44.0 \
    accelerate>=0.33.0 \
    datasets>=2.20.0 \
    deepspeed>=0.14.0 \
    wandb>=0.17.0 \
    flash-attn>=2.6.0 \
    bitsandbytes>=0.43.0 \
    einops>=0.8.0 \
    scipy>=1.14.0 \
    tqdm>=4.66.0 \
    numpy>=1.26.0 \
    pandas>=2.2.0 \
    safetensors>=0.4.0 \
    huggingface-hub>=0.24.0 \
    tokenizers>=0.19.0 \
    evaluate>=0.4.0 \
    protobuf>=5.27.0

# Login to Hugging Face (optional, for gated models like Llama)
echo ""
echo "If you need access to Llama 3.1 70B, run: huggingface-cli login"
echo "Make sure your HuggingFace token has access to: meta-llama/Llama-3.1-70B"
echo ""

# Verify CUDA
python3 -c "import torch; print(f'PyTorch {torch.__version__} — CUDA available: {torch.cuda.is_available()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

echo ""
echo "✓ Environment ready!"
echo "Run: source .venv/bin/activate"
echo "Then: bash training/run_training.sh"
