#!/bin/bash
# Download base model from HuggingFace
# Requires: huggingface-cli login (for gated models)

set -euo pipefail

MODEL_NAME="${1:-meta-llama/Llama-3.1-70B}"
OUTPUT_DIR="${2:-./models}"

echo "Downloading model: $MODEL_NAME"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Try using huggingface-cli
if command -v huggingface-cli &> /dev/null; then
    huggingface-cli download "$MODEL_NAME" \
        --local-dir "$OUTPUT_DIR/$(basename $MODEL_NAME)" \
        --local-dir-use-symlinks False \
        --resume-download \
        --exclude "*.safetensors.index.json" \
        --include "*.safetensors" \
        --include "*.json" \
        --include "*.model" \
        --include "tokenizer*" \
        --include "config*" \
        --include "special_tokens_map*"
    echo "✓ Model downloaded to: $OUTPUT_DIR/$(basename $MODEL_NAME)"
else
    echo "huggingface-cli not found. Install with: pip install huggingface-hub"
    echo ""
    echo "Alternative: use git lfs:"
    echo "  git lfs install"
    echo "  git clone https://huggingface.co/$MODEL_NAME $OUTPUT_DIR/$(basename $MODEL_NAME)"
    exit 1
fi
