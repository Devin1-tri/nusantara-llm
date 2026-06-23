# Llama 3.1 70B — Full-Parameter Fine-Tuning

Fine-tuning a 70B LLM on a single GPU using FSDP. Optimized for NVIDIA H200 (141 GB HBM3e).

This repo contains the training pipeline, dataset preparation, and inference server for full-parameter fine-tuning of Llama 3.1 70B on a single H200 — no multi-GPU parallelism needed.

## Why Single-GPU on H200?

70B parameters in BF16 = ~140 GB. The H200's **141 GB HBM3e** is the only single-GPU option that fits this. On H100 (80 GB) you'd need tensor parallelism across 2+ GPUs, adding communication overhead and complexity.

## Pipeline

```
Text Data → Tokenize → FSDP Training (BF16) → Evaluation → Export
```

### Training Config

| Parameter | Value |
|-----------|-------|
| Model | Llama 3.1 70B |
| Precision | BF16 mixed precision |
| Attention | Flash Attention 2 |
| Distributed | FSDP FULL_SHARD |
| Context | 32,768 tokens |
| Batch | 2 per GPU × 16 grad accum |
| Optimizer | AdamW, 1e-5, cosine decay |
| Steps | 10,000 |

### Hardware

| Component | Spec |
|-----------|------|
| GPU | 1× NVIDIA H200 (141 GB) |
| CPU | 16+ cores |
| RAM | 64+ GB |
| Storage | 500+ GB NVMe |

## Getting Started

```bash
git clone https://github.com/Devin1-tri/nusantara-llm
cd nusantara-llm

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run training
bash training/run_training.sh
```

## Project Structure

```
├── config/           # Training & FSDP configs
├── data/             # Dataset preparation scripts
├── training/         # FSDP training loop
├── evaluation/       # Benchmark scripts
├── inference/        # vLLM server + API client
└── scripts/          # Setup & download utilities
```

## License

Apache 2.0
