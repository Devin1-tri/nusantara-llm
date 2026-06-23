# 🇮🇩 Nusantara LLM — Fine-Tuning Large Language Models for Indonesian & Regional Languages

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)
![Model](https://img.shields.io/badge/model-Llama%203.1%2070B-orange)

## 🎯 Mission

Nusantara LLM aims to bridge the **representation gap** in large language models for **Indonesian (Bahasa Indonesia)** and its **700+ regional languages** (Javanese, Sundanese, Minangkabau, Balinese, Buginese, etc.).

While models like Llama 3.1 70B perform well on English benchmarks, their performance degrades significantly for Indonesian morphosyntax and is near-random for regional languages. We are **full-parameter fine-tuning** a 70B-parameter model to create:

- **Nusantara-70B-Base** — A foundation model with strong Indonesian/regional language capabilities
- **Nusantara-70B-Instruct** — Instruction-tuned variant for downstream applications
- **Nusantara-70B-Chat** — Chat-optimized variant for conversational AI

## 🏗️ Architecture

```
Base Model: meta-llama/Llama-3.1-70B (or Qwen/Qwen2.5-72B)
Fine-Tuning: Full Parameter (not LoRA/QLoRA)
Parallelism: FSDP (Fully Sharded Data Parallel) + DeepSpeed ZeRO-3
Precision: BF16 mixed precision
Context Length: 32,768 tokens
```

### Why Full-Parameter Fine-Tuning?

| Approach | VRAM Required | Quality Gain | Convergence Speed |
|----------|--------------|--------------|-------------------|
| LoRA (r=64) | ~16 GB | Moderate | Slow |
| QLoRA (4-bit) | ~8 GB | Limited | Slow |
| **Full FT (BF16)** | **~140 GB** | **Maximum** | **Fastest** |

Full-parameter fine-tuning on **H200 (141 GB HBM3e)** lets us fit the entire 70B model in **BF16 on a single GPU** — no tensor parallelism across nodes needed, drastically reducing communication overhead and training instability. This is the key advantage of H200 over H100 (80 GB).

## 📊 Dataset

We curate the **Nusantara Corpus v1**, comprising:

| Source | Size | Languages |
|--------|------|-----------|
| OSCAR Indonesian subset | 12B tokens | id |
| CommonCrawl Indonesian | 8B tokens | id |
| Wikipedia (id, jv, su, ms) | 1.2B tokens | id, jv, su, ms |
| Local news crawl | 3B tokens | id, jv, su |
| Government documents | 500M tokens | id |
| Parallel corpus (id-en) | 200M tokens | id, en |
| Regional language web | 1B tokens | jv, su, min, bug, bal |
| Instruction tuning (synthetic) | 5M pairs | id |
| **Total** | **~26B tokens** | **8+ languages** |

## 🚀 Training Pipeline

```
Raw Data → Dedup → Quality Filter → Tokenize → Shard → Train → Evaluate → Release
```

### Hardware Requirements

| Component | Requirement | Why |
|-----------|-------------|-----|
| GPU | **NVIDIA H200 (141 GB)** | Fits 70B model BF16 single-GPU |
| CPU | ≥64 cores | Data loading / preprocessing |
| RAM | ≥256 GB | Dataset caching |
| Storage | ≥2 TB NVMe | Dataset + checkpoints |
| Network | ≥25 Gbps | Fast checkpoint save/load |

### Training Configuration

- **Batch Size:** 2 per GPU (gradient accumulation ×16 → effective 32)
- **Learning Rate:** 1e-5 with cosine decay
- **Warmup:** 200 steps
- **Weight Decay:** 0.1
- **Optimizer:** AdamW (BF16 master weights)
- **Precision:** BF16 mixed precision (FSDP)
- **Checkpointing:** Every 500 steps (full state dict)
- **Target Steps:** 10,000 (approximately 1 epoch on curated subset)

## 📈 Evaluation Benchmarks

| Benchmark | What It Measures |
|-----------|-----------------|
| IndoMMLU | Indonesian knowledge + reasoning |
| IndoNLG | Generation quality (summarization, translation) |
| NusaX | Regional language sentiment + translation |
| Indo4B | Indonesian NLI, QA, sentiment, POS tagging |
| WMT Id-En | Indonesian ↔ English translation |
| Custom Perplexity | Per-language perplexity on held-out text |

## 🛠️ Getting Started

```bash
# Clone
git clone https://github.com/<your-org>/nusantara-llm
cd nusantara-llm

# Environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download & prepare dataset
python data/prepare_dataset.py --output ./data/processed

# Launch training (single GPU H200)
bash training/run_training.sh
```

## 📁 Project Structure

```
nusantara-llm/
├── README.md                   # This file
├── requirements.txt            # Dependencies
├── Makefile                    # Common commands
├── config/
│   ├── training.yaml           # Training hyperparameters
│   └── accelerate_config.yaml  # FSDP / DeepSpeed config
├── data/
│   ├── prepare_dataset.py      # Dataset download + preprocessing
│   ├── dataset_stats.py        # Dataset analysis + statistics
│   └── tokenize.py             # Tokenization pipeline
├── training/
│   ├── run_fsdp.py             # Main FSDP training script
│   ├── run_training.sh         # Launch script
│   └── monitor.py              # Training monitoring (W&B)
├── evaluation/
│   ├── evaluate.py             # Benchmark evaluation
│   └── benchmarks.md           # Evaluation configs
├── inference/
│   ├── serve.py                # vLLM / TGI inference server
│   └── client.py               # API client example
├── scripts/
│   ├── setup_env.sh            # Environment setup
│   └── download_model.sh       # Model download
└── notebooks/
    └── exploratory_analysis.ipynb  # Dataset exploration
```

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

Apache 2.0

---

*Building together. For Indonesia.*
