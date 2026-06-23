.PHONY: help setup data train evaluate serve clean

VENV := .venv
PYTHON := $(VENV)/bin/python

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Create virtual environment and install dependencies
	bash scripts/setup_env.sh

data: ## Prepare training dataset
	$(PYTHON) data/prepare_dataset.py --output ./data/processed --max-samples 500000

data-stats: ## Show dataset statistics
	$(PYTHON) data/dataset_stats.py --dataset ./data/processed

train: ## Start FSDP training
	bash training/run_training.sh

evaluate: ## Run evaluation on a trained model
	$(PYTHON) evaluation/evaluate.py --model-path ./checkpoints/final

serve: ## Start inference server (vLLM)
	$(PYTHON) inference/serve.py --mode server --model-path ./checkpoints/final --port 8000

interactive: ## Interactive chat with model
	$(PYTHON) inference/serve.py --mode interactive --model-path ./checkpoints/final

clean: ## Clean checkpoints and cache
	rm -rf ./checkpoints ./data/processed ./data/tokenized __pycache__ ./.cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -delete

download-model: ## Download base model from HuggingFace
	bash scripts/download_model.sh
