"""
Nusantara LLM — Evaluation Suite
Runs standard benchmarks on fine-tuned models.
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


BENCHMARK_CONFIGS = {
    "indommlu": {
        "description": "Indonesian MMLU — knowledge and reasoning",
        "dataset": "indonlp/indommlu",
        "metric": "accuracy",
        "split": "test",
    },
    "indonlg": {
        "description": "Indonesian NLG — generation quality",
        "dataset": "indonlp/indonlg",
        "metric": "rouge",
        "split": "test",
    },
    "nusax": {
        "description": "NusaX — Regional language sentiment + translation",
        "dataset": "indonlp/nusax",
        "metric": "accuracy",
        "split": "test",
    },
    "indo4b": {
        "description": "Indo4B — NLI, QA, sentiment, POS",
        "dataset": "indonlp/indo4b",
        "metric": "accuracy",
        "split": "test",
    },
}


def load_model_for_eval(model_path: str) -> tuple:
    """Load model in eval mode."""
    logger.info(f"Loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def compute_perplexity(
    model, tokenizer, texts: List[str], max_length: int = 32768
) -> Dict[str, float]:
    """Compute per-language perplexity on held-out text."""
    logger.info("Computing perplexity...")
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in tqdm(texts, desc="Perplexity"):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(model.device)

            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            num_tokens = inputs["input_ids"].shape[1]

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))

    return {"perplexity": perplexity, "avg_loss": avg_loss}


def evaluate_benchmark(
    model, tokenizer, benchmark_name: str, config: Dict
) -> Dict:
    """Run a specific benchmark."""
    logger.info(f"Running benchmark: {benchmark_name} ({config['description']})")

    try:
        dataset = load_dataset(config["dataset"], split=config["split"])
    except Exception as e:
        logger.warning(f"Could not load {config['dataset']}: {e}")
        return {"error": str(e)}

    if config["metric"] == "accuracy":
        correct = 0
        total = 0
        for example in tqdm(dataset, desc=f"Evaluating {benchmark_name}"):
            # Format as multiple-choice / classification task
            if "question" in example and "choices" in example:
                prompt = f"Pertanyaan: {example['question']}\n\nPilihan:\n"
                for i, choice in enumerate(example["choices"]):
                    prompt += f"{chr(65+i)}. {choice}\n"
                prompt += "\nJawaban yang benar adalah huruf:"
            elif "text" in example:
                prompt = example["text"]
            else:
                continue

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            # Check answer
            answer = str(example.get("answer", example.get("label", ""))).strip().upper()
            if answer and response.strip().upper().startswith(answer[0]):
                correct += 1
            total += 1

        accuracy = correct / max(total, 1)
        return {"accuracy": accuracy, "correct": correct, "total": total}

    elif config["metric"] == "rouge":
        rouge = evaluate.load("rouge")
        predictions, references = [], []
        for example in tqdm(dataset, desc=f"Evaluating {benchmark_name}"):
            source = example.get("source", example.get("text", ""))
            reference = example.get("target", example.get("summary", ""))

            if not source or not reference:
                continue

            inputs = tokenizer(source, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            prediction = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            predictions.append(prediction)
            references.append(reference)

        results = rouge.compute(predictions=predictions, references=references)
        return results

    return {"error": f"Unknown metric: {config['metric']}"}


def main():
    parser = argparse.ArgumentParser(description="Nusantara LLM Evaluation")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to fine-tuned model")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["indommlu", "indonlg", "nusax", "indo4b"],
                        help="Benchmarks to run")
    parser.add_argument("--perplexity-texts", type=str, default=None,
                        help="File with texts for perplexity computation")
    parser.add_argument("--output", type=str, default="./evaluation_results.json",
                        help="Output path for results")
    args = parser.parse_args()

    model, tokenizer = load_model_for_eval(args.model_path)

    results = {"model": args.model_path}

    # Run benchmarks
    for benchmark in args.benchmarks:
        if benchmark in BENCHMARK_CONFIGS:
            config = BENCHMARK_CONFIGS[benchmark]
            result = evaluate_benchmark(model, tokenizer, benchmark, config)
            results[benchmark] = result
            logger.info(f"  {benchmark}: {json.dumps(result, indent=2)}")

    # Perplexity
    if args.perplexity_texts:
        with open(args.perplexity_texts, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
        ppl = compute_perplexity(model, tokenizer, texts)
        results["perplexity"] = ppl
        logger.info(f"  Perplexity: {ppl['perplexity']:.2f}")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
