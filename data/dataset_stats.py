"""
Nusantara LLM — Dataset Statistics & Analysis
"""

import json
import logging
from pathlib import Path

import numpy as np
from datasets import load_from_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def compute_stats(dataset_path: str):
    """Compute and display dataset statistics."""
    data_path = Path(dataset_path)
    if not data_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return

    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_from_disk(str(data_path))

    for split_name in dataset:
        ds = dataset[split_name]
        lengths = [len(sample["input_ids"]) for sample in ds]
        tokens = sum(lengths)

        stats = {
            "samples": len(ds),
            "total_tokens": tokens,
            "mean_length": float(np.mean(lengths)),
            "median_length": float(np.median(lengths)),
            "min_length": int(np.min(lengths)),
            "max_length": int(np.max(lengths)),
            "std_length": float(np.std(lengths)),
        }

        logger.info(f"\n{'='*50}")
        logger.info(f"Split: {split_name}")
        logger.info(f"{'='*50}")
        for k, v in stats.items():
            if isinstance(v, float):
                logger.info(f"  {k}: {v:,.2f}")
            else:
                logger.info(f"  {k}: {v:,}")

    # Aggregate
    total_tokens = sum(
        sum(len(sample["input_ids"]) for sample in dataset[split])
        for split in dataset
    )
    total_samples = sum(len(dataset[split]) for split in dataset)

    logger.info(f"\n{'='*50}")
    logger.info("FINAL TOTALS")
    logger.info(f"{'='*50}")
    logger.info(f"  Total samples: {total_samples:,}")
    logger.info(f"  Total tokens: {total_tokens:,}")
    logger.info(f"  ~ Parameters: {total_tokens * 2 / 1e9:.2f}B tokens (approx training data)")

    # Save aggregate stats
    agg_stats = {
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "estimated_training_data_gb": round(total_tokens * 4 / 1e9, 2),  # ~4 bytes per token (int32)
    }
    with open(data_path / "aggregate_stats.json", "w") as f:
        json.dump(agg_stats, f, indent=2)

    return agg_stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="./data/processed")
    args = parser.parse_args()
    compute_stats(args.dataset)
