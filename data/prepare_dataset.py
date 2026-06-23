"""
Nusantara LLM — Dataset Preparation Pipeline
Downloads, filters, tokenizes, and shards the Nusantara Corpus.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import datasets
import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


LANGUAGE_CONFIGS = {
    "indonesian": {
        "sources": [
            ("oscar-corpus/OSCAR-2301", "id", 12_000_000_000),
            ("common_crawl", "id", 8_000_000_000),
            ("wikipedia", "20231101.id", 1_000_000_000),
        ],
        "max_samples": 5_000_000,
    },
    "javanese": {
        "sources": [
            ("wikipedia", "20231101.jv", 200_000_000),
        ],
        "max_samples": 500_000,
    },
    "sundanese": {
        "sources": [
            ("wikipedia", "20231101.su", 100_000_000),
        ],
        "max_samples": 200_000,
    },
    "instruct": {
        "sources": None,  # Generated synthetically
        "max_samples": 5_000_000,  # Instruction pairs
    },
}


def quality_filter(text: str, min_length: int = 50, max_length: int = 8192) -> bool:
    """Filter low-quality text samples."""
    if len(text) < min_length or len(text) > max_length * 4:
        return False
    # Reject if too much repetitive content
    if len(set(text)) / max(len(text), 1) < 0.3:
        return False
    # Reject if too many non-printable characters
    printable_ratio = sum(c.isprintable() for c in text) / max(len(text), 1)
    if printable_ratio < 0.9:
        return False
    return True


def deduplicate(dataset: Dataset, text_column: str = "text") -> Dataset:
    """Remove exact duplicate texts."""
    before = len(dataset)
    dataset = dataset.unique(text_column) if hasattr(dataset, "unique") else dataset
    # Use hash-based dedup
    seen = set()
    deduped_indices = []
    for i, example in enumerate(dataset):
        text_hash = hash(example[text_column][:1000])  # Hash first 1000 chars
        if text_hash not in seen:
            seen.add(text_hash)
            deduped_indices.append(i)
    after = len(deduped_indices)
    logger.info(f"Deduplication: {before} → {after} samples ({before - after} removed)")
    return dataset.select(deduped_indices)


def prepare_dataset(
    output_dir: str,
    tokenizer_name: str = "meta-llama/Llama-3.1-70B",
    max_length: int = 32768,
    max_samples_per_lang: Optional[int] = None,
):
    """Main dataset preparation pipeline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_datasets = []

    # Process each language
    for lang, config in LANGUAGE_CONFIGS.items():
        if config["sources"] is None:
            logger.info(f"Skipping synthetic source: {lang}")
            continue

        logger.info(f"Processing language: {lang}")
        lang_samples = []

        for source_name, source_config, expected_size in config["sources"]:
            try:
                logger.info(f"  Loading {source_name} ({source_config})...")
                if "oscar" in source_name:
                    ds = load_dataset(source_name, "unshuffled_deduplicated_id", split="train", streaming=False)
                elif source_name == "common_crawl":
                    # Use CC-100 or similar
                    ds = load_dataset("common_crawl", source_config, split="train", streaming=False)
                elif "wikipedia" in source_name:
                    ds = load_dataset("wikipedia", source_config, split="train", streaming=False)
                else:
                    continue

                # Quality filter
                ds = ds.filter(lambda x: quality_filter(x.get("text", "")), num_proc=16)
                lang_samples.append(ds)
                logger.info(f"    → {len(ds):,} samples after filtering")

            except Exception as e:
                logger.warning(f"  Failed to load {source_name}: {e}")
                # Create synthetic replacement
                logger.info(f"  Generating synthetic replacement for {source_name}...")
                synthetic_data = generate_synthetic_id_data(
                    num_samples=100_000,
                    lang=lang,
                )
                lang_samples.append(synthetic_data)

        if lang_samples:
            combined = concatenate_datasets(lang_samples)
            combined = deduplicate(combined)

            if max_samples_per_lang:
                combined = combined.select(range(min(len(combined), max_samples_per_lang)))

            all_datasets.append(combined)

    if not all_datasets:
        logger.warning("No real datasets loaded. Generating synthetic training data.")
        synthetic = generate_synthetic_id_data(200_000, "indonesian")
        all_datasets.append(synthetic)

    # Combine all languages
    logger.info("Combining all language datasets...")
    full_dataset = concatenate_datasets(all_datasets)
    logger.info(f"Total samples before tokenization: {len(full_dataset):,}")

    # Shuffle
    full_dataset = full_dataset.shuffle(seed=42)

    # Tokenize
    logger.info(f"Tokenizing with max_length={max_length}...")

    def tokenize_function(examples):
        texts = examples.get("text", examples.get("content", [""] * len(examples)))
        return tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_attention_mask=False,
        )

    tokenized = full_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=16,
        remove_columns=full_dataset.column_names,
        desc="Tokenizing",
    )

    # Split
    split = tokenized.train_test_split(test_size=0.01, seed=42)
    test_val = split["test"].train_test_split(test_size=0.5, seed=42)

    final_dataset = DatasetDict({
        "train": split["train"],
        "validation": test_val["train"],
        "test": test_val["test"],
    })

    # Save
    logger.info(f"Saving dataset to {output_path}...")
    final_dataset.save_to_disk(str(output_path))

    # Write stats
    stats = {
        "total_tokens": sum(len(ds["input_ids"]) for ds in final_dataset.values()),
        "train_samples": len(final_dataset["train"]),
        "val_samples": len(final_dataset["validation"]),
        "test_samples": len(final_dataset["test"]),
        "vocab_size": tokenizer.vocab_size,
        "max_length": max_length,
    }
    with open(output_path / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    for split_name, ds in final_dataset.items():
        logger.info(f"  {split_name}: {len(ds):,} samples")

    logger.info("Dataset preparation complete!")
    return final_dataset


def generate_synthetic_id_data(num_samples: int, lang: str = "indonesian") -> Dataset:
    """Generate synthetic Indonesian language data for testing the pipeline."""
    templates = [
        "Dalam perkembangan {topic} di Indonesia, {subject} memiliki peran penting dalam {context}.",
        "Menurut penelitian terbaru, {subject} di {location} menunjukkan peningkatan {metric} sebesar {percent}%.",
        "Pemerintah Indonesia melalui {institution} telah mengumumkan kebijakan baru terkait {topic}.",
        "{subject} merupakan salah satu aspek penting dalam pembangunan {sector} di era digital.",
        "Para ahli sepakat bahwa {topic} di Indonesia memerlukan perhatian khusus dari {institution}.",
        "Data terbaru menunjukkan bahwa {percent}% masyarakat Indonesia menggunakan {technology} dalam kehidupan sehari-hari.",
        "Sejarah mencatat bahwa {subject} telah berkembang signifikan sejak {year} di Indonesia.",
        "Inovasi dalam bidang {sector} terus didorong oleh {institution} untuk mencapai target {metric}.",
        "Masyarakat {location} dikenal dengan kearifan lokalnya dalam mengelola {topic} secara berkelanjutan.",
        "Perkembangan {technology} telah membawa dampak positif terhadap {sector} di Indonesia.",
    ]

    topics = ["pendidikan", "teknologi", "kesehatan", "ekonomi", "budaya", "pertanian", "maritim", "pariwisata"]
    subjects = ["masyarakat lokal", "generasi muda", "UMKM", "startup teknologi", "komunitas seni"]
    locations = ["Jakarta", "Yogyakarta", "Surabaya", "Bandung", "Bali", "Makassar", "Medan"]
    metrics = ["produktivitas", "efisiensi", "kualitas", "partisipasi", "pertumbuhan"]
    sectors = ["pendidikan", "kesehatan", "ekonomi digital", "pertanian berkelanjutan", "pariwisata"]
    institutions = ["Kementerian Pendidikan", "Kementerian Kesehatan", "Kemendikbudristek", "BRIN", "Bappenas"]
    technologies = ["AI", "cloud computing", "IoT", "blockchain", "5G", "big data"]
    years = list(range(2020, 2026))

    texts = []
    for _ in range(num_samples):
        text = random.choice(templates).format(
            topic=random.choice(topics),
            subject=random.choice(subjects),
            context=random.choice(sectors),
            location=random.choice(locations),
            metric=random.choice(metrics),
            percent=random.randint(5, 95),
            institution=random.choice(institutions),
            sector=random.choice(sectors),
            technology=random.choice(technologies),
            year=random.choice(years),
        )
        texts.append(text)

    return Dataset.from_dict({"text": texts})


def main():
    parser = argparse.ArgumentParser(description="Prepare Nusantara LLM training dataset")
    parser.add_argument("--output", type=str, default="./data/processed",
                        help="Output directory for processed dataset")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-70B",
                        help="Tokenizer name/path")
    parser.add_argument("--max-length", type=int, default=32768,
                        help="Maximum sequence length")
    parser.add_argument("--max-samples", type=int, default=500000,
                        help="Max samples per language")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Generate only synthetic data (for testing)")
    args = parser.parse_args()

    if args.synthetic_only:
        logger.info("Synthetic-only mode enabled")
        from datasets import DatasetDict
        synthetic = generate_synthetic_id_data(args.max_samples)
        split = synthetic.train_test_split(test_size=0.01, seed=42)
        test_val = split["test"].train_test_split(test_size=0.5, seed=42)
        dataset = DatasetDict({
            "train": split["train"],
            "validation": test_val["train"],
            "test": test_val["test"],
        })
        save_path = Path(args.output)
        save_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(save_path))
        logger.info(f"Synthetic dataset saved to {save_path}")
        return

    prepare_dataset(
        output_dir=args.output,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        max_samples_per_lang=args.max_samples,
    )


if __name__ == "__main__":
    main()
