"""
Nusantara LLM — Tokenization Pipeline
Converts raw text to tokenized IDs with packing for efficient training.
"""

import argparse
import json
import logging
import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def pack_sequences(
    token_ids: List[int],
    max_length: int,
    eos_token_id: int,
    pad_token_id: int = 0,
) -> Dataset:
    """
    Pack multiple short sequences into fixed-length chunks for efficiency.
    Each chunk is padded to max_length with EOS tokens as separators.
    """
    chunks = []
    current_chunk = []

    for tid in token_ids:
        current_chunk.append(tid)
        if tid == eos_token_id:
            # Sequence boundary — check if adding more would overflow
            if len(current_chunk) >= max_length:
                # Pad current chunk
                if len(current_chunk) < max_length:
                    current_chunk.extend([pad_token_id] * (max_length - len(current_chunk)))
                chunks.append({"input_ids": current_chunk[:max_length]})
                current_chunk = []

    # Handle the last chunk
    if current_chunk:
        if len(current_chunk) < max_length:
            current_chunk.extend([pad_token_id] * (max_length - len(current_chunk)))
        chunks.append({"input_ids": current_chunk[:max_length]})

    return Dataset.from_list(chunks)


def tokenize_dataset(
    input_path: str,
    output_path: str,
    tokenizer_name: str = "meta-llama/Llama-3.1-70B",
    max_length: int = 32768,
    pack: bool = True,
    num_proc: int = 16,
):
    """Tokenize a text dataset with optional sequence packing."""
    input_dir = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loading dataset from {input_path}")
    dataset = load_from_disk(str(input_dir))

    def _tokenize_batch(examples, text_column="text"):
        texts = examples.get(text_column, [""] * len(examples.get("text", [None])))
        return tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_attention_mask=False,
        )

    tokenized_datasets = {}
    for split_name in dataset:
        logger.info(f"Tokenizing split: {split_name}")
        ds = dataset[split_name]

        tokenized = ds.map(
            _tokenize_batch,
            batched=True,
            num_proc=num_proc,
            remove_columns=ds.column_names,
            desc=f"Tokenizing {split_name}",
        )

        if pack:
            logger.info(f"  Packing sequences for {split_name}...")
            # Flatten all token IDs
            all_ids = []
            for example in tokenized:
                all_ids.extend(example["input_ids"])
                all_ids.append(tokenizer.eos_token_id)

            tokenized = pack_sequences(
                all_ids,
                max_length=max_length,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            logger.info(f"  After packing: {len(tokenized):,} sequences")

        tokenized_datasets[split_name] = tokenized

    final = DatasetDict(tokenized_datasets)

    # Save
    logger.info(f"Saving tokenized dataset to {output_dir}")
    final.save_to_disk(str(output_dir))

    # Stats
    for split_name, ds in final.items():
        lengths = [len(s["input_ids"]) for s in ds]
        logger.info(f"  {split_name}: {len(ds):,} sequences, "
                     f"mean length={np.mean(lengths):.1f}")

    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize Nusantara LLM dataset")
    parser.add_argument("--input", type=str, default="./data/processed",
                        help="Input dataset directory")
    parser.add_argument("--output", type=str, default="./data/tokenized",
                        help="Output directory for tokenized dataset")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-70B",
                        help="Tokenizer name/path")
    parser.add_argument("--max-length", type=int, default=32768,
                        help="Maximum sequence length")
    parser.add_argument("--no-pack", action="store_true",
                        help="Disable sequence packing")
    args = parser.parse_args()

    tokenize_dataset(
        input_path=args.input,
        output_path=args.output,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        pack=not args.no_pack,
    )
