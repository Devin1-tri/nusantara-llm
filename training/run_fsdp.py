"""
Nusantara LLM — Main FSDP Training Script
Full-parameter fine-tuning of Llama 3.1 70B on Nusantara Corpus.
Optimized for NVIDIA H200 (141 GB HBM3e).
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    default_data_collator,
)

logger = get_logger(__name__)


@dataclass
class TrainingArgs:
    """Training configuration."""
    model_name: str = "meta-llama/Llama-3.1-70B"
    dataset_path: str = "./data/processed"
    output_dir: str = "./checkpoints"
    run_name: str = "nusantara-llm-70b"
    seed: int = 42
    num_epochs: int = 1
    max_steps: int = 10000
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    save_total_limit: int = 5
    bf16: bool = True
    tf32: bool = True
    flash_attn: bool = True
    gradient_checkpointing: bool = True
    max_length: int = 32768


def setup_training(args: TrainingArgs) -> Tuple[Accelerator, Dict]:
    """Initialize training components."""
    # Set precision
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Set seed
    set_seed(args.seed)

    # Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16" if args.bf16 else "no",
        log_with="wandb",
        project_dir=args.output_dir,
    )

    # Logging
    accelerator.init_trackers(
        project_name="nusantara-llm",
        config=vars(args),
        init_kwargs={"wandb": {"name": args.run_name}},
    )

    return accelerator, {}


def load_model_and_tokenizer(args: TrainingArgs, accelerator: Accelerator):
    """Load model and tokenizer with H200-optimized settings."""
    logger.info(f"Loading model: {args.model_name}")

    # Model config
    config = AutoConfig.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    config.use_cache = False  # Required for gradient checkpointing

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine attention implementation
    attn_implementation = "flash_attention_2" if args.flash_attn else "sdpa"

    # Load model with memory-efficient settings
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Log model size
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {param_count / 1e9:.2f}B")
    logger.info(f"Trainable parameters: {trainable_params / 1e9:.2f}B")
    logger.info(f"Memory required (BF16): {param_count * 2 / 1e9:.1f} GB")

    return model, tokenizer


def load_dataset(args: TrainingArgs, tokenizer) -> Tuple[DataLoader, DataLoader]:
    """Load and prepare dataset."""
    logger.info(f"Loading dataset from {args.dataset_path}")
    dataset = load_from_disk(args.dataset_path)

    def collate_fn(batch):
        input_ids = torch.stack([torch.tensor(b["input_ids"], dtype=torch.long) for b in batch])
        # For causal LM, labels == input_ids
        labels = input_ids.clone()
        # Mask padding tokens (if pad_token_id is set)
        if tokenizer.pad_token_id is not None:
            labels[labels == tokenizer.pad_token_id] = -100
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": (input_ids != tokenizer.pad_token_id).long(),
        }

    train_loader = DataLoader(
        dataset["train"],
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    eval_loader = DataLoader(
        dataset.get("validation", dataset["train"].select(range(min(500, len(dataset["train"]))))),
        batch_size=args.per_device_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    ) if "validation" in dataset else None

    return train_loader, eval_loader


def get_optimizer_and_scheduler(
    model, args: TrainingArgs, accelerator: Accelerator, num_training_steps: int
):
    """Create optimizer and learning rate scheduler."""
    # Prepare optimizer
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )

    return optimizer, scheduler


def evaluate(
    model, eval_loader, accelerator, args, step: int
) -> Dict[str, float]:
    """Run evaluation on validation set."""
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in eval_loader:
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            total_steps += 1

    avg_loss = total_loss / max(total_steps, 1)
    perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow

    metrics = {
        "eval_loss": avg_loss.item(),
        "eval_perplexity": perplexity,
        "eval_step": step,
    }

    if accelerator.is_main_process:
        logger.info(f"Step {step} | Eval Loss: {avg_loss:.4f} | Perplexity: {perplexity:.2f}")

    model.train()
    return metrics


def train(args: TrainingArgs):
    """Main training loop."""
    accelerator, config = setup_training(args)
    model, tokenizer = load_model_and_tokenizer(args, accelerator)
    train_loader, eval_loader = load_dataset(args, tokenizer)

    # Compute training steps
    num_update_steps_per_epoch = len(train_loader) // args.gradient_accumulation_steps
    num_training_steps = min(
        args.max_steps,
        num_update_steps_per_epoch * args.num_epochs,
    )

    optimizer, scheduler = get_optimizer_and_scheduler(
        model, args, accelerator, num_training_steps
    )

    # Prepare with Accelerator
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # Training loop
    global_step = 0
    train_loss = 0.0
    best_loss = float("inf")
    start_time = time.time()

    logger.info("===== Starting Training =====")
    logger.info(f"  Total training steps: {num_training_steps}")
    logger.info(f"  Batch size (per device): {args.per_device_batch_size}")
    logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {args.per_device_batch_size * args.gradient_accumulation_steps}")
    logger.info(f"  Precision: {'BF16' if args.bf16 else 'FP32'}")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Device: {accelerator.device}")

    progress_bar = tqdm(
        range(num_training_steps),
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )

    for epoch in range(args.num_epochs):
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                train_loss += loss.detach().float()

                # Logging
                if global_step % args.logging_steps == 0:
                    avg_loss = train_loss / args.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    samples_per_sec = (
                        global_step * args.per_device_batch_size * args.gradient_accumulation_steps / elapsed
                    )

                    accelerator.log(
                        {
                            "loss": avg_loss.item(),
                            "lr": lr,
                            "epoch": epoch,
                            "samples/sec": samples_per_sec,
                            "global_step": global_step,
                        },
                        step=global_step,
                    )

                    if accelerator.is_main_process:
                        logger.info(
                            f"Step {global_step}/{num_training_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {lr:.2e} | "
                            f"Speed: {samples_per_sec:.1f} samples/s"
                        )
                    train_loss = 0.0

                # Evaluation
                if global_step % args.eval_steps == 0 and eval_loader is not None:
                    eval_metrics = evaluate(model, eval_loader, accelerator, args, global_step)
                    accelerator.log(eval_metrics, step=global_step)

                    if eval_metrics["eval_loss"] < best_loss:
                        best_loss = eval_metrics["eval_loss"]
                        if accelerator.is_main_process:
                            logger.info(f"New best loss: {best_loss:.4f}")

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                        accelerator.unwrap_model(model).save_pretrained(
                            save_path,
                            safe_serialization=True,
                        )
                        tokenizer.save_pretrained(save_path)
                        logger.info(f"Checkpoint saved: {save_path}")

                        # Clean old checkpoints
                        all_checkpoints = sorted(Path(args.output_dir).glob("checkpoint-*"))
                        while len(all_checkpoints) > args.save_total_limit:
                            oldest = all_checkpoints.pop(0)
                            import shutil
                            shutil.rmtree(oldest)
                            logger.info(f"Removed old checkpoint: {oldest}")

                progress_bar.update(1)

                if global_step >= num_training_steps:
                    break

        if global_step >= num_training_steps:
            break

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = Path(args.output_dir) / "final"
        accelerator.unwrap_model(model).save_pretrained(final_path, safe_serialization=True)
        tokenizer.save_pretrained(final_path)
        logger.info(f"Final model saved: {final_path}")

        # Training summary
        total_time = time.time() - start_time
        summary = {
            "model": args.model_name,
            "total_steps": global_step,
            "total_time_hours": total_time / 3600,
            "best_eval_loss": best_loss if best_loss != float("inf") else None,
            "hardware": "NVIDIA H200 (141 GB)",
            "run_name": args.run_name,
        }
        with open(Path(args.output_dir) / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{'='*50}")
        logger.info("TRAINING COMPLETE")
        logger.info(f"{'='*50}")
        logger.info(f"  Total steps: {global_step}")
        logger.info(f"  Total time: {total_time / 3600:.2f} hours")
        logger.info(f"  Model saved to: {final_path}")

    accelerator.end_training()


def main():
    parser = argparse.ArgumentParser(description="Nusantara LLM — FSDP Training")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-70B")
    parser.add_argument("--dataset", type=str, default="./data/processed")
    parser.add_argument("--output", type=str, default="./checkpoints")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--run-name", type=str, default="nusantara-llm-70b")
    parser.add_argument("--no-flash-attn", action="store_true")
    args = parser.parse_args()

    training_args = TrainingArgs(
        model_name=args.model,
        dataset_path=args.dataset,
        output_dir=args.output,
        run_name=args.run_name,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=args.warmup,
        save_steps=args.save_steps,
        flash_attn=not args.no_flash_attn,
    )

    train(training_args)


if __name__ == "__main__":
    main()
