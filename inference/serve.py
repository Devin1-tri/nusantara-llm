"""
Nusantara LLM — Inference Server (vLLM-compatible)
Serves the fine-tuned Nusantara model via OpenAI-compatible API.
"""

import argparse
import logging
import os
from typing import AsyncGenerator, Dict, List, Optional

import torch
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Optional: vLLM integration
try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logger.warning("vLLM not installed. Install with: pip install vllm")


class NusantaraInference:
    """
    Inference wrapper for Nusantara LLM.
    Supports vLLM backend (preferred) and raw HF Transformers.
    """

    def __init__(
        self,
        model_path: str,
        use_vllm: bool = True,
        tensor_parallel_size: int = 1,
        max_num_seqs: int = 64,
        max_model_len: int = 32768,
        gpu_memory_utilization: float = 0.90,
        dtype: str = "bfloat16",
    ):
        self.model_path = model_path
        self.use_vllm = use_vllm and VLLM_AVAILABLE

        if self.use_vllm:
            logger.info(f"Initializing vLLM with {model_path}")
            engine_args = AsyncEngineArgs(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                dtype=dtype,
                trust_remote_code=True,
                enforce_eager=False,
            )
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("vLLM engine initialized")
        else:
            logger.info(f"Initializing HF Transformers with {model_path}")
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
            self.model.eval()
            logger.info("HF model loaded")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> str:
        """Generate text from a prompt."""
        if self.use_vllm:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
                stop_token_ids=[self.tokenizer.eos_token_id],
            )
            request_id = f"nusantara-{os.urandom(4).hex()}"

            full_text = ""
            async for result in self.engine.generate(prompt, sampling_params, request_id):
                full_text = result.outputs[0].text

            return full_text.strip()
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            return response.strip()

    def format_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Format chat messages for the model."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        # Fallback for models without chat template
        formatted = ""
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg["content"]
            formatted += f"<|{role}|>\n{content}\n"
        formatted += "<|Assistant|>\n"
        return formatted


def start_server(args):
    """Start vLLM OpenAI-compatible API server."""
    if not VLLM_AVAILABLE:
        logger.error("vLLM is required for server mode. Install: pip install vllm")
        return

    logger.info(f"Starting Nusantara LLM API server...")
    logger.info(f"  Model: {args.model_path}")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Tensor parallel: {args.tensor_parallel_size}")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices or "0"

    import uvicorn
    uvicorn.run(
        "vllm.entrypoints.openai.api_server:app",
        host="0.0.0.0",
        port=args.port,
        log_level="info",
    )


def main():
    parser = argparse.ArgumentParser(description="Nusantara LLM Inference")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["interactive", "server", "one-shot"],
                        default="interactive")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cuda-devices", type=str, default=None)
    parser.add_argument("--prompt", type=str, default=None, help="One-shot prompt")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if args.mode == "server":
        start_server(args)
        return

    infer = NusantaraInference(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    if args.mode == "one-shot" and args.prompt:
        response = infer.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"\n{'='*60}")
        print(f"Prompt: {args.prompt[:100]}...")
        print(f"{'='*60}")
        print(f"Response: {response}")
        return

    # Interactive mode
    print("\nNusantara LLM — Interactive Mode")
    print("Type 'quit' to exit, 'clear' to clear context.")
    print("=" * 60)

    context = []
    while True:
        prompt = input("\n>> ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if prompt.lower() == "clear":
            context = []
            print("Context cleared.")
            continue

        messages = [{"role": "user", "content": prompt}]
        formatted = infer.format_chat_prompt(messages)

        response = infer.generate(formatted, max_tokens=args.max_tokens)
        print(f"\n🤖 {response}")


if __name__ == "__main__":
    main()
