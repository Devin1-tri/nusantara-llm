"""
Nusantara LLM — API Client Example
Demonstrates how to interact with the inference server.
"""

import argparse
import json
import requests


def query_server(
    prompt: str,
    server_url: str = "http://localhost:8000",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    stream: bool = False,
) -> str:
    """Query the Nusantara LLM server via completion endpoint."""
    payload = {
        "model": "nusantara-llm",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }

    if stream:
        response = requests.post(
            f"{server_url}/v1/completions",
            json=payload,
            stream=True,
        )
        full_text = ""
        for line in response.iter_lines():
            if line:
                data = line.decode("utf-8").removeprefix("data: ").strip()
                if data and data != "[DONE]":
                    chunk = json.loads(data)
                    text = chunk.get("choices", [{}])[0].get("text", "")
                    print(text, end="", flush=True)
                    full_text += text
        print()
        return full_text
    else:
        response = requests.post(
            f"{server_url}/v1/completions",
            json=payload,
        )
        result = response.json()
        return result.get("choices", [{}])[0].get("text", "").strip()


def chat(
    messages: list,
    server_url: str = "http://localhost:8000",
) -> str:
    """Use the chat completion endpoint."""
    payload = {
        "model": "nusantara-llm",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    response = requests.post(
        f"{server_url}/v1/chat/completions",
        json=payload,
    )
    result = response.json()
    return result.get("choices", [{}])[0].get("message", {}).get("content", "")


def main():
    parser = argparse.ArgumentParser(description="Nusantara LLM API Client")
    parser.add_argument("--server", type=str, default="http://localhost:8000")
    parser.add_argument("--mode", choices=["completion", "chat"], default="completion")
    parser.add_argument("--prompt", type=str,
                        default="Explain artificial intelligence in Indonesian language.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    if args.mode == "chat":
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant for Indonesian users."},
            {"role": "user", "content": args.prompt},
        ]
        response = chat(messages, args.server)
    else:
        response = query_server(
            args.prompt,
            args.server,
            args.max_tokens,
            stream=args.stream,
        )

    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    main()
