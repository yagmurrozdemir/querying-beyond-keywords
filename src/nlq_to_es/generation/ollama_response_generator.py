#!/usr/bin/env python3
import re
from openai import OpenAI

from nlq_to_es.config import BACKEND_CONFIG, MODEL_CONFIG


def remove_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def get_ollama_response(prompt_text: str, setting: str) -> str:
    model_cfg = MODEL_CONFIG[setting]
    backend = model_cfg["backend"]

    if backend == "ollama":
        backend_cfg = BACKEND_CONFIG["ollama"]

        client = OpenAI(
            base_url=backend_cfg["api_base"],
            api_key=backend_cfg["api_key"],
        )

        response = client.chat.completions.create(
            model=model_cfg["model_name"],
            messages=[{"role": "user", "content": prompt_text}],
            temperature=model_cfg["temperature"],
        )

        text = response.choices[0].message.content
        return remove_think_blocks(text)

    elif backend == "huggingface":
        raise NotImplementedError(
            "Hugging Face fine-tuned inference is not added yet in model_response_generator.py"
        )

    else:
        raise ValueError(f"Unsupported backend: {backend}")
