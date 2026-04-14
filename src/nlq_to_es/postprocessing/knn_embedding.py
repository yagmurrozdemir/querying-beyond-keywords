import requests
from nlq_to_es.config import EMBEDDING_CONFIG

def extract_text_from_output(raw_output: str):
    if "~" not in raw_output:
        return None
    _, text = raw_output.split("~", 1)
    return text.strip()


def get_embedding(embedding_model: str, text: str):
    if embedding_model == "clip":
        response = requests.post(
            EMBEDDING_CONFIG["clip_url"],
            json={"text": text},
            headers={"Content-Type": "application/json"}
        )
        return response.json()["embedding"][0]

    elif embedding_model == "qwen":
        response = requests.post(
            EMBEDDING_CONFIG["qwen_url"],
            json={"model": "dengcao/Qwen3-Embedding-8B:Q4_K_M", "prompt": text},
            headers={"Content-Type": "application/json"}
        )
        return response.json()["embedding"]

    else:
        raise ValueError(f"Unsupported embedding model: {embedding_model}")
