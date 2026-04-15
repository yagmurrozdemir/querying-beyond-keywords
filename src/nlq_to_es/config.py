import os
from pathlib import Path

# =========================
# Project root
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =========================
# Common directories
# =========================
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
QUERIES_DIR = DATASET_DIR / "queries"
TABLES_DIR = DATASET_DIR / "tables"
FT_PROMPTS_DIR = DATA_DIR / "finetuning_prompts"
OUTPUTS_DIR = DATA_DIR / "outputs"
CACHE_DIR = PROJECT_ROOT / "cache"
PREDICTIONS_DIR = DATA_DIR / "predictions"
RESULTS_DIR = DATA_DIR / "results"


# =========================
# Elasticsearch
# =========================

ELASTIC_CONFIG = {
    "host": os.getenv("ES_HOST", "https://localhost:9200"),
    "username": os.getenv("ES_USERNAME", "elastic"),
    "password": os.getenv("ES_PASSWORD", ""),
    "verify_certs": os.getenv("ES_VERIFY_CERTS", "false").lower() == "true",
}

# =========================
# OpenAI (optional / placeholder)
# =========================
GPT_CONFIG = {
    "api_key": os.getenv("OPENAI_API_KEY", "YOUR_API_KEY_HERE")
}

# =========================
# Embedding services
# =========================
EMBEDDING_CONFIG = {
    "clip_url": "http://127.0.0.1:8000/text-embedding/",
    "qwen_url": "http://127.0.0.1:11434/api/embeddings"
}

# =========================
# Backends
# =========================
BACKEND_CONFIG = {
    "ollama": {
        "api_base": "http://localhost:11434/v1",
        "api_key": "ollama"
    },
    "huggingface": {
        "device": "cuda"
    }
}

# =========================
# Model configs
# =========================
MODEL_CONFIG = {
    "zero_shot": {
        "backend": "ollama",
        "model_name": "qwen2.5-coder:14b-instruct-q4_K_M",
        "temperature": 0.7
    },
    "few_shot": {
        "backend": "ollama",
        "model_name": "qwen2.5-coder:14b-instruct-q4_K_M",
        "temperature": 0.7
    },
    "finetuned": {
        "backend": "huggingface",
        "base_model_name": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 512,
        "adapters": {
            "basic": "ayselyagmur/Qwen2.5-Coder-32B-Instruct-esdsl-basic",
            "agg": "ayselyagmur/Qwen2.5-Coder-32B-Instruct-esdsl-agg",
            "knn": "ayselyagmur/Qwen2.5-Coder-32B-Instruct-esdsl-knn",
            "mixed": "ayselyagmur/Qwen2.5-Coder-32B-Instruct-esdsl-mixed"
        }
    }
}

# =========================
# Training config (shared)
# =========================
TRAINING_CONFIG = {
    "model_name_or_path": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "max_length": 2048,
    "per_device_train_batch_size": 1,
    "per_device_eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "num_train_epochs": 1,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "logging_steps": 10,
    "eval_steps": 200,
    "save_steps": 200,
    "save_total_limit": 2,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "gpu": 1,
    "output_path": {
        "basic": OUTPUTS_DIR / "basic_adapter",
        "agg": OUTPUTS_DIR / "agg_adapter",
        "knn": OUTPUTS_DIR / "knn_adapter",
        "mixed": OUTPUTS_DIR / "mixed_adapter"
    }
}

SAMPLING_CONFIG = {
    "split": "train",
    "alpha": 0.4,
    "seed": 42,
    "size": 0
}
# =========================
# Dataset + paths
# =========================
DATA_CONFIG = {
    "train": {
        "basic": QUERIES_DIR / "train/basic_query.jsonl",
        "agg": QUERIES_DIR / "train/agg_query.jsonl",
        "knn": QUERIES_DIR / "train/knn_query.jsonl",
        "tables": TABLES_DIR / "train.jsonl"
    },
    "validation": {
        "basic": QUERIES_DIR / "validation/basic_query.jsonl",
        "agg": QUERIES_DIR / "validation/agg_query.jsonl",
        "knn": QUERIES_DIR / "validation/knn_query.jsonl",
        "tables": TABLES_DIR / "validation.jsonl"
    },
    "test": {
        "basic": QUERIES_DIR / "test/basic_query.jsonl",
        "agg": QUERIES_DIR / "test/agg_query.jsonl",
        "knn": QUERIES_DIR / "test/knn_query.jsonl",
        "tables": TABLES_DIR / "test.jsonl"
    }
}

FT_PROMPT_CONFIG = {
    "train": {
        "basic": FT_PROMPTS_DIR / "train/basic.jsonl",
        "agg": FT_PROMPTS_DIR / "train/agg.jsonl",
        "knn": FT_PROMPTS_DIR / "train/knn.jsonl",
        "mixed": FT_PROMPTS_DIR / "train/mixed.jsonl"
    },
    "validation": {
        "basic": FT_PROMPTS_DIR / "validation/basic.jsonl",
        "agg": FT_PROMPTS_DIR / "validation/agg.jsonl",
        "knn": FT_PROMPTS_DIR / "validation/knn.jsonl",
        "mixed": FT_PROMPTS_DIR / "validation/mixed.jsonl"
    },
    "test": {
        "basic": FT_PROMPTS_DIR / "test/basic.jsonl",
        "agg": FT_PROMPTS_DIR / "test/agg.jsonl",
        "knn": FT_PROMPTS_DIR / "test/knn.jsonl",
        "mixed": FT_PROMPTS_DIR / "test/mixed.jsonl"
    }
}

HF_DATASET_CONFIG = {
    "dataset_name": "ayselyagmur/dataset",
    "cache_dir": CACHE_DIR / "huggingface",
    "local_output_dir": DATASET_DIR,
}

TABLE_UPLOAD_CONFIG = {
    "jsonl_file": TABLES_DIR / "test.jsonl"
}
