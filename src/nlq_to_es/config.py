import os

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
    "gpu": 1
}

# =========================
# Dataset + paths
# =========================
DATA_CONFIG = {
    "train": {
        "basic": "data/dataset/queries/train/basic_query.jsonl",
        "agg": "data/dataset/queries/train/agg_query.jsonl",
        "knn": "data/dataset/queries/train/knn_query.jsonl",
        "tables": "data/dataset/tables/train.jsonl"
    },
    "validation": {
        "basic": "data/dataset/queries/validation/basic_query.jsonl",
        "agg": "data/dataset/queries/validation/agg_query.jsonl",
        "knn": "data/dataset/queries/validation/knn_query.jsonl",
        "tables": "data/dataset/tables/validation.jsonl"
    },
    "test": {
        "basic": "data/dataset/queries/test/basic_query.jsonl",
        "agg": "data/dataset/queries/test/agg_query.jsonl",
        "knn": "data/dataset/queries/test/knn_query.jsonl",
        "tables": "data/dataset/tables/test.jsonl"
    }
}


HF_DATASET_CONFIG = {
    "dataset_name": "ayselyagmur/dataset",
    "cache_dir": "cache/huggingface",
    "local_output_dir": "data/dataset",
}

TABLE_UPLOAD_CONFIG = {
    "jsonl_file": "data/dataset/tables/test.jsonl"
}
