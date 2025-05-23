# 🔍 Querying Beyond Keywords: NL-to-Elasticsearch DSL with Vector Search

This repository contains the **code, data, and pipeline** to reproduce the experiments presented in our paper:

**📝 Title**: _Querying Beyond Keywords: Translating Natural Language to Elasticsearch DSL with Vector Search_  
**📍 Submitted to**: NeurIPS 2025 (under review)  

---

## 🧠 Project Overview

We introduce the **first benchmark and pipeline** for translating natural language queries (NLQs) into **hybrid Elasticsearch DSL queries** that combine:

- ✅ **Keyword-based filters**
- ✅ **Vector-based semantic similarity via CLIP**
- ✅ **Structured aggregation & filtering**
- 🧠 Powered by **Large Language Models (LLMs)** via [Ollama](https://ollama.com/)

This project supports end-to-end training, evaluation, and reproducibility of results for:

- ✅ Generating DSL queries from NLQs using models like `qwen2.5-coder`
- ✅ Executing and comparing those queries on enriched tables
- ✅ Evaluating structural correctness and execution accuracy

---

## 📦 Repository Structure

```text
project/
├── pipeline/
│   ├── full_pipeline_runner.py
│   ├── generate_query_from_nlq.py
│   ├── export_index_mapping.py
│   ├── generate_embedding.py
│   ├── inject_embedding_into_query.py
│   ├── execute_query.py
│   ├── run_correct_query.py
│   ├── compare_results.py
│   ├── inputs/
│   │   ├── prompt.txt
│   │   ├── headers.csv
│   │   └── types.csv
│   ├── data/
│   │   └── TEST_SET/
│   │       ├── agg_query.jsonl
│   │       ├── knn_query.jsonl
│   │       └── other_query.jsonl
│   ├── outputs/
│   ├── requirements.txt
│   └── Dockerfile
├── clip-api/              # CLIP FastAPI service
├── uploader/              # Table uploader to Elasticsearch
├── docker-compose.yml
└── 1966_Querying_Beyond_Keywords_.pdf  # Paper
```

---

## 📥 Setup Instructions

### 1. 🧠 Ollama (for LLM-based query generation)

Install [Ollama](https://ollama.com) and pull the model:

```bash
curl https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:14b
```

Start Ollama in **OpenAI-compatible mode**:

```bash
OLLAMA_OPENAI_COMPAT=1 ollama serve
```

> ✅ This enables `/v1/chat/completions` and is required for `openai.ChatCompletion.create(...)`.

---

### 2. 🐳 Build the Project

From the root:

```bash
docker compose down
docker compose up --build
```

This launches:
- Elasticsearch (8.15.0)
- CLIP embedding service
- Uploader + Pipeline

---

## ⚙️ How the Pipeline Works

Each NLQ is converted and evaluated through these stages:

1. **Mapping**: export index schema from Elasticsearch
2. **Generation**: LLM (Qwen) generates DSL from prompt
3. **Embedding**: CLIP API vectorizes semantic fields
4. **Injection**: DSL is updated with dense vector
5. **Execution**: query is run on the table index
6. **Post-process**: result is compared to gold label

Run it manually via:

```bash
docker compose run pipeline
```

---

## 📊 Reproducing Paper Results

The paper reports performance on three groups:

| Model                      | agg (%) | other (%) | knn (%) | avg (%) |
|---------------------------|---------|-----------|---------|---------|
| Qwen2.5-Coder-32B         | **47.51** | **74.75**  | **74.77** | **68.43** |
| qwen2.5-coder:14b         | 38.96   | 73.49     | 52.07   | 63.64  |

You can replicate this via:

```bash
python full_pipeline_runner.py
```

---

## 📤 Uploading Your Own Tables

1. Format your table data as `tables.jsonl`
2. Mount into `uploader/`
3. Run:

```bash
docker compose run uploader
```

---

## 🧪 Test the CLIP API

```bash
curl -X POST http://localhost:8000/text-embedding/   -H "Content-Type: application/json"   -d '{"text": "The Eiffel Tower is in Paris."}'
```

---

## 🧪 Benchmark Design (from paper)

Our benchmark is built from 12,957 question–query–table triples, enriched with:

- Wikipedia summaries (as semantic signal)
- CLIP embeddings injected into Elasticsearch index
- k-NN clauses in DSL when relevant

Each NLQ is paired with a hybrid DSL query that:
- Filters structured fields
- Ranks results via semantic similarity on text

---

## 🧾 Citation

```bibtex
@inproceedings{
  queryingbeyond2025,
  title     = {Querying Beyond Keywords: Translating Natural Language to Elasticsearch DSL with Vector Search},
  author    = {Anonymous},
  booktitle = {NeurIPS 2025 Submission},
  year      = {2025}
}
```

---

## 📜 License

MIT License © 2025 Aysel Yagmur 
Developed in support of the EXA4MIND project (EU Horizon 101092944)
