# Querying Beyond Keywords: NLQ to Elasticsearch DSL with Vector Search

This repository contains the implementation of the benchmark and experimental pipeline introduced in:

**“Querying Beyond Keywords: Translating Natural Language to Elasticsearch DSL with Vector Search Support”**

---

## Overview

This project investigates the ability of Large Language Models (LLMs) to translate natural language queries (NLQs) into executable Elasticsearch DSL queries, including hybrid queries that combine:

* keyword-based filtering
* vector-based semantic retrieval (k-NN)

The repository provides:

* a benchmark derived from WikiSQL
* a pipeline for hybrid query construction
* inference scripts for multiple LLMs
* execution-based evaluation
* parameter-efficient fine-tuning (QLoRA)

---

## Requirements

* Python 3.10+
* A running Elasticsearch instance (see below)
* (Optional) Ollama for local model inference

---

## Elasticsearch Requirement

⚠️ Elasticsearch must be running before executing the setup or pipeline.

This project depends on a local Elasticsearch instance for index creation and query execution.

Your Elasticsearch credentials must be defined **before running the setup script**. This can be done in either:

* a `.env` file
* or directly in `src/nlq_to_es/config.py`


⚠️ If Elasticsearch is not running or credentials are incorrect:

* index creation will fail
* the pipeline will not work

---

## Ollama Requirement (for Local Models)

If you are using local models, **Ollama must be running**.

This project supports inference via Ollama-based models. Ensure that:

* Ollama is running locally
* the required model is installed

You can start a model with:

```bash
ollama run <model_name>
```

Example configuration is defined in `src/nlq_to_es/config.py`.

⚠️ If Ollama is not running or the model is not available:

* inference will fail
* no predictions will be generated

---

## Project Structure

```id="7t1b7v"
nlq_to_es_project/
├── src/                  # Core implementation
├── scripts/
│   ├── setup/            # Setup utilities (dataset + indices)
│   ├── run_batch_inference.py
│   └── run_evaluation.py
├── prompts/              # Prompt templates
├── data/                 # Dataset and outputs
├── tests/
├── setup.sh              # One-command setup
├── requirements.txt
├── requirements_finetuning.txt
└── README.md
```

---

## Data Organization

```id="2u7d7x"
data/
├── dataset/              # Benchmark dataset (downloaded automatically)
├── inputs/               # Reserved (currently empty)
├── predictions/          # Raw LLM outputs (.out)
├── outputs/              # Fine-tuning outputs (adapters/checkpoints)
├── intermediate/         # Temporary files
├── resources/            # Mappings and schemas
```

---

## Setup (Recommended)

⚠️ Ensure Elasticsearch is running and credentials are configured before proceeding.

Run the full setup with one command:

```bash id="2twr6k"
bash setup.sh
```

This will:

1. Create a virtual environment (`venv/`)
2. Install dependencies
3. Download dataset from Hugging Face
4. Build Elasticsearch indices

---

## Manual Setup (Optional)

### 1. Create environment

```bash id="l9u2c9"
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash id="k6k3mw"
pip install -r requirements.txt
```

### 3. Download dataset

```bash id="k6bc8g"
python scripts/setup/download_hf_dataset.py
```

### 4. Build indices

```bash id="f5r3n5"
python scripts/setup/build_indices.py
```

---

## Dataset

The dataset is hosted on Hugging Face and downloaded automatically during setup:

https://huggingface.co/datasets/ayselyagmur/dataset

Expected structure:

```id="eq8g3k"
data/dataset/
├── queries/
│   ├── train/
│   ├── validation/
│   ├── test/
├── tables/
```

---

## Running the Pipeline

### 1. Inference

```bash id="s7m0lc"
python scripts/run_batch_inference.py
```

### 2. Evaluation

```bash id="qg5j7n"
python scripts/run_evaluation.py
```

---

## Evaluation Protocol

We use **execution accuracy** as the primary metric:

* A query is correct if it returns the same result set as the reference query
* Each experiment is repeated **5 times**
* Results are reported as **mean ± standard deviation**
* Evaluated across:

  * Basic queries
  * Aggregation queries
  * k-NN queries

---

## Fine-Tuning

We fine-tune **Qwen2.5-Coder-32B-Instruct** using QLoRA:

```bash id="ewg0tf"
python scripts/run_finetuning.py
```

Adapters:

* Basic
* Aggregation
* k-NN
* Mixed

---

## Notes

* k-NN queries use a placeholder (`$vector$`)
* Embeddings are generated externally
* Elasticsearch must be running before executing queries

---

## License

MIT License © 2026 Aysel Yağmur Özdemir

Developed in support of the **EXA4MIND project**
(EU Horizon Europe Grant No. 101092944)

---

## Citation

```id="l5wfxf"
@article{ozdemir2026querying,
  title={Querying Beyond Keywords: Translating Natural Language to Elasticsearch DSL with Vector Search Support},
  author={Ozdemir, Aysel Yagmur and Karagoz, Pinar and Toroslu, Ismail Hakki},
  journal={},
  year={2026}
}
```
