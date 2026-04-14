# Querying Beyond Keywords: NLQ to Elasticsearch DSL with Vector Search

This repository contains the implementation of the benchmark and experimental pipeline introduced in:

**“Querying Beyond Keywords: Translating Natural Language to Elasticsearch DSL with Vector Search Support”**

The project investigates the ability of Large Language Models (LLMs) to translate natural language queries (NLQs) into executable Elasticsearch DSL queries, including hybrid queries that combine keyword-based filtering and vector-based semantic retrieval (k-NN).

---

## Note

For legal and licensing conditions, please make sure to refer to the [LICENSE](./LICENSE) file. Any external software or product 
referenced in any manner from this repository is remaining under its original license and usage conditions, and we deny any 
liability for its usage and for any consequences thereof.

The EXA4MIND platform is currently under significant development. When using our modules, please consider security aspects. We are
happy to receive feedback from you.

---

## Installation and Usage

### Installation

Clone the repository:

```bash
git clone https://github.com/yagmurrozdemir/querying-beyond-keywords.git
cd querying-beyond-keywords
```

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### Requirements

* Python 3.10+
* A running Elasticsearch instance
* (Optional) Ollama for local model inference

---

### Elasticsearch Requirement

**Elasticsearch must be running** before executing the setup or pipeline.

This project depends on a local Elasticsearch instance for index creation and query execution.

Your Elasticsearch credentials must be defined before running the setup script. This can be done in either:

* a `.env` file
* or directly in `src/nlq_to_es/config.py`

⚠️ If Elasticsearch is not running or credentials are incorrect:

* index creation will fail
* the pipeline will not work

---

### Ollama Requirement (Optional)

If local models are used, **Ollama must be running** and the required model must be available.

You can start a model with:

```bash
ollama run <model_name>
```

⚠️ If Ollama is not running or the model is unavailable:

* inference will fail
* no predictions will be generated

---

## Setup

Run the full setup:

```bash
bash setup.sh
```

This will:

1. Create a virtual environment
2. Install dependencies
3. Download the dataset
4. Build Elasticsearch indices

---

## Dataset

The dataset is hosted on Hugging Face and downloaded automatically during setup:

https://huggingface.co/datasets/ayselyagmur/dataset

Expected structure:

```
data/dataset/
├── queries/
│   ├── train/
│   ├── validation/
│   ├── test/
├── tables/
```

---

## Running the Pipeline

### Inference

```bash
python scripts/run_batch_inference.py
```

### Evaluation

```bash
python scripts/run_evaluation.py
```

---

## Evaluation Protocol

We use execution accuracy as the primary metric:

* A query is correct if it returns the same result set as the reference query
* Each experiment is repeated 5 times
* Results are reported as mean ± standard deviation

Evaluation is performed across:

* Basic queries
* Aggregation queries
* k-NN queries

---

## Fine-Tuning

We fine-tune **Qwen2.5-Coder-32B-Instruct** using QLoRA:

```bash
python scripts/run_finetuning.py
```

Adapters:

* Basic
* Aggregation
* k-NN
* Mixed

---

## Project Structure

```
nlq_to_es_project/
├── src/                  # Core implementation
├── scripts/
│   ├── setup/            # Setup utilities (dataset + indices)
│   ├── run_batch_inference.py
│   └── run_evaluation.py
├── prompts/              # Prompt templates
├── data/                 # Dataset and outputs
├── setup.sh              # One-command setup
├── requirements.txt
├── requirements_finetuning.txt
└── README.md
```

---

## Data Organization

```
data/
├── dataset/              # Benchmark dataset (downloaded automatically)
├── inputs/               # Reserved (currently empty)
├── predictions/          # Raw LLM outputs (.out)
├── outputs/              # Fine-tuning outputs (adapters/checkpoints)
├── intermediate/         # Temporary files
├── resources/            # Mappings and schemas
```

---

## Initial version contributor(s)

Aysel Yagmur Ozdemir (METU)

## Acknowledgement

This work received the support of the EXA4MIND project, funded by the European Union´s Horizon Europe Research and Innovation Programme, under Grant Agreement N° 101092944. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Commission. Neither the European Union nor the granting authority can be held responsible for them.

We thank the authors of all open-source work re-used or leveraged upon here.