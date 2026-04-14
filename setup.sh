#!/bin/bash

set -e

echo "🚀 Setting up NLQ-to-ES project..."

echo "[1/4] Creating virtual environment..."
python3 -m venv venv

echo "[2/4] Installing dependencies..."
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt

echo "[3/4] Downloading dataset..."
venv/bin/python scripts/setup/download_hf_dataset.py

echo "[4/4] Building Elasticsearch indices..."
venv/bin/python scripts/setup/build_indices.py

echo "✅ Setup complete!"
echo ""
echo "👉 Activate environment with:"
echo "source venv/bin/activate"