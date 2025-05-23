FROM python:3.11-slim

WORKDIR /app

# Add system dependencies (for torch/transformers)
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all code
COPY . .

CMD ["python", "full_pipeline_runner.py"]
