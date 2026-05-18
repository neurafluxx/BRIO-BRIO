FROM python:3.11-slim

WORKDIR /app

ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/tmp/st_cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Install numpy first to avoid C API warning at runtime
RUN pip install --no-cache-dir "numpy>=1.24,<2.0"

# Install torch 2.4.0 — Linux PyPI wheel is CPU-only (no CUDA, ~760MB)
RUN pip install --no-cache-dir torch==2.4.0

# Install app dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

EXPOSE 8000

CMD uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}
