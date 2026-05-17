FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set HuggingFace cache to writable /tmp paths
ENV HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    SENTENCE_TRANSFORMERS_HOME=/tmp/st_cache \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch FIRST (before other deps, so nothing pulls CUDA)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source
COPY src/ ./src/

# Expose port (Railway sets $PORT at runtime)
EXPOSE 8000

# Start command
CMD uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}
