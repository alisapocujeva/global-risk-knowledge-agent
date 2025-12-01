FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=100
ENV PIP_NO_CACHE_DIR=0

# Install system dependencies for sentence-transformers, chromadb, and xhtml2pdf
# Using --no-install-recommends and cleaning in same layer to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    pkg-config \
    libcairo2-dev \
    libpango1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
# Using pip cache and parallel builds for faster installation
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Note: Embedding model will be downloaded on first use (lazy loading)
# This saves build time - model download happens at runtime on first query

# Copy application code (this layer changes most often)
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
