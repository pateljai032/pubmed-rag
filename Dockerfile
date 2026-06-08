# syntax=docker/dockerfile:1.6
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System packages needed for some Python wheels (faiss, sentence-transformers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps in a layer that caches well: only re-runs when pyproject.toml changes.
# We create a placeholder package so `pip install .` resolves the metadata before source exists.
COPY pyproject.toml ./
RUN mkdir -p src/pubmed_rag && touch src/pubmed_rag/__init__.py \
    && pip install --upgrade pip \
    && pip install .

# Now copy the real source code
COPY src/ ./src/
COPY configs/ ./configs/

# Reinstall in editable mode so the real code is registered
RUN pip install -e .

# Run as non-root for security
RUN useradd -m -u 1000 app && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["uvicorn", "pubmed_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]