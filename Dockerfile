# Single-container build that runs the Chainlit UI against the already-populated
# Pinecone index. The embedding model is baked into the image so the container
# answers immediately on start, with no model download at run time.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only torch first, so sentence-transformers doesn't pull the large CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model into the image (must match EMBEDDING_MODEL).
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY . .

EXPOSE 8000

# PORT is read from --env-file; defaults to 8000 if unset. The ${PORT%% *} trims any
# trailing text (e.g. an inline comment that --env-file didn't strip) so a stray
# comment in .env can't break startup.
CMD ["sh", "-c", "P=\"${PORT:-8000}\"; chainlit run app.py --host 0.0.0.0 --port \"${P%% *}\""]
