"""Central configuration, read from the environment.

I keep every tunable in one place so the ingestion script and the running app
agree on the index name, embedding model, and chunking parameters without me
having to remember to change two files. Everything has a sensible default so
the app boots even if only the secrets are set.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # --- Secrets / connections ---
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")

    # --- Vector store ---
    index_name: str = os.getenv("PINECONE_INDEX", "iphone-user-guide")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    # --- Models ---
    # bge-small is an open 384-dim model; it runs in-container with no API key,
    # which is also what let me benchmark retrieval offline (see docs/embeddings.md).
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedding_dim: int = _int("EMBEDDING_DIM", 384)
    # Opus 4.8 follows the "only answer from the context" instruction the most
    # reliably of the models I tried, which is the whole game for strict grounding.
    chat_model: str = os.getenv("CHAT_MODEL", "claude-opus-4-8")

    # --- Ingestion ---
    pdf_path: str = os.getenv("PDF_PATH", "iPhone User Guide 1.pdf")
    chunk_size: int = _int("CHUNK_SIZE", 800)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 120)

    # --- Retrieval / generation ---
    top_k: int = _int("TOP_K", 5)
    # Below this best-match cosine score I treat retrieval as "nothing relevant" and
    # refuse rather than answer from a weak hit. First guess — I'll revisit it once
    # I've measured the real in-scope vs out-of-scope score distributions.
    min_relevance_score: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.32"))

    # --- UI ---
    port: int = _int("PORT", 8000)


settings = Settings()
