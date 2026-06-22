"""Pinecone connection used by both ingestion and the running app.

The app only ever reads — it connects to an index that ingest.py has already
populated and runs similarity search. I return scores alongside documents so the
graph can refuse when the best match is too weak (the strict-grounding gate).
"""

from __future__ import annotations

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from .config import settings
from .embeddings import get_embeddings


def _client() -> Pinecone:
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not set — copy .env.example to .env.")
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_index() -> Pinecone:
    """Create the serverless index if it doesn't exist yet (ingestion only)."""
    pc = _client()
    existing = {i["name"] for i in pc.list_indexes()}
    if settings.index_name not in existing:
        pc.create_index(
            name=settings.index_name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    return pc


def get_vectorstore() -> PineconeVectorStore:
    _client()  # fail fast with a clear message if the key is missing
    return PineconeVectorStore(
        index_name=settings.index_name,
        embedding=get_embeddings(),
        text_key="text",  # raw passage text lives in metadata["text"]
    )


def search(query: str, k: int | None = None):
    """Return [(Document, score)] best-first; score is cosine similarity."""
    k = k or settings.top_k
    return get_vectorstore().similarity_search_with_score(query, k=k)
