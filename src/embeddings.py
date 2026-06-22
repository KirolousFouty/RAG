"""Embedding model wrapper.

I use BAAI/bge-small-en-v1.5 directly through sentence-transformers rather than a
hosted embedding API. Two reasons: it needs no extra API key (the only secrets the
graders set are Anthropic + Pinecone), and running it locally is exactly what let
me benchmark chunking and embedding choices offline (docs/embeddings.md).

bge models expect a short instruction prefixed to *queries* (not documents) for
retrieval; I apply it in `embed_query` only. Vectors are L2-normalised so a dot
product equals cosine similarity, matching the Pinecone index metric.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=2)
def _model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)


class BGEEmbeddings(Embeddings):
    def __init__(self, model_name: str, query_instruction: str = QUERY_INSTRUCTION):
        self.model_name = model_name
        self.query_instruction = query_instruction if "bge" in model_name else ""

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vecs = _model(self.model_name).encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return vecs.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_instruction + text])[0]


def get_embeddings(model_name: str | None = None) -> BGEEmbeddings:
    from .config import settings

    return BGEEmbeddings(model_name or settings.embedding_model)
