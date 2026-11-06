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


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- Secrets / connections ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
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
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-6-luna")
    # Completion budgets include non-visible reasoning tokens. Leave enough room for
    # structured responses instead of allowing a reasoning model to exhaust its
    # budget before emitting JSON.
    openai_max_completion_tokens: int = _int("OPENAI_MAX_COMPLETION_TOKENS", 2048)
    # Whole-guide summaries receive the complete extracted PDF, so reserve room
    # for a useful synthesis beyond the normal question-answer budget.
    openai_summary_max_completion_tokens: int = _int("OPENAI_SUMMARY_MAX_COMPLETION_TOKENS", 4096)
    # Contextualization, claim extraction, and evidence checks are bounded routing
    # tasks, so low reasoning effort keeps latency and token use predictable.
    openai_reasoning_effort: str = os.getenv("OPENAI_REASONING_EFFORT", "low")

    # --- Ingestion ---
    pdf_path: str = os.getenv("PDF_PATH", "iPhone User Guide 1.pdf")
    chunk_size: int = _int("CHUNK_SIZE", 800)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 120)

    # --- Retrieval / generation ---
    top_k: int = _int("TOP_K", 5)
    # Coarse pre-filter: below this best-match cosine score I treat retrieval as
    # "nothing relevant" and refuse before even calling the model. bge similarities
    # sit high (~0.5 even for unrelated text), so this only catches the obviously
    # off-topic; the authoritative grounding check is the model's refusal instruction
    # over the actual excerpts. Both numbers are justified in docs/retrieval.md.
    min_relevance_score: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.55"))
    # Keep untrusted user input and conversational context within a predictable
    # provider/context budget.  History is used only to resolve a follow-up.
    max_question_chars: int = _int("MAX_QUESTION_CHARS", 4000)
    max_history_messages: int = _int("MAX_HISTORY_MESSAGES", 12)
    # Disabled by default because contextualized queries may contain user text.
    enable_rag_diagnostics: bool = _bool("ENABLE_RAG_DIAGNOSTICS")

    # --- UI ---
    port: int = _int("PORT", 8000)
    # Seconds between displayed characters while streaming a response.
    stream_char_delay: float = _float("STREAM_CHAR_DELAY", 0.009)

    def __post_init__(self):
        # The deployed Pinecone vectors were created with this exact embedding space.
        # Accepting an arbitrary environment override would produce invalid retrieval
        # and make Docker download a model that was not baked into the image.
        if self.embedding_model != "BAAI/bge-small-en-v1.5" or self.embedding_dim != 384:
            raise ValueError(
                "EMBEDDING_MODEL and EMBEDDING_DIM must match the pre-ingested "
                "BAAI/bge-small-en-v1.5 / 384-dimension Pinecone index."
            )
        if (
            self.max_question_chars < 1
            or self.max_history_messages < 1
            or self.openai_max_completion_tokens < 1
            or self.openai_summary_max_completion_tokens < 1
        ):
            raise ValueError(
                "MAX_QUESTION_CHARS, MAX_HISTORY_MESSAGES, and "
                "OPENAI_MAX_COMPLETION_TOKENS, and OPENAI_SUMMARY_MAX_COMPLETION_TOKENS must be positive."
            )


settings = Settings()
