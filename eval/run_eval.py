"""Offline retrieval benchmark used to justify my chunking and embedding choices.

It is deliberately self-contained and uses no cloud services: chunks are embedded
in memory with sentence-transformers and scored with cosine similarity, so I can
sweep configurations in seconds without touching Pinecone or spending API calls.

Two comparisons live here:
  * chunking  — naive fixed-window over raw page text vs. my structure-aware chunker
  * embedding — bge-small-en-v1.5 vs. all-MiniLM-L6-v2

Metrics on the eval set (eval/questions.yaml): hit@k (is the right page in the top-k?)
and MRR (how high did the first correct page rank?).

Run:  python -m eval.run_eval
"""

from __future__ import annotations

import sys
from pathlib import Path

import fitz
import numpy as np
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.chunking import build_chunks  # noqa: E402
from src.config import settings  # noqa: E402
from src.embeddings import QUERY_INSTRUCTION  # noqa: E402

TOP_K = 5
QUESTIONS = yaml.safe_load((Path(__file__).parent / "questions.yaml").read_text())


def naive_chunks(pdf_path: str, chunk_size: int, overlap: int):
    """Baseline: split raw per-page text on a fixed window, page number only.

    No header/footer stripping, no section awareness, no heading prefix — this is
    what you get from a generic PyPDF + RecursiveCharacterTextSplitter pipeline.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    doc = fitz.open(pdf_path)
    out = []
    for idx in range(doc.page_count):
        text = doc[idx].get_text().strip()
        if not text:
            continue
        for piece in splitter.split_text(text):
            out.append((piece, idx + 1))  # printed page ~= index + 1
    doc.close()
    return out


def structured_chunks(pdf_path: str, chunk_size: int, overlap: int):
    chunks = build_chunks(pdf_path, chunk_size, overlap)
    return [(c.text, c.metadata["page"]) for c in chunks]


def evaluate(pairs, model_name: str, k: int = TOP_K):
    model = SentenceTransformer(model_name)
    texts = [t for t, _ in pairs]
    pages = [p for _, p in pairs]
    doc_vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    instr = QUERY_INSTRUCTION if "bge" in model_name else ""
    queries = [instr + item["q"] for item in QUESTIONS]
    q_vecs = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)

    hits = 0
    rr_sum = 0.0
    for qv, item in zip(q_vecs, QUESTIONS):
        scores = doc_vecs @ qv
        order = np.argsort(scores)[::-1][:k]
        ranked_pages = [pages[i] for i in order]
        expected = set(item["pages"])
        rank = next((r for r, pg in enumerate(ranked_pages, 1) if pg in expected), None)
        if rank:
            hits += 1
            rr_sum += 1.0 / rank
    n = len(QUESTIONS)
    return hits / n, rr_sum / n


def main():
    pdf = settings.pdf_path
    print(f"Eval set: {len(QUESTIONS)} questions  |  top_k={TOP_K}\n")

    print("== Chunking (embedding model held at bge-small-en-v1.5) ==")
    configs = [
        ("naive fixed-window 1000/150", naive_chunks(pdf, 1000, 150)),
        ("naive fixed-window 500/100", naive_chunks(pdf, 500, 100)),
        ("structure-aware 800/120", structured_chunks(pdf, 800, 120)),
        ("structure-aware 500/80", structured_chunks(pdf, 500, 80)),
    ]
    for name, pairs in configs:
        hit, mrr = evaluate(pairs, "BAAI/bge-small-en-v1.5")
        print(f"  {name:32}  chunks={len(pairs):4}  hit@5={hit:.2f}  MRR={mrr:.3f}")

    print("\n== Embedding model (chunking held at structure-aware 800/120) ==")
    pairs = structured_chunks(pdf, 800, 120)
    for model_name in ["BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2"]:
        hit, mrr = evaluate(pairs, model_name)
        print(f"  {model_name:42}  hit@5={hit:.2f}  MRR={mrr:.3f}")


if __name__ == "__main__":
    main()
