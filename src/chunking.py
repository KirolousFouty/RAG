"""Turn metadata-tagged segments into embedding-ready chunks.

I split *within* a section rather than across the whole document. Because the
loader already broke the guide at heading boundaries, a section is usually one or
two short instructions — most segments fit in a single chunk untouched, and only
the long ones get split. That keeps each chunk topically coherent and lets every
chunk inherit its segment's page/chapter/section, which is what the citation layer
reads back out.

The split itself is a recursive character split on paragraph -> line -> sentence
-> word boundaries. I compared this against a naive fixed-window split over the
raw page text in docs/chunking.md; structure-aware won on every sample question.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .pdf_loader import Segment, load_segments


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def _splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def _chunk_id(source: str, page: int, ordinal: int) -> str:
    raw = f"{source}|{page}|{ordinal}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def build_chunks(
    pdf_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    source_name: str | None = None,
) -> list[Chunk]:
    segments = load_segments(pdf_path)
    splitter = _splitter(chunk_size, chunk_overlap)
    source = source_name or pdf_path.rsplit("/", 1)[-1]

    chunks: list[Chunk] = []
    ordinal = 0
    for seg in segments:
        for piece in splitter.split_text(seg.text):
            piece = piece.strip()
            if len(piece) < 2:
                continue
            # Prepend the section heading so a chunk carries its own topic into the
            # embedding — it measurably helped retrieval on heading-style queries.
            heading = " > ".join(p for p in (seg.chapter, seg.section) if p)
            embed_text = f"{heading}\n{piece}" if heading else piece
            chunks.append(
                Chunk(
                    id=_chunk_id(source, seg.page, ordinal),
                    text=embed_text,
                    metadata={
                        "source": source,
                        "page": seg.page,
                        "pdf_page": seg.pdf_page,
                        "chapter": seg.chapter,
                        "section": seg.section,
                        "text": piece,
                    },
                )
            )
            ordinal += 1
    return chunks


def _segment_count(pdf_path: str) -> int:
    return len(load_segments(pdf_path))


if __name__ == "__main__":
    import statistics
    import sys

    from .config import settings

    path = sys.argv[1] if len(sys.argv) > 1 else settings.pdf_path
    cs = build_chunks(path, settings.chunk_size, settings.chunk_overlap)
    lengths = [len(c.metadata["text"]) for c in cs]
    print(f"segments: {_segment_count(path)}  chunks: {len(cs)}")
    print(
        f"chunk chars  min={min(lengths)}  median={int(statistics.median(lengths))}"
        f"  mean={int(statistics.mean(lengths))}  max={max(lengths)}"
    )
