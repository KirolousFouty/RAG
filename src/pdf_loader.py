"""Parse the iPhone User Guide into clean, metadata-tagged text segments.

The guide has a very regular layout that I exploit instead of dumping raw text:

  * Every body page carries a footer (near the bottom of the page) with the
    chapter number, chapter title, and the *printed* page number. That footer is
    the authoritative source for the page/chapter metadata that flows through to
    citations, so I extract it and then strip it from the body.
  * Headings are set in larger/semibold type (chapter titles ~20pt, section
    headings 12-14pt) over 10pt body text. I use font size to recover the section
    a passage lives under.
  * Figure labels are tiny 7.5pt Helvetica fragments ("Volume up") that add only
    noise, so I drop them.

The output is a list of `Segment`s: a contiguous run of body text under one
heading, tagged with the page, chapter, and section it belongs to. Splitting into
fixed-size chunks happens later (chunking.py) — keeping this layer at the
section granularity means each chunk inherits an accurate section label.

Notes / gotchas I hit while writing this (documented more fully in docs/ingestion.md):
  * The printed page number differs from the PDF page index by the front matter;
    I trust the footer's number and fall back to index+1.
  * Spans within a line are sometimes split mid-sentence by inline link styling.
    PyMuPDF's plain text join handles spacing, but I need font sizes per line, so
    I reconstruct lines from the dict view and re-insert spaces from the x-gaps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF

# Layout thresholds, in points, tuned against this document.
FOOTER_Y = 720.0          # anything below this is the running footer
CHAPTER_TITLE_SIZE = 18.0  # chapter title on a chapter-opening page
HEADING_SIZE = 11.5        # section / sub-section headings (12pt and 14pt)
MIN_BODY_SIZE = 9.5        # below this is a figure caption / fine print -> drop
PARAGRAPH_GAP = 8.0        # vertical gap that signals a paragraph break


@dataclass
class Segment:
    text: str
    page: int            # printed page number, as the reader sees it
    pdf_page: int        # 1-based index into the PDF, for debugging
    chapter: str
    section: str = ""


@dataclass
class _Line:
    text: str
    size: float
    y0: float
    y1: float


def _reconstruct_line(spans: list[dict]) -> _Line | None:
    """Join a line's spans back into text, re-inserting spaces lost to styling."""
    spans = [s for s in spans if s["text"]]
    if not spans:
        return None
    out = spans[0]["text"]
    for prev, cur in zip(spans, spans[1:]):
        gap = cur["bbox"][0] - prev["bbox"][2]
        if gap > 1.0 and not out.endswith(" ") and not cur["text"].startswith(" "):
            out += " "
        out += cur["text"]
    size = max(s["size"] for s in spans)
    y0 = min(s["bbox"][1] for s in spans)
    y1 = max(s["bbox"][3] for s in spans)
    return _Line(text=_clean(out), size=size, y0=y0, y1=y1)


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace(" ", " ").replace(" ", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def _footer(page: fitz.Page) -> tuple[str, int | None]:
    """Return (chapter label, printed page number) from the page footer."""
    parts: list[tuple[float, str]] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                t = span["text"].strip()
                if t and span["bbox"][1] > FOOTER_Y:
                    parts.append((span["bbox"][0], t))
    parts.sort()
    tokens = [p[1] for p in parts]

    printed_page = None
    label_tokens = []
    for tok in tokens:
        if tok.isdigit():
            printed_page = int(tok)
        elif tok != "Contents":
            label_tokens.append(tok)

    # "Chapter  1" + "iPhone at a Glance" -> "Chapter 1: iPhone at a Glance"
    chapter = ""
    if label_tokens:
        head = _clean(label_tokens[0])
        rest = _clean(" ".join(label_tokens[1:]))
        chapter = f"{head}: {rest}" if rest else head
    return chapter, printed_page


def _opener_chapter(page: fitz.Page) -> str:
    """Reconstruct the chapter label from a chapter-opening page.

    Opener pages carry no chapter footer, but they set the chapter number/letter
    in giant type (~55pt) and the title just below (~20pt). Without this, the first
    page of most chapters (e.g. Siri's "Make requests") would be dropped entirely.
    """
    number = ""
    title = ""
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                t = span["text"].strip()
                if not t:
                    continue
                if span["size"] >= 40 and not number:
                    number = t
                elif 18 <= span["size"] < 40 and not title:
                    title = _clean(t)
    if not (number and title):
        return ""
    label = "Appendix" if number.isalpha() else "Chapter"
    return f"{label} {number}: {title}"


def load_segments(pdf_path: str) -> list[Segment]:
    doc = fitz.open(pdf_path)
    segments: list[Segment] = []
    current_chapter = ""
    current_section = ""

    for idx in range(doc.page_count):
        page = doc[idx]
        chapter, printed = _footer(page)
        if not chapter:
            # chapter-opening pages carry the label in giant type, not the footer
            chapter = _opener_chapter(page)

        # The cover and the table-of-contents pages have no chapter label at all and
        # are pure navigation — they would only pollute retrieval, so I skip them.
        if not chapter:
            continue
        current_chapter = chapter
        printed_page = printed if printed is not None else idx + 1

        lines: list[_Line] = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                if line["spans"] and line["spans"][0]["bbox"][1] > FOOTER_Y:
                    continue  # footer, already captured
                rl = _reconstruct_line(line["spans"])
                if rl and rl.text:
                    lines.append(rl)
        lines.sort(key=lambda ln: ln.y0)

        buffer: list[str] = []
        prev_y1 = None

        def flush():
            nonlocal buffer
            if buffer:
                text = " ".join(buffer)
                text = re.sub(r"\s+\n", "\n", text).strip()
                if len(text) > 1:
                    segments.append(
                        Segment(
                            text=text,
                            page=printed_page,
                            pdf_page=idx + 1,
                            chapter=current_chapter,
                            section=current_section,
                        )
                    )
            buffer = []

        for ln in lines:
            if ln.size < MIN_BODY_SIZE:
                continue  # figure caption / fine print
            if ln.size >= CHAPTER_TITLE_SIZE:
                # chapter opener title; chapter name already comes from the footer
                flush()
                current_section = ""
                prev_y1 = None
                continue
            if ln.size >= HEADING_SIZE:
                flush()
                current_section = ln.text
                prev_y1 = None
                continue
            # body line: keep paragraph structure via the vertical gap
            if prev_y1 is not None and (ln.y0 - prev_y1) > PARAGRAPH_GAP:
                buffer.append("\n")
            buffer.append(ln.text)
            prev_y1 = ln.y1

        flush()

    doc.close()
    return segments


if __name__ == "__main__":
    import sys

    from .config import settings

    segs = load_segments(sys.argv[1] if len(sys.argv) > 1 else settings.pdf_path)
    print(f"{len(segs)} segments")
    for s in segs[:6]:
        print(f"\n[p{s.page} | {s.chapter} | {s.section}]\n{s.text[:200]}")
