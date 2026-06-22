# Ingestion pipeline & vector-store metadata

The ingestion path is `src/ingest.py`, which composes three modules:
`pdf_loader.py` → `chunking.py` → `embeddings.py` → Pinecone. It runs once, before
submission; the app only ever queries the index.

## 1. Load & parse (`src/pdf_loader.py`)

I parse with **PyMuPDF (fitz)** rather than a plain text extractor, because I need
font sizes and positions to recover structure — and that structure is exactly what
makes citations and chunking work.

The guide has a very regular layout that I lean on:

- **Footer (near the page bottom):** `Chapter N`, the chapter title, and the *printed*
  page number. I read it for metadata, then strip it from the body.
- **Headings:** chapter titles ~20pt, section headings 12–14pt, over 10pt body text.
  Font size tells me where a section starts.
- **Figure labels:** tiny 7.5pt Helvetica fragments ("Volume up", "Siri's response").

The loader walks every page and emits **segments** — a contiguous run of body text
under one heading, tagged with `page`, `pdf_page`, `chapter`, and `section`. It yields
325 segments for this guide.

## 2. PDF-parsing gotchas I hit (and how I handled them)

These are the things a naive `get_text()` dump gets wrong:

- **Headers/footers repeat on every page.** The running footer (`Chapter 3  Basics
  32`) is noise in the body but gold for metadata. I detect it by vertical position
  (`y > 720pt`), extract the chapter + printed page from it, and drop it from the text.
- **Printed page ≠ PDF page index.** The footer's printed number is what cross-
  references and a human reader use, so that's what I cite; I keep the raw PDF index as
  `pdf_page` for debugging. They differ by the front matter.
- **Cover and table-of-contents pages.** Pure navigation, no chapter — I skip them so
  they don't pollute retrieval (a TOC line "32  Do Not Disturb" would otherwise be a
  tempting but useless match).
- **Chapter-opening pages have *no* chapter footer.** They set the chapter
  number/letter in giant ~55pt type and the title in ~20pt instead. My first version
  silently dropped the first page of almost every chapter (including Siri's "Make
  requests"). I now reconstruct the label from those big glyphs —
  `"Chapter 4: Siri"`, `"Appendix A: Accessibility"` — which recovered ~50 segments.
- **Spans split mid-sentence.** Inline link styling chops lines into separate spans
  ("Voice" / "Control" / "on page 29"). I rebuild each line from its spans and
  re-insert spaces from the x-gaps, then normalise non-breaking spaces and stray spaces
  before punctuation.
- **Multi-column / figure callouts.** The 7.5pt Helvetica figure labels are fragmentary
  and add only noise, so I drop anything below 9.5pt. Body text in this guide is
  single-column, so I sort lines top-to-bottom and reconstruct paragraphs from the
  vertical gaps between lines.
- **Tables.** This guide has essentially no data tables (it's gesture/setting
  instructions), so I didn't build special table handling — I note it here as a known
  limitation rather than pretend it's solved.

## 3. Chunk (`src/chunking.py`)

Each segment is split with a recursive character splitter (800 / 120, see
`chunking.md`). Most segments are short enough to pass through whole. Every chunk keeps
its segment's metadata and gets a **deterministic ID** — `sha1(source|page|ordinal)` —
so re-running ingestion upserts over the same vectors instead of duplicating them.

## 4. Vector-store metadata — what I store and why

Each vector carries this metadata payload:

| Field      | Example                          | Why it earns its place |
| ---------- | -------------------------------- | ---------------------- |
| `text`     | the raw passage                  | what the model reads and what's shown; `text_key` for LangChain's Pinecone wrapper |
| `page`     | `32`                             | **the citation.** Printed page number the user can turn to — mandatory and tested |
| `section`  | `"Do Not Disturb"`               | sharpens the citation to a sub-heading, and is prepended at embed time to anchor topic |
| `chapter`  | `"Chapter 3: Basics"`            | human-readable context in the citation; lets me group/scope results |
| `pdf_page` | `33`                             | debugging — maps a citation back to the actual PDF page when they differ |
| `source`   | `"iPhone User Guide 1.pdf"`      | provenance; makes the index safe to extend to more documents later |

The design principle: **page and section are not decoration, they are the product.**
The assessment requires every answer to cite its source, so page/section flow
untouched from the parser, through the chunk metadata, into Pinecone, and back out into
the "Sources" block under each answer (`app.py`). If parsing got the page wrong, the
citation would be wrong — which is why I spent most of the effort on the loader.

## 5. Embed & upsert

`ingest.py` embeds chunks with bge-small in batches of 100 and upserts to a serverless
Pinecone index (cosine, 384-dim) that it creates on first run. Re-runnable via
`python -m src.ingest`, or `--recreate` to rebuild from scratch.
