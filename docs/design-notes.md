# Design notes — overview

These are my engineering notes for the build. I split them by decision so each one
stands on its own; this page is just the map.

- [chunking.md](chunking.md) — chunk size, overlap, and splitting method, with the
  retrieval benchmark that picked structure-aware 800/120 over fixed-window splits.
- [embeddings.md](embeddings.md) — why `bge-small-en-v1.5`, and how it compares to
  `all-MiniLM-L6-v2`.
- [ingestion.md](ingestion.md) — how I load, clean, chunk, and embed the PDF, the
  PDF-parsing gotchas I hit, and what metadata I store per chunk (and why).
- [retrieval.md](retrieval.md) — the LangGraph pipeline, multi-turn memory, the
  two-layer strict-grounding design, and how citations are produced.

## How the pieces fit

```
              ingestion (run once, offline)                     serving (per question)
  PDF ─► pdf_loader ─► chunking ─► embeddings ─► Pinecone ◄──── retrieve ◄─ contextualize ◄─ user
        (fitz, layout)  (800/120)   (bge-small)   (cosine)         │
                                                                   ▼
                                                          grounded? ──no──► "not in the guide"
                                                                   │yes
                                                                   ▼
                                                      generate (Claude Opus 4.8) ─► answer + Sources
```

The reproducible evidence behind the choices lives in `eval/` — `questions.yaml` is the
graded eval set (expected pages verified by hand) and `run_eval.py` is the offline
benchmark that produced every number in these notes.
