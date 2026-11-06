# Design notes — overview

These are my engineering notes for the build. I split them by decision so each one
stands on its own; this page is just the map.

- [chunking.md](chunking.md) — chunk size, overlap, and splitting method, with the
  retrieval benchmark that picked structure-aware 800/120 over fixed-window splits.
- [embeddings.md](embeddings.md) — why `bge-small-en-v1.5`, and how it compares to
  `all-MiniLM-L6-v2`.
- [ingestion.md](ingestion.md) — how I load, clean, chunk, and embed the PDF, the
  PDF-parsing gotchas I hit, and what metadata I store per chunk (and why).
- [retrieval.md](retrieval.md) — request normalisation, retrieval, claim-level
  evidence validation, multilingual answers, guide-wide summaries, and citations.
- [evaluation.md](evaluation.md) — the two evaluation layers: the offline retrieval
  benchmark and the DeepEval LLM-answer scoring (faithfulness / relevancy).

## How the pieces fit

```
              ingestion (run once, offline)                         serving (per request)
  PDF ─► pdf_loader ─► chunking ─► embeddings ─► Pinecone ◄──── retrieve ◄─ contextualize ◄─ user
        (fitz, layout)  (800/120)   (bge-small)   (cosine)         │
                                                                   ▼
                                                          relevant? ──no──► refuse
                                                                   │yes
                                                                   ▼
                                                     claims + evidence IDs (OpenAI)
                                                                   │
                                                                   ▼
                                                       independent validation
                                                         │              │
                                                        pass          fail
                                                         │              │
                                                         ▼              ▼
                                            deterministic citations    refuse

  Guide-wide summary requests bypass top-k retrieval and use all extracted,
  page-tagged PDF segments in a separate, citation-checked summary pass.
```

The reproducible retrieval evidence lives in `eval/` — `questions.yaml` is the
graded eval set (expected pages verified by hand) and `run_eval.py` is the offline
benchmark that produced the chunking and embedding numbers in these notes.
