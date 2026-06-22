# Chunking strategy

## Hypothesis

This document isn't free-flowing prose — it's a reference manual made of short,
self-contained "how to" instructions grouped under headings ("Do Not Disturb",
"Touch ID sensor", "Take photos and videos"). My hypothesis was that **respecting
those heading boundaries** would beat a generic fixed-window split, because a window
that ignores structure tends to either glue two unrelated instructions together or
cut one in half — both of which blur the embedding and hurt retrieval.

## What I tried

I built a tiny offline benchmark (`eval/run_eval.py`) so I could compare configs in
seconds without touching the cloud index: chunks are embedded in memory with
bge-small and scored against a 10-question eval set (`eval/questions.yaml`) whose
expected pages I confirmed by reading the guide. Metrics:

- **hit@5** — does the correct page appear in the top-5 retrieved chunks?
- **MRR** — how highly is the first correct page ranked? (rewards precision, not
  just presence)

Two families, embedding model held constant at bge-small-en-v1.5:

- **Naive fixed-window** — a generic `PyPDF + RecursiveCharacterTextSplitter`
  pipeline over raw per-page text. No header/footer stripping, no section awareness.
- **Structure-aware** (my `src/chunking.py`) — split *within* the section segments
  the loader produces, and prepend the `chapter > section` heading to each chunk
  before embedding.

## What I observed

| Configuration                  | Chunks | hit@5 | MRR   |
| ------------------------------ | -----: | :---: | :---: |
| naive fixed-window 1000 / 150  |    394 | 0.90  | 0.800 |
| naive fixed-window 500 / 100   |    767 | 0.90  | 0.825 |
| **structure-aware 800 / 120**  |    525 | **1.00** | **0.950** |
| structure-aware 500 / 80       |    779 | 1.00  | 0.875 |

Two clear signals:

1. **Structure-aware beats naive on every metric.** Both naive configs missed the
   same question outright (hit@5 0.90) and ranked correct pages lower (MRR ~0.80–0.83).
   The miss was the AirDrop question: in the naive split the AirDrop instruction got
   merged with neighbouring share-sheet text and the page footer noise, and the
   blended chunk never surfaced. Keeping the section intact fixed it.
2. **Among structure-aware configs, 800/120 beats 500/80 on MRR** (0.950 vs 0.875).
   Both find the right page, but the larger window keeps each instruction whole, so
   the correct chunk ranks first more often. 500/80 fragments a few longer
   instructions, which pushes the best chunk down the ranking. 800 chars is roughly
   one full instruction plus a little context here, which is the sweet spot.

I also kept the heading prefix on the embedded text: a chunk about brightness embeds
as `Chapter 3: Basics > Adjust the brightness\n<text>`. It anchors short, ambiguous
passages to their topic and costs nothing at query time (the *displayed* text and the
citation still come from the clean metadata, not the prefixed copy).

## Decision

**Structure-aware split, `chunk_size=800`, `chunk_overlap=120`, recursive on
paragraph → line → sentence → word boundaries.** It scored a perfect hit@5 and the
best MRR, and it produces a sensible 525 chunks (median ~508 characters).

**Trade-off I accepted:** structure-aware chunking is coupled to *this* document's
layout (it relies on the loader's heading detection). A generic fixed-window splitter
would port to any PDF unchanged. For a single, well-structured manual where retrieval
quality and citation accuracy are the whole point, I think the coupling is worth it —
and it's all driven by `CHUNK_SIZE` / `CHUNK_OVERLAP` env vars if the numbers need
revisiting.
