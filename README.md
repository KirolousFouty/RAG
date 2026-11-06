# iPhone User Guide — RAG chatbot

A retrieval-augmented chatbot that lets you have a conversation with the contents of
the **Apple iPhone User Guide (iOS 7.1)**. It answers **only** from the guide, **cites
the page** every answer comes from, and **says so explicitly** when the answer isn't in
the document — it never falls back on general knowledge.

I wrote up the reasoning behind the main decisions in [`docs/`](docs/design-notes.md)
(chunking, embeddings, ingestion/metadata, retrieval & grounding), backed by an offline
retrieval benchmark in [`eval/`](eval/).

---

## Port

The app serves on **port 8000** by default (configurable via the `PORT` env var). The
run command below maps it to `localhost:8000`.

## How the graders run it

There is **no ingestion step at grade time** — the Pinecone index is already populated,
and the app answers immediately on container start.

```bash
# 1. Clone
git clone https://github.com/KirolousFouty/RAG
cd RAG

# 2. Fill in credentials
cp .env.example .env
#    then edit .env and set OPENAI_API_KEY and PINECONE_API_KEY

# 3. Build
docker build -t chatbot:1.0 .

# 4. Run
docker run -p 8000:8000 --env-file .env chatbot:1.0

# 5. Open http://localhost:8000 and chat
```

If you change `PORT` in `.env`, map that port instead, e.g. `PORT=9000` →
`docker run -p 9000:9000 --env-file .env chatbot:1.0`.

> The first build downloads the embedding model and bakes it into the image, so the
> container starts without any model download. Expect the build to take a few minutes.

---

## Models (exact names & versions)

| Role            | Model                                              | Notes |
| --------------- | -------------------------------------------------- | ----- |
| Chat / generation | **GPT-6 Luna** — `gpt-6-luna` (OpenAI) | The configured OpenAI model is used for every chat. |
| Embeddings      | **BAAI/bge-small-en-v1.5** (384-dim)               | open model run in-container via `sentence-transformers`; no extra API key, and it let me benchmark retrieval offline |

The chat model is configurable with `OPENAI_CHAT_MODEL`. The
embedding model and dimension are intentionally locked to the pre-ingested Pinecone
index; changing either requires re-ingestion and a matching image build.

## Stack

- **Orchestration:** LangChain, with a compiled LangGraph state-graph definition
- **UI:** Chainlit
- **Vector store:** Pinecone (serverless, free tier; cosine, 384-dim)
- **PDF parsing:** PyMuPDF

Pinned versions are in [`requirements.txt`](requirements.txt).

---

## Architecture

```
              ingestion (run once, offline — src/ingest.py)
  PDF ─► pdf_loader ─► chunking ─► embeddings ─► Pinecone
        (fitz, layout)  (800/120)  (bge-small)   (cosine, 384-dim)

              serving (per request — app.py + src/rag_graph.py)
  user ─► sanitize ─► contextualize ─► retrieve(top-5) ─► relevant? ─no─► refuse
                                                            │yes
                                                            ▼
                                             generate claims + evidence IDs
                                                            │
                                                            ▼
                                               independently validate claims
                                                    │                 │
                                                   pass              fail
                                                    │                 │
                                                    ▼                 ▼
                                    render deterministic citations     refuse
```

Guide-wide summary requests bypass top-k retrieval and use all extracted, page-tagged
PDF segments in a separate summary pass.

- **`src/pdf_loader.py`** — parses the guide with PyMuPDF, using font size and the
  page footer to recover the chapter, section, and printed page for every passage, and
  to strip running headers/footers and figure noise.
- **`src/chunking.py`** — splits within sections (800/120) and keeps page/section
  metadata on every chunk.
- **`src/embeddings.py`** — bge-small wrapper (applies the bge query instruction).
- **`src/vectorstore.py`** — Pinecone connection + similarity search with scores.
- **`src/rag_graph.py`** — request sanitisation, contextualisation, per-need retrieval,
  claim generation, independent evidence validation, localisation, and guide-wide
  summaries. It also defines a compiled LangGraph state graph; the current UI request
  path orchestrates those helpers directly.
- **`src/ingest.py`** — standalone, re-runnable ingestion script.
- **`app.py`** — Chainlit UI; per-session conversation memory; renders the Sources block.

## Grounding & citations (the mandatory behaviours)

- **Strictly grounded.** The model is instructed to use only the retrieved excerpts and
  given no licence to use outside knowledge. Earlier turns are used only to interpret
  the question, never as a source of facts.
- **Explicit "not in the document" fallback.** Three layers: a relevance gate rejects
  empty or weak retrieval before answer generation; the model returns either a
  structured refusal or evidence-linked claims; and an independent validation pass
  must approve every claim. Citations are dropped on any refusal. Details and the
  measured threshold are in [`docs/retrieval.md`](docs/retrieval.md).
- **Page/section citations on every answer.** The answer model returns claims linked to
  excerpt IDs. A separate evidence check must approve every claim; the application then
  renders inline page references and the **Sources** block exclusively from the approved
  chunks' metadata. Model-written page numbers are never trusted.
- **Multi-turn memory.** Recent conversation is kept per session only to rewrite a
  follow-up into a standalone retrieval query. Final factual generation receives fresh
  guide excerpts, not raw prior assistant answers.
- **Multi-part and multilingual requests.** Independent information needs are retrieved
  and validated separately before their results are combined. Retrieval is normalised to
  English; validated claims can then be localised, while citations remain deterministic.

---

## Configuration

Every required variable is documented in [`.env.example`](.env.example). The real
`.env` is gitignored and is **never** committed — no key appears anywhere in the git
history. Required at run time: `PINECONE_API_KEY` and `OPENAI_API_KEY` (plus the index name,
which defaults to `iphone-user-guide`).

## Re-running ingestion (not needed for grading)

The index is populated before submission. To rebuild it yourself (with `.env`
configured and dependencies installed):

```bash
pip install -r requirements.txt
python -m src.ingest            # build/refresh the index from the PDF
python -m src.ingest --recreate # delete and rebuild from scratch
```

## Reproducing the experiments

Two layers of evaluation (see [`docs/evaluation.md`](docs/evaluation.md)):

```bash
# 1. Retrieval benchmark — offline, no cloud, no API calls
python -m eval.run_eval

# 2. LLM-answer evaluation with DeepEval — needs .env (live index + chat model)
pip install -r requirements-eval.txt
python -m eval.deepeval_eval
```

`run_eval` prints the hit@5 / MRR tables quoted in the design notes. `deepeval_eval`
scores the generated answers for faithfulness (groundedness), answer relevancy, and
contextual relevancy, using the configured OpenAI model as the judge.
