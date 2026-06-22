# Embedding model choice

## Model: `BAAI/bge-small-en-v1.5` (384-dim)

Exact version: `BAAI/bge-small-en-v1.5` as served on the Hugging Face Hub, run
through `sentence-transformers==5.6.0`. 384-dimensional, normalised embeddings.

## Why this model

I had three requirements and bge-small hit all of them:

1. **No extra API key / no per-query cost.** The graders only set Anthropic and
   Pinecone keys. A hosted embedding API (OpenAI `text-embedding-3-small` etc.) would
   add a third secret and bill on every query. bge-small runs in-process, so embedding
   is free and offline. It's ~130 MB, which I bake into the Docker image at build time
   so the container answers immediately with no download.
2. **Strong retrieval for its size.** It sits near the top of the MTEB retrieval
   leaderboard among small models and clearly beat the obvious lighter alternative in
   my own test (below).
3. **It let me actually measure things.** Because it runs locally, I could sweep
   chunking and embedding configs in seconds (`eval/run_eval.py`) instead of paying
   for API calls — which is how every number in these docs got produced.

bge models expect a short instruction prefixed to **queries** (not documents) for
retrieval: `"Represent this sentence for searching relevant passages: "`. I apply it
in `embed_query` only (`src/embeddings.py`); documents are embedded as-is.

## Comparison: bge-small vs all-MiniLM-L6-v2

`all-MiniLM-L6-v2` is the default many RAG tutorials reach for — smaller (384-dim too,
~80 MB) and faster — so it's the natural baseline. Same chunks (structure-aware
800/120), same eval set:

| Embedding model                          | hit@5 | MRR   |
| ---------------------------------------- | :---: | :---: |
| **BAAI/bge-small-en-v1.5**               | **1.00** | **0.950** |
| sentence-transformers/all-MiniLM-L6-v2   | 0.90  | 0.800 |

bge-small retrieved the correct page for every question and ranked it first far more
often (MRR 0.950 vs 0.800). MiniLM missed one question entirely and ranked correct
pages lower. The retrieval-tuned training and query instruction of bge clearly pay off
on this short-instruction content. The speed/size edge of MiniLM isn't worth a 15-point
MRR drop for an interactive assistant where retrieval quality drives both answer
correctness and citation accuracy.

## Effect on retrieval quality

Because the answer is only as grounded as the chunks we hand the model, embedding
quality is upstream of everything — a missed retrieval becomes either a wrong answer
or a false "not in the guide". bge's perfect hit@5 on the eval set is what lets me set
a relatively strict relevance gate (see `retrieval.md`) without over-refusing real
questions.

One property worth flagging: bge similarities are **compressed into a high band** —
even unrelated text scores ~0.5 cosine. That's fine for *ranking* (what retrieval
needs) but means an absolute score threshold is a blunt instrument, which is why
grounding leans on the model's refusal instruction rather than the score alone
(`retrieval.md`).
