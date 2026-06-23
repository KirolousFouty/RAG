# Evaluation

I evaluate this system at two levels, because "good retrieval" and "good answer" are
different things and can fail independently.

## Layer 1 — retrieval (offline, `eval/run_eval.py`)

Does the right page come back? This is the cheap, deterministic layer: chunks are
embedded in memory with bge-small and scored against a 10-question set whose expected
pages I verified by hand (`eval/questions.yaml`). No cloud, no API calls, so I could
sweep configurations in seconds. Metrics: **hit@5** (is the correct page in the top-5?)
and **MRR** (how highly is it ranked?). The full chunking and embedding comparisons —
and the numbers that picked structure-aware 800/120 + bge-small — are in
[`chunking.md`](chunking.md) and [`embeddings.md`](embeddings.md). Headline:

| What I varied             | Winner                     | hit@5 | MRR   |
| ------------------------- | -------------------------- | :---: | :---: |
| chunking (bge held fixed) | structure-aware 800/120    | 1.00  | 0.950 |
| embedding (chunks fixed)  | bge-small-en-v1.5          | 1.00  | 0.950 |

## Layer 2 — generated answer (DeepEval, `eval/deepeval_eval.py`)

Retrieval being right doesn't guarantee the *answer* is grounded — the model could
still wander off the context or miss the question. **Rana suggested in the interview
that I use DeepEval to evaluate the LLM side**, which was a good call: it gives me
answer-level metrics with an LLM judge instead of me eyeballing outputs.

DeepEval normally judges with OpenAI; since I use Anthropic everywhere else, I wrapped
the same Claude chat model as the judge (`eval/claude_judge.py`) so there's one
provider and one key. I run three metrics over a handful of in-guide questions, scoring
the real answers against the context actually retrieved from Pinecone:

- **Faithfulness** — is every claim in the answer supported by the retrieved context?
  This is the groundedness / anti-hallucination check, and the one that matters most
  for this assignment.
- **Answer Relevancy** — does the answer actually address the question asked?
- **Contextual Relevancy** — was the retrieved context itself relevant to the question?

### Results

Averaged over the four questions (Claude Opus 4.8 as judge):

| Metric                | Average | Reading |
| --------------------- | :-----: | ------- |
| Faithfulness          | **1.00** | every claim in every answer was supported by the retrieved context |
| Answer Relevancy      | 0.83    | answers address the question; the inline `(page N)` citations and the occasional caveat are counted by the metric as not-strictly-answering, which pulls it off 1.0 |
| Contextual Relevancy  | 0.20    | only a small fraction of the *retrieved* text is on-topic — see below |

**Faithfulness = 1.00 is the result I care about most.** It's direct evidence that the
answers stay inside the guide rather than drifting into the model's own knowledge —
exactly the grounding contract the bot is built around.

**Answer Relevancy = 0.83** is good and the gap is mostly a measurement artifact: the
judge treats the mandatory `(page N)` citation as a statement that doesn't answer the
question, so a perfectly good cited answer scores ~0.75 instead of 1.0. I'd rather keep
the citations and accept the metric's nitpick.

**Contextual Relevancy = 0.20 is low, and that's expected, not a bug.** This metric asks
"what fraction of the *retrieved context* is relevant to the question?". I retrieve
`top_k=5` chunks, but usually only one actually contains the answer — the other four are
near-misses that top-k always drags along. So four-fifths of the context reads as
"irrelevant" to this metric even when retrieval did its job (the right chunk is in
there, which is why Faithfulness and the hit@5 in layer 1 are perfect). The honest
takeaway is a real lever for future work: a smaller `k` or a reranking step would raise
contextual relevancy, at some risk to recall. For this assistant I deliberately favour
recall — better to hand the model a bit of extra context and let it stay faithful (which
it does, 1.00) than to retrieve too narrowly and miss the answer.

## Why both

Layer 1 is fast and free, so it drives the design decisions (chunking, embedding,
threshold). Layer 2 is slower and costs API calls, so I use it as a quality gate on the
end product — confirming that good retrieval is actually turning into grounded,
on-topic answers. Strict grounding and the explicit refusal are additionally checked by
the four-way logic test described in [`retrieval.md`](retrieval.md) and by the manual
out-of-guide questions in the README.
