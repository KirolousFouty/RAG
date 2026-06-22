# Retrieval, grounding & citations

This covers how the running app turns a question into a grounded, cited answer, and
how it refuses when it should. The pipeline is a small LangGraph state machine
(`src/rag_graph.py`):

```
contextualize -> retrieve -> [grounded?] --yes--> generate -> answer + sources
                                  \--no--> refuse -> "I couldn't find this..."
```

## Retrieval

For each turn I embed the (contextualised) question with bge-small and run a top-5
cosine similarity search against Pinecone. `k=5` was enough for hit@5 = 1.00 on the
eval set, so going wider would only add noise (and tokens) to the model's context.

## Multi-turn memory

Memory is per-session and has two distinct jobs, which I keep separate on purpose:

- **`contextualize`** rewrites a follow-up into a standalone question *for retrieval*.
  "What about on the 4s?" embeds terribly on its own; rewritten to "Does Touch ID work
  on the iPhone 4s?" it retrieves the right page. This is a cheap LLM rewrite that only
  runs when there's prior history.
- **`generate`** is given the prior turns so the *answer* reads as a conversation. But
  the system prompt is explicit that earlier turns are only for understanding what the
  user means — the facts must still come from the excerpts retrieved this turn.

The Chainlit app holds the history in the user session and passes it in each turn, so
the bot "remembers" within a session without any server-side state.

## Strict grounding — two layers

The bot must answer only from the guide and say so otherwise. I don't trust a single
mechanism for this; there are two layers, and either one alone would be weaker:

**Layer 1 — a relevance gate, before the model is even called.** If retrieval is empty
or the best cosine score is below `MIN_RELEVANCE_SCORE`, the graph branches straight to
`refuse` and returns the fixed message. No model call, no chance for the model to
improvise.

Picking the threshold needed real numbers, because bge similarities sit in a high,
compressed band. I measured top-1 cosine for in-scope and out-of-scope questions:

| Query type                                   | top-1 cosine (range) |
| -------------------------------------------- | -------------------- |
| in-scope, direct ("open Control Center")     | 0.71 – 0.80          |
| in-scope, paraphrased ("my screen is too dark") | 0.65 – 0.75       |
| out-of-scope ("who won the World Cup?")      | 0.47 – 0.64          |

The bands **overlap** (paraphrased in-scope bottoms out ~0.648; out-of-scope tops
~0.639), so a clean separating threshold doesn't exist. I set the gate at **0.55** — low
enough not to false-refuse a real, oddly-worded question, high enough to drop the
blatantly irrelevant (capital of France, World Cup). It is a coarse pre-filter, not the
whole grounding story.

**Layer 2 — the model's refusal instruction, the authoritative check.** The system
prompt (`src/prompts.py`) tells the model, in plain terms, to use only the excerpts and
to reply with the exact sentence *"I couldn't find this in the iPhone User Guide."* when
they don't answer the question. This is what catches the overlap region: a question
about baking bread scores ~0.575 (above the gate), retrieves some iPhone text, and the
model correctly refuses because that text doesn't answer it. When the model returns the
refusal sentence, the graph drops any citations — I never cite sources for a non-answer.

I validated all four paths offline with stubbed retrieval and model: strong match →
grounded + cited; weak score → hard refuse (no model call); gate-passing but
unanswerable → model refuse + citations dropped; empty retrieval → refuse.

## Citations

Every grounded answer carries its sources two ways:

1. **Inline**, because the system prompt asks the model to cite the page next to each
   claim, e.g. "Turn on Do Not Disturb in Settings (page 32)."
2. **A deterministic "Sources" block** appended in code (`app.py`) from the chunk
   metadata — unique, page-ordered `page — section (chapter)` lines. This doesn't rely
   on the model remembering to cite; it's built from the exact chunks that fed the
   answer, so it's always present and always accurate.

The page/section in that block is the same metadata the parser pulled from the footer
and headings, carried untouched through ingestion — so a citation always points at the
page a reader can actually turn to.
