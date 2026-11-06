# Retrieval, grounding & citations

This covers how the running app turns a question into a grounded, cited answer, and
how it refuses when it should. The pipeline is implemented in `src/rag_graph.py`.
For one ordinary information need, its core path is:

```
sanitize -> contextualize -> retrieve -> [relevant?] --no--> refuse
                                             |
                                            yes
                                             v
                              generate claims + evidence IDs -> validate claims
                                                              |             |
                                                             pass          fail
                                                              |             |
                                                              v             v
                                             render citations from metadata  refuse
```

Guide-wide summary requests take a separate path: rather than pretending that a
top-5 search represents the manual, the app supplies all extracted, page-tagged PDF
segments to a summary pass and accepts only claims with valid source pages.

## Retrieval

For each information need, the app embeds the contextualised English question with
bge-small and runs a top-5 cosine similarity search against Pinecone. `k=5` was enough
for hit@5 = 1.00 on the retrieval eval set, so going wider would only add noise (and
tokens) to the model's context. A small terminology layer adds guide terms for a few
high-value synonyms (for example, “background” also searches for “wallpaper”), and
chapter-number questions include the corresponding chapter title.

Multi-part requests are not embedded as one blended question. The contextualizer can
return up to five independent information needs; otherwise a bounded splitter preserves
explicit task boundaries and asks the model to decompose visibly compound requests.
Each need is retrieved, generated, validated, and retried in isolation. The final
answer combines supported parts in user-request order and clearly marks any part that
lacks support. A request about whether two features interact includes a separate
interaction query, so evidence for each feature alone is never treated as evidence of
their relationship.

## Multi-turn memory

Memory is per-session and has one deliberately limited job:

- **`contextualize`** rewrites a follow-up into a standalone question *for retrieval*.
  "What about on the 4s?" embeds terribly on its own; rewritten to "Does Touch ID work
  on the iPhone 4s?" it retrieves the right page. The same bounded classification step
  also normalises a first-turn question and detects ambiguity.
- Answer generation and validation receive the resolved question and freshly retrieved
  excerpts, not raw conversation history. Prior turns can identify the topic but
  cannot become factual evidence.

The Chainlit app keeps a bounded recent history in the user session, so the bot
"remembers" within a session without persistent server-side conversation storage. User
text in both the current message and history is sanitised before it reaches a model;
prompt-control clauses are removed while a legitimate guide question in the same
message is retained. If a follow-up has no clear antecedent, the pipeline asks which
feature the user means instead of guessing.

## Strict grounding — three layers

The bot must answer only from the guide and say so otherwise. It does not rely on a
single instruction-following step; three checks cover different failure modes:

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

**Layer 2 — evidence-linked claim generation.** The answer model must return either an
explicit structured refusal or a small set of claims, each paired with one or more
retrieved excerpt IDs. It receives only the current excerpts and resolved question.
This catches the overlap region: a question about baking bread can score above the
coarse gate and retrieve iPhone text, but the model should refuse because that text
does not answer it.

**Layer 3 — independent claim validation.** A second model call checks every proposed
claim against its cited excerpts *and* the resolved question. Every claim must pass;
malformed output, an unsupported claim, or a partial validation result becomes a
refusal. The application, not the model, renders the refusal message and drops
citations for every non-answer.

Explicit iPhone model references also receive a deterministic scope check before
retrieval: this iOS 7.1 guide covers iPhone 4, 4s, 5, 5c, and 5s. This prevents a
semantically similar passage for a supported device from answering a question about an
unsupported model. Temporary contextualization or validation failures are retried a
bounded number of times, always against the same retrieved evidence.

## Citations

Every grounded answer carries its sources two ways, both derived after validation:

1. **Inline** page citations are appended to each approved claim from the pages in its
   validated evidence IDs, e.g. `Turn on Do Not Disturb in Settings (page 32)`.
   Provider-written page numbers are stripped and never trusted.
2. **A deterministic "Sources" block** is appended in `app.py` from the metadata of
   those approved evidence chunks — unique, page-ordered `page — section (chapter)`
   lines. Retrieved chunks that did not support a rendered claim are omitted.

The page/section in that block is the same metadata the parser pulled from the footer
and headings, carried untouched through ingestion — so a citation always points at the
page a reader can actually turn to.

## Language and delivery

The contextualizer translates the information need to English for retrieval while
retaining the requested answer language. After claim validation, claim text can be
translated before citations are attached, which keeps citation tokens deterministic.
Arabic has dedicated refusal and partial-answer templates; other languages use a
bounded translation fallback. The UI detects right-to-left answers and labels the
Sources block in Arabic when appropriate.

Streaming is presentation-only: Chainlit displays the final, fully validated answer
with a character-by-character effect, never an unvalidated provider draft.
