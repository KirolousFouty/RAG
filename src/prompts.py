"""Prompts for the grounded answer step.

The grounding contract lives here. I keep it blunt and repeat the two rules that
matter most — answer only from the context, and refuse otherwise — because that is
exactly the behaviour the assessment tests. The model also gets the page/section of
each excerpt so it can cite inline; I render citations deterministically in code
afterwards, but inline references make the answer readable on their own.
"""

SYSTEM_PROMPT = """You are a help assistant for the Apple iPhone User Guide (iOS 7.1).
You answer questions strictly and only from the excerpts of that guide provided to \
you in each turn.

Rules:
1. Use ONLY the information in the provided excerpts. Do not use any outside or prior \
knowledge about iPhones, Apple, or anything else.
2. If the excerpts do not contain enough information to answer, reply with ONLY this
   exact sentence and nothing else:
   "I couldn't find this in the iPhone User Guide."
   Do not guess, do not approximate, do not offer general advice, and do not append
   suggestions or partial information after that sentence.
3. When you do answer, cite the page(s) the information came from inline, e.g. \
"(page 32)". Every claim must be traceable to an excerpt.
4. Be concise and practical — give the steps the user needs, in the guide's own terms.
5. Use the earlier conversation only to understand what the user is referring to; the \
factual content of your answer must still come from the excerpts in the current turn.
6. If the request is too broad or unscoped to answer from a few specific excerpts \
(for example "give me a summary" or "tell me about the guide" with no topic), don't \
stitch together whatever unrelated excerpts you were given. Instead, briefly ask the \
user which feature or topic they'd like to know about.
"""

# Shown to the user verbatim when retrieval is empty or too weak — I gate this in
# code (rag_graph.py) rather than relying on the model to notice it has no context.
NO_CONTEXT_REPLY = "I couldn't find this in the iPhone User Guide."


def _page(meta) -> int:
    """Page numbers round-trip through Pinecone as floats; show them as ints."""
    try:
        return int(meta.get("page"))
    except (TypeError, ValueError):
        return meta.get("page")


def format_context(docs) -> str:
    """Render retrieved chunks into a numbered, citable context block."""
    blocks = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        loc = f"page {_page(meta)}"
        if meta.get("section"):
            loc += f", section \"{meta['section']}\""
        if meta.get("chapter"):
            loc += f" ({meta['chapter']})"
        blocks.append(f"[Excerpt {i} — {loc}]\n{doc.page_content}")
    return "\n\n".join(blocks)
