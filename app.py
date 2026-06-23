"""Chainlit UI for the grounded iPhone User Guide chatbot.

Conversation memory is per-session: I keep the running history in the Chainlit
user session and hand it to the pipeline each turn, so follow-up questions work and
the bot "remembers" what we were talking about. The pipeline call is synchronous
(model + Pinecone), so I push it onto a thread with `cl.make_async` to keep the UI
responsive.

Every grounded answer is followed by an explicit Sources block built from chunk
metadata — that is the mandatory page/section citation, surfaced in the UI.
"""

import chainlit as cl

from src.config import settings
from src.rag_graph import RagPipeline

WELCOME = (
    "**iPhone User Guide assistant**\n\n"
    "Ask me anything about the iPhone User Guide (iOS 7.1). I answer only from the "
    "guide and cite the page each answer comes from. If something isn't in the guide, "
    "I'll tell you so rather than guess."
)


def _format_sources(sources: list[dict]) -> str:
    lines = []
    for s in sources:
        loc = f"page {s['page']}"
        if s.get("section"):
            loc += f" — {s['section']}"
        if s.get("chapter"):
            loc += f" ({s['chapter']})"
        lines.append(f"- {loc}")
    return "**Sources**\n" + "\n".join(lines)


@cl.on_chat_start
async def on_chat_start():
    # Build the pipeline once per session: this loads the embedding model and opens
    # the Pinecone connection. Kept off the event loop so the UI shows immediately.
    pipeline = await cl.make_async(RagPipeline)()
    cl.user_session.set("pipeline", pipeline)
    cl.user_session.set("history", [])
    await cl.Message(content=WELCOME).send()


@cl.on_message
async def on_message(message: cl.Message):
    pipeline: RagPipeline = cl.user_session.get("pipeline")
    history: list[dict] = cl.user_session.get("history")
    question = message.content

    thinking = cl.Message(content="")
    await thinking.send()

    result = await cl.make_async(pipeline.answer)(question, history)

    answer = result["answer"]
    if result["grounded"] and result["sources"]:
        answer = f"{answer}\n\n{_format_sources(result['sources'])}"

    thinking.content = answer
    await thinking.update()

    # Persist the turn so the next question has the conversation as context.
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": result["answer"]})
    cl.user_session.set("history", history)


if __name__ == "__main__":
    # Convenience for `python app.py`; the documented entrypoint is the chainlit CLI.
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
