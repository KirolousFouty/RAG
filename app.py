"""Chainlit UI for the grounded iPhone User Guide chatbot.

Conversation memory is per-session: I keep the running history in the Chainlit
user session and hand it to the pipeline each turn, so follow-up questions work and
the bot "remembers" what we were talking about. The pipeline call is synchronous
(model + Pinecone), so I push it onto a thread with `cl.make_async` to keep the UI
responsive.

Every grounded answer is followed by an explicit Sources block built from chunk
metadata — that is the mandatory page/section citation, surfaced in the UI.
"""

import logging
import re

import chainlit as cl

from src.config import settings
from src.rag_graph import RagPipeline

logger = logging.getLogger(__name__)

WELCOME = (
    "**iPhone User Guide assistant**\n\n"
    "Ask me anything about the iPhone User Guide (iOS 7.1). I answer only from the "
    "guide and cite the page each answer comes from. If something isn't in the guide, "
    "I'll tell you so rather than guess."
)


_RTL_CHARACTERS = re.compile(r"[\u0590-\u08FF]")
_INLINE_PAGE_CITATION = re.compile(r"\(page\s+\d+\)")


def _is_rtl(text: str) -> bool:
    return bool(_RTL_CHARACTERS.search(text))


def _format_answer_for_display(answer: str, rtl: bool) -> str:
    if not rtl:
        return answer
    # Keep the English citation as one left-to-right unit inside Arabic/Hebrew text.
    return _INLINE_PAGE_CITATION.sub(lambda match: f"\u2066{match.group(0)}\u2069", answer)


def _format_sources(sources: list[dict], rtl: bool = False) -> str:
    lines = []
    for s in sources:
        loc = f"الصفحة \u2066{s['page']}\u2069" if rtl else f"page {s['page']}"
        if s.get("section"):
            loc += f" — {s['section']}"
        if s.get("chapter"):
            loc += f" ({s['chapter']})"
        lines.append(f"- {loc}")
    heading = "**المصادر**" if rtl else "**Sources**"
    return heading + "\n" + "\n".join(lines)


@cl.on_chat_start
async def on_chat_start():
    choice = await cl.AskActionMessage(
        content="**Choose an AI provider to start this chat.**\n\n"
        "If your choice is unavailable, the other provider is tried automatically.",
        actions=[
            cl.Action(
                name="choose_provider",
                payload={"provider": "anthropic"},
                label="Anthropic (Claude)",
            ),
            cl.Action(
                name="choose_provider",
                payload={"provider": "openai"},
                label="OpenAI",
            ),
        ],
        timeout=300,
    ).send()
    if choice is None:
        await cl.Message(content="No provider was selected. Please start a new chat and choose one.").send()
        return

    provider = choice["payload"]["provider"]
    try:
        # Build the pipeline once per session. This loads the embedding model and
        # opens the Pinecone connection off the event loop.
        pipeline = await cl.make_async(RagPipeline)(provider)
    except Exception:
        logger.exception("Failed to initialise the RAG pipeline")
        await cl.Message(
            content="I couldn't start the chatbot. Check the server logs and API configuration, then start a new chat."
        ).send()
        return

    cl.user_session.set("pipeline", pipeline)
    cl.user_session.set("history", [])
    provider_label = "Anthropic (Claude)" if provider == "anthropic" else "OpenAI"
    await cl.Message(content=f"{WELCOME}\n\n**Selected provider:** {provider_label}").send()


@cl.on_message
async def on_message(message: cl.Message):
    pipeline: RagPipeline = cl.user_session.get("pipeline")
    if pipeline is None:
        await cl.Message(content="Choose a provider to start the chat before sending a question.").send()
        return
    history: list[dict] = cl.user_session.get("history")
    question = message.content

    # No placeholder message here: leaving the turn pending lets Chainlit show its
    # native animated "thinking" loader while the (synchronous) pipeline runs on a
    # thread. Sending an empty message instead would replace that with a static bubble.
    try:
        result = await cl.make_async(pipeline.answer)(question, history)
    except Exception:
        logger.exception("RAG pipeline failed while answering a message")
        await cl.Message(
            content="I couldn't complete that request because both AI providers failed. Please try again shortly."
        ).send()
        return

    answer = result["answer"]
    rtl = _is_rtl(question)
    answer = _format_answer_for_display(answer, rtl)
    if result["grounded"] and result["sources"]:
        answer = f"{answer}\n\n{_format_sources(result['sources'], rtl)}"

    await cl.Message(content=answer).send()

    # Persist the turn so the next question has the conversation as context.
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": result["answer"]})
    cl.user_session.set("history", history)


if __name__ == "__main__":
    # Convenience for `python app.py`; the documented entrypoint is the chainlit CLI.
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
