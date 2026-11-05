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
import asyncio
import uuid

import chainlit as cl
from engineio.payload import Payload

from src.config import settings
from src.rag_graph import GuideRetrievalUnavailable, RagPipeline

logger = logging.getLogger(__name__)

# These libraries log cache validation and optional-file probes at INFO/WARNING
# during the first embedding request. They are not application failures; retain
# our own pipeline logs while hiding that startup noise.
for _noisy_logger in ("httpx", "huggingface_hub", "sentence_transformers", "transformers"):
    logging.getLogger(_noisy_logger).setLevel(logging.ERROR)

# Chainlit's initial client connection can legitimately batch more than
# python-engineio's very conservative default of 16 packets. Raise the cap before
# Uvicorn starts serving requests; it remains bounded to limit request work.
_ENGINEIO_MAX_DECODE_PACKETS = 100
Payload.max_decode_packets = max(Payload.max_decode_packets, _ENGINEIO_MAX_DECODE_PACKETS)

WELCOME = (
    "**iPhone User Guide assistant**\n\n"
    "Ask me anything about the iPhone User Guide (iOS 7.1). I answer only from the "
    "guide and cite the page each answer comes from. If something isn't in the guide, "
    "I'll tell you so rather than guess."
)


_RTL_CHARACTERS = re.compile(r"[\u0590-\u08FF]")


def _is_rtl(text: str) -> bool:
    return bool(_RTL_CHARACTERS.search(text))


def _format_sources(sources: list[dict], rtl: bool = False) -> str:
    lines = []
    for s in sources:
        # Keep the deterministic citation token byte-for-byte identical in every
        # output language. The UI isolates its direction at render time.
        loc = f"(page {s['page']})"
        if s.get("section"):
            loc += f" — {s['section']}"
        if s.get("chapter"):
            loc += f" ({s['chapter']})"
        lines.append(f"- {loc}")
    heading = "**المصادر**" if rtl else "**Sources**"
    return heading + "\n" + "\n".join(lines)


def _next_stream_event(stream):
    """Return None instead of propagating StopIteration across a worker thread."""
    return next(stream, None)


async def _stream_smoothly(message: cl.Message, token: str):
    """Turn provider-sized chunks into a readable typing effect."""
    for character in token:
        await message.stream_token(character)
        await asyncio.sleep(settings.stream_char_delay)


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
    except GuideRetrievalUnavailable:
        logger.exception("guide_retrieval_unavailable_during_startup")
        await cl.Message(
            content="I couldn't access the iPhone User Guide right now. Please try again shortly."
        ).send()
        return
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
    trace_id = uuid.uuid4().hex[:12]
    logger.info("chat_request_received trace_id=%s history_turns=%d", trace_id, len(history) // 2)

    # No placeholder message here: leaving the turn pending lets Chainlit show its
    # native animated "thinking" loader while the (synchronous) pipeline runs on a
    # thread. Sending an empty message instead would replace that with a static bubble.
    try:
        stream = pipeline.stream_answer(question, history, trace_id)
        # Do the retrieval work before sending a message so Chainlit keeps showing
        # its native animated loader until the first answer token is available.
        event = await cl.make_async(_next_stream_event)(stream)
        if event is None:
            raise RuntimeError("Answer stream ended without a final result")
        response = cl.Message(content="")
        await response.send()
        while event:
            if token := event.get("token"):
                await _stream_smoothly(response, token)
            if result := event.get("final"):
                break
            event = await cl.make_async(_next_stream_event)(stream)
        else:
            raise RuntimeError("Answer stream ended without a final result")
    except GuideRetrievalUnavailable:
        logger.exception("guide_retrieval_unavailable trace_id=%s", trace_id)
        await cl.Message(
            content="I couldn't access the iPhone User Guide right now. Please try again shortly."
        ).send()
        return
    except Exception:
        logger.exception("chat_request_failed trace_id=%s", trace_id)
        await cl.Message(
            content="I couldn't complete that request because both AI providers failed. Please try again shortly."
        ).send()
        return

    answer = result["answer"]
    # This is presentation-only. Base direction follows the rendered answer, not
    # the retrieval question, so an English question requesting Arabic output is
    # displayed correctly and an English reply after an Arabic turn remains LTR.
    rtl = _is_rtl(answer)
    if result["grounded"] and result["sources"]:
        answer = f"{answer}\n\n{_format_sources(result['sources'], rtl)}"

    response.content = answer
    await response.update()

    # Persist the turn so the next question has the conversation as context.
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": result["answer"]})
    # Retain only enough untrusted conversational context to resolve a follow-up.
    # The pipeline independently applies this bound before contextualisation.
    cl.user_session.set("history", history[-settings.max_history_messages:])


if __name__ == "__main__":
    # Convenience for `python app.py`; the documented entrypoint is the chainlit CLI.
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
