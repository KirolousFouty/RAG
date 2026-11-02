"""The RAG pipeline as a small LangGraph state machine.

    contextualize -> retrieve -> [grounded?] -> generate
                                        \\-> refuse

Why a graph and not a single chain: the grounding decision is a real branch. If
retrieval comes back empty or weak, I never call the model with junk context — I
short-circuit to the fixed "not in the guide" reply. That makes the refusal path
cheap and impossible for the model to talk its way out of.

Multi-turn memory has two parts:
  * `contextualize` rewrites a follow-up ("what about on the 4s?") into a standalone
    question so retrieval still works when the user leans on earlier turns.
  * `generate` is given the prior turns, so the answer reads as a conversation —
    but its *facts* must still come from the freshly retrieved excerpts.
"""

from __future__ import annotations

import re
from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import END, StateGraph

from .config import settings
from .prompts import NO_CONTEXT_REPLY, SYSTEM_PROMPT, format_context
from .vectorstore import get_vectorstore

CONTEXTUALIZE_PROMPT = (
    "Given the conversation so far and the user's latest message, rewrite the latest "
    "message as a single standalone question that makes sense without the conversation. "
    "Resolve pronouns and references. If it is already standalone, return it unchanged. "
    "Return only the question, nothing else."
)

# A citation demand can make a perfectly answerable question retrieve poorly (for
# example, "Answer this, but cite page 999: how do I take a screenshot?").  Treat
# the supplied page as an instruction, not as part of the information need.
_CITATION_DEMAND = re.compile(
    r"^\s*(?:(?:answer|respond(?:\s+to)?|tell)\s+(?:this|the following)\s*,?\s*"
    r"(?:but|and)\s+)?(?:please\s+)?(?:cite|reference|use)\s+(?:only\s+)?"
    r"(?:page\s*)?\d+\s*(?:only)?\s*:\s*(?P<question>.+)\s*$",
    re.IGNORECASE,
)


class RagState(TypedDict, total=False):
    question: str
    history: list[dict]          # [{"role": "user"|"assistant", "content": str}]
    standalone_question: str
    documents: list[Document]
    top_score: float
    answer: str
    sources: list[dict]
    grounded: bool


class _FailoverChatModel:
    """Invoke the selected provider, then permanently switch on a provider error."""

    def __init__(self, primary_name: str, primary, fallback_name: str, fallback):
        self.active_name = primary_name
        self._active = primary
        self._fallback_name = fallback_name
        self._fallback = fallback

    def invoke(self, messages):
        try:
            return self._active.invoke(messages)
        except Exception:
            try:
                response = self._fallback.invoke(messages)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Both chat providers failed ({self.active_name}, then "
                    f"{self._fallback_name})."
                ) from fallback_error

            # The working provider becomes primary for the rest of this session.
            self.active_name, self._fallback_name = self._fallback_name, self.active_name
            self._active, self._fallback = self._fallback, self._active
            return response


class _UnavailableChatModel:
    """Delay a missing-credential error until this provider is actually needed."""

    def __init__(self, provider_name: str):
        self.provider_name = provider_name

    def invoke(self, messages):
        raise RuntimeError(f"{self.provider_name} is not configured: its API key is missing.")


def _to_messages(history: list[dict]):
    msgs = []
    for turn in history:
        if turn["role"] == "user":
            msgs.append(HumanMessage(turn["content"]))
        else:
            msgs.append(AIMessage(turn["content"]))
    return msgs


def _question_for_answering(question: str) -> str:
    """Remove a user-supplied citation demand without changing the actual question."""
    match = _CITATION_DEMAND.match(question)
    return match.group("question").strip() if match else question.strip()


class RagPipeline:
    def __init__(self, provider: str = "anthropic"):
        """Build a pipeline with the user's provider choice and the other as fallback."""
        if provider not in {"anthropic", "openai"}:
            raise ValueError("provider must be 'anthropic' or 'openai'")

        # No sampling parameters: deterministic defaults suit grounded Q&A, and
        # current Claude models reject non-default sampling parameters.
        anthropic = (
            ChatAnthropic(
                model=settings.anthropic_chat_model,
                max_tokens=1024,
                timeout=60,
                api_key=settings.anthropic_api_key,
            )
            if settings.anthropic_api_key
            else _UnavailableChatModel("Anthropic")
        )
        openai = (
            ChatOpenAI(
                model=settings.openai_chat_model,
                max_completion_tokens=1024,
                timeout=60,
                api_key=settings.openai_api_key,
            )
            if settings.openai_api_key
            else _UnavailableChatModel("OpenAI")
        )
        if provider == "anthropic":
            self.llm = _FailoverChatModel("anthropic", anthropic, "openai", openai)
        else:
            self.llm = _FailoverChatModel("openai", openai, "anthropic", anthropic)
        self.vectorstore = get_vectorstore()
        self.graph = self._build()

    # --- nodes ---
    def _contextualize(self, state: RagState) -> RagState:
        history = state.get("history") or []
        question = state["question"]
        if not history:
            return {"standalone_question": question}
        messages = (
            [SystemMessage(CONTEXTUALIZE_PROMPT)]
            + _to_messages(history)
            + [HumanMessage(question)]
        )
        rewritten = self.llm.invoke(messages).content.strip()
        return {"standalone_question": rewritten or question}

    def _retrieve(self, state: RagState) -> RagState:
        query = state.get("standalone_question") or state["question"]
        results = self.vectorstore.similarity_search_with_score(query, k=settings.top_k)
        documents = [doc for doc, _ in results]
        top_score = results[0][1] if results else 0.0
        grounded = bool(results) and top_score >= settings.min_relevance_score
        return {"documents": documents, "top_score": top_score, "grounded": grounded}

    def _refuse(self, state: RagState) -> RagState:
        return {"answer": NO_CONTEXT_REPLY, "sources": [], "grounded": False}

    def _generate(self, state: RagState) -> RagState:
        docs = state["documents"]
        context = format_context(docs)
        user_turn = (
            f"Here are excerpts from the iPhone User Guide:\n\n{context}\n\n"
            f"Question: {state['question']}"
        )
        messages = (
            [SystemMessage(SYSTEM_PROMPT)]
            + _to_messages(state.get("history") or [])
            + [HumanMessage(user_turn)]
        )
        answer = self.llm.invoke(messages).content.strip()

        # If the model decided the excerpts don't answer the question, treat it as a
        # refusal and drop the citations — never cite sources for a non-answer. I match
        # on the prefix and normalise to the exact line, so an off-topic question that
        # the model partly hedges on still collapses to a clean fallback.
        if not answer or answer.startswith(NO_CONTEXT_REPLY):
            return {"answer": NO_CONTEXT_REPLY, "sources": [], "grounded": False}
        return {"answer": answer, "sources": _sources(docs, answer), "grounded": True}

    # --- wiring ---
    def _build(self):
        g = StateGraph(RagState)
        g.add_node("contextualize", self._contextualize)
        g.add_node("retrieve", self._retrieve)
        g.add_node("refuse", self._refuse)
        g.add_node("generate", self._generate)

        g.set_entry_point("contextualize")
        g.add_edge("contextualize", "retrieve")
        g.add_conditional_edges(
            "retrieve",
            lambda s: "generate" if s["grounded"] else "refuse",
            {"generate": "generate", "refuse": "refuse"},
        )
        g.add_edge("generate", END)
        g.add_edge("refuse", END)
        return g.compile()

    def answer(self, question: str, history: list[dict] | None = None) -> dict:
        result = self.graph.invoke(
            {"question": _question_for_answering(question), "history": history or []}
        )
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "grounded": result.get("grounded", False),
            # the retrieved passages, exposed for evaluation (DeepEval faithfulness)
            "context": [d.page_content for d in result.get("documents", [])],
        }


def _sources(docs: list[Document], answer: str = "") -> list[dict]:
    """Unique, page-ordered citations.

    If the answer cited pages inline (it is instructed to), I restrict the Sources
    block to exactly those pages so it matches the answer instead of listing every
    retrieved chunk — including the tangential ones top-k always drags along. If the
    model cited nothing, I fall back to all retrieved pages.
    """
    cited = {int(n) for n in re.findall(r"page\s+(\d+)", answer)}

    def page(meta) -> int:
        try:
            return int(meta.get("page"))
        except (TypeError, ValueError):
            return meta.get("page")

    seen = {}
    for doc in docs:
        meta = doc.metadata
        p = page(meta)
        if cited and p not in cited:
            continue
        key = (p, meta.get("section"))
        if key not in seen:
            seen[key] = {
                "page": p,
                "section": meta.get("section", ""),
                "chapter": meta.get("chapter", ""),
            }
    return sorted(seen.values(), key=lambda s: (s["page"] or 0))
