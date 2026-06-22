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

from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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


class RagState(TypedDict, total=False):
    question: str
    history: list[dict]          # [{"role": "user"|"assistant", "content": str}]
    standalone_question: str
    documents: list[Document]
    top_score: float
    answer: str
    sources: list[dict]
    grounded: bool


def _to_messages(history: list[dict]):
    msgs = []
    for turn in history:
        if turn["role"] == "user":
            msgs.append(HumanMessage(turn["content"]))
        else:
            msgs.append(AIMessage(turn["content"]))
    return msgs


class RagPipeline:
    def __init__(self):
        # No temperature/sampling params — Opus 4.8 rejects them, and the default
        # is the right behaviour for grounded Q&A anyway.
        self.llm = ChatAnthropic(model=settings.chat_model, max_tokens=1024, timeout=60)
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
        # refusal and drop the citations — never cite sources for a non-answer.
        if answer == NO_CONTEXT_REPLY or not answer:
            return {"answer": NO_CONTEXT_REPLY, "sources": [], "grounded": False}
        return {"answer": answer, "sources": _sources(docs), "grounded": True}

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
        result = self.graph.invoke({"question": question, "history": history or []})
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "grounded": result.get("grounded", False),
        }


def _sources(docs: list[Document]) -> list[dict]:
    """Unique, page-ordered citations from the chunks that fed the answer."""
    seen = {}
    for doc in docs:
        meta = doc.metadata
        key = (meta.get("page"), meta.get("section"))
        if key not in seen:
            seen[key] = {
                "page": meta.get("page"),
                "section": meta.get("section", ""),
                "chapter": meta.get("chapter", ""),
            }
    return sorted(seen.values(), key=lambda s: (s["page"] or 0))
