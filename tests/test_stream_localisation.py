"""Regression coverage for localized streaming responses."""

from src.rag_graph import RagPipeline


class _Chunk:
    def __init__(self, content: str):
        self.content = content


class _StreamingModel:
    active_name = "test"
    _fallback_name = "fallback"

    def set_trace(self, _trace_id):
        pass

    def set_stage(self, _stage):
        pass

    def stream(self, _messages):
        yield _Chunk("English ")
        yield _Chunk("draft")


def test_non_english_stream_emits_only_localized_answer(monkeypatch):
    """Never expose the English retrieval draft before localization completes."""
    pipeline = object.__new__(RagPipeline)
    pipeline.llm = _StreamingModel()
    pipeline._contextualize = lambda state: {"standalone_question": state["question"]}
    pipeline._split_retrieval_queries = lambda state: {"retrieval_queries": [state["question"]]}
    pipeline._subrequest_state = lambda state, question: {
        **state,
        "question": question,
        "standalone_question": question,
        "documents": [],
        "grounded": True,
    }
    pipeline._complete_answer = lambda _state, answer, **_kwargs: {
        "answer": "مسودة مترجمة",
        "sources": [],
        "grounded": True,
    }
    monkeypatch.setattr("src.rag_graph._requires_localisation", lambda _question: True)

    events = list(pipeline.stream_answer("كيف أفعل هذا؟"))

    assert [event["token"] for event in events if "token" in event] == ["مسودة مترجمة"]
    assert events[-1]["final"]["answer"] == "مسودة مترجمة"


def test_english_stream_keeps_low_latency_draft_tokens(monkeypatch):
    pipeline = object.__new__(RagPipeline)
    pipeline.llm = _StreamingModel()
    pipeline._contextualize = lambda state: {"standalone_question": state["question"]}
    pipeline._split_retrieval_queries = lambda state: {"retrieval_queries": [state["question"]]}
    pipeline._subrequest_state = lambda state, question: {
        **state,
        "question": question,
        "standalone_question": question,
        "documents": [],
        "grounded": True,
    }
    pipeline._complete_answer = lambda _state, answer, **_kwargs: {
        "answer": answer,
        "sources": [],
        "grounded": True,
    }
    monkeypatch.setattr("src.rag_graph._requires_localisation", lambda _question: False)

    events = list(pipeline.stream_answer("How do I do this?"))

    assert [event["token"] for event in events if "token" in event] == ["English ", "draft"]
