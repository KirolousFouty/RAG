"""Regression coverage for localized streaming responses."""

from src.rag_graph import RagPipeline


class _StreamingModel:
    active_name = "test"

    def set_trace(self, _trace_id):
        pass

    def set_stage(self, _stage):
        pass

def test_stream_emits_only_validated_localized_answer(monkeypatch):
    """The UI receives only a completed validated result, never an LLM draft."""
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
    pipeline._generate = lambda _state: {
        "answer": "مسودة مترجمة",
        "sources": [],
        "grounded": True,
    }
    events = list(pipeline.stream_answer("كيف أفعل هذا؟"))

    assert [event["token"] for event in events if "token" in event] == ["مسودة مترجمة"]
    assert events[-1]["final"]["answer"] == "مسودة مترجمة"


def test_stream_emits_only_completed_validated_answer(monkeypatch):
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
    pipeline._generate = lambda _state: {
        "answer": "Validated answer",
        "sources": [],
        "grounded": True,
    }
    events = list(pipeline.stream_answer("How do I do this?"))

    assert [event["token"] for event in events if "token" in event] == ["Validated answer"]
