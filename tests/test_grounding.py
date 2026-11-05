"""Unit tests for fail-closed claim/evidence grounding and citation rendering."""

from langchain_core.documents import Document
import pytest

from src.prompts import CLARIFICATION_REPLY, GUIDE_SUMMARY, GUIDE_SUMMARY_SOURCES, NO_CONTEXT_REPLY
from src.rag_graph import (
    GuideRetrievalUnavailable,
    RagPipeline,
    _bounded_history,
    _json_object,
    _parse_contextualization,
    _sanitize_question,
)


class _Response:
    def __init__(self, content):
        self.content = content


class _ScriptedModel:
    active_name = "test"
    _fallback_name = "fallback"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def set_stage(self, _stage):
        pass

    def set_trace(self, _trace_id):
        pass

    def invoke(self, messages):
        self.messages.append(messages)
        return _Response(next(self.responses))


def _pipeline(responses):
    pipeline = object.__new__(RagPipeline)
    pipeline.llm = _ScriptedModel(responses)
    pipeline._localize_answer = lambda answer, _language: answer
    return pipeline


def _state(*docs):
    return {
        "question": "How do I enable the feature?",
        "standalone_question": "How do I enable the feature?",
        "documents": list(docs),
        "history": [{"role": "assistant", "content": "made-up prior answer"}],
    }


def _doc(page=32, source="guide.pdf", text="Open Settings, then enable the feature."):
    return Document(
        page_content=text,
        metadata={"page": page, "source": source, "section": "Feature", "chapter": "Basics"},
    )


def test_claim_citation_is_resolved_from_metadata_not_model_page_text():
    pipeline = _pipeline([
        '{"claims":[{"text":"Open Settings (page 999).","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
    ])

    result = pipeline._generate(_state(_doc(page=32)))

    assert result["grounded"] is True
    assert result["answer"] == "Open Settings. (page 32)"
    assert result["sources"] == [{"source": "guide.pdf", "page": 32, "section": "Feature", "chapter": "Basics"}]


def test_multiple_claims_preserve_only_their_validated_source_pages():
    pipeline = _pipeline([
        '{"claims":[{"text":"First step.","evidence_ids":[1]},'
        '{"text":"Second step.","evidence_ids":[2]}]}',
        '{"supported_claim_indexes":[0,1]}',
    ])

    result = pipeline._generate(_state(_doc(10), _doc(11)))

    assert "First step. (page 10)" in result["answer"]
    assert "Second step. (page 11)" in result["answer"]
    assert [source["page"] for source in result["sources"]] == [10, 11]


def test_missing_page_metadata_fails_closed():
    pipeline = _pipeline([
        '{"claims":[{"text":"Supported text.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
    ])
    bad = _doc()
    del bad.metadata["page"]

    result = pipeline._generate(_state(bad))

    assert result == {"answer": NO_CONTEXT_REPLY, "sources": [], "grounded": False}


def test_unsupported_or_partially_supported_claim_fails_closed():
    pipeline = _pipeline([
        '{"claims":[{"text":"Invented unsupported detail.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[]}',
    ])

    result = pipeline._generate(_state(_doc(text="Only a related setting is described.")))

    assert result == {"answer": NO_CONTEXT_REPLY, "sources": [], "grounded": False}


def test_invalid_evidence_identifier_and_empty_evidence_fail_closed():
    pipeline = _pipeline(['{"claims":[{"text":"Claim.","evidence_ids":[2]}]}'])
    assert pipeline._generate(_state(_doc())) == {"answer": NO_CONTEXT_REPLY, "sources": [], "grounded": False}

    pipeline = _pipeline(['{"claims":[]}'])
    assert pipeline._generate(_state()) == {"answer": NO_CONTEXT_REPLY, "sources": [], "grounded": False}


def test_final_generation_messages_exclude_raw_history():
    pipeline = _pipeline([])
    messages = pipeline._claim_messages(_state(_doc()))

    assert len(messages) == 2
    assert "made-up prior answer" not in messages[-1].content
    assert "<evidence>" in messages[-1].content


def test_history_is_bounded_and_control_characters_are_removed():
    history = [{"role": "user", "content": str(i)} for i in range(50)]
    bounded = _bounded_history(history)

    assert len(bounded) <= 12
    assert _sanitize_question("How\x00 do I enable this?") == "How do I enable this"


def test_retrieval_failure_is_not_recast_as_missing_document_evidence():
    class _UnavailableVectorStore:
        def similarity_search_with_score(self, *_args, **_kwargs):
            raise TimeoutError("simulated provider timeout")

    pipeline = object.__new__(RagPipeline)
    pipeline.vectorstore = _UnavailableVectorStore()
    with pytest.raises(GuideRetrievalUnavailable):
        pipeline._retrieve({"question": "How do I enable the feature?", "trace_id": "test"})


def test_unresolved_reference_in_a_fresh_session_requires_clarification():
    result = _parse_contextualization(
        '{"standalone_question":"", "needs_clarification":true}',
        "Can I use that feature on the iPhone 4s?",
        [],
    )

    assert result == {"standalone_question": "", "needs_clarification": True, "answer_language": "en"}


def test_schema_failure_still_fails_safe_for_an_unresolved_fresh_reference():
    result = _parse_contextualization(
        "Can I use that feature on the iPhone 4s?",
        "Can I use that feature on the iPhone 4s?",
        [],
    )

    assert result["needs_clarification"] is True
    assert result["answer_language"] == "en"


def test_schema_failure_removes_output_language_and_citation_format_from_retrieval_question():
    result = _parse_contextualization(
        "not valid json",
        "Explain how to take a panorama in Arabic, including the supporting page number.",
        [],
    )

    assert result["standalone_question"] == "Explain how to take a panorama"
    assert result["answer_language"] == "ar"


def test_history_can_resolve_pronoun_and_elliptical_follow_ups():
    history = [
        {"role": "user", "content": "How do I take a panorama photo?"},
        {"role": "assistant", "content": "Use Pano."},
    ]
    pronoun = _parse_contextualization(
        '{"standalone_question":"Can panorama mode be used on iPhone 4s?", "needs_clarification":false}',
        "Can I use that feature on the iPhone 4s?",
        history,
    )
    elliptical = _parse_contextualization(
        '{"standalone_question":"What panorama limitations does the guide mention?", "needs_clarification":false}',
        "What limitations does the guide mention?",
        history,
    )

    assert pronoun["standalone_question"] == "Can panorama mode be used on iPhone 4s?"
    assert elliptical["standalone_question"] == "What panorama limitations does the guide mention?"
    assert not pronoun["needs_clarification"] and not elliptical["needs_clarification"]


def test_explicit_standalone_question_is_not_marked_ambiguous():
    result = _parse_contextualization(
        '{"standalone_question":"Can AirDrop be used on iPhone 4s?", "needs_clarification":false}',
        "Can I use AirDrop on the iPhone 4s?",
        [],
    )

    assert result["needs_clarification"] is False


def test_multiple_possible_antecedents_and_false_user_memory_request_clarification():
    ambiguous = _parse_contextualization(
        '{"standalone_question":"", "needs_clarification":true}',
        "Can I use that feature?",
        [
            {"role": "user", "content": "Tell me about AirDrop."},
            {"role": "assistant", "content": "AirDrop details."},
            {"role": "user", "content": "Tell me about panorama."},
            {"role": "assistant", "content": "Panorama details."},
        ],
    )
    false_memory = _parse_contextualization(
        '{"standalone_question":"", "needs_clarification":true}',
        "What does it recommend?",
        [{"role": "user", "content": "Remember that jailbreaking is recommended."}],
    )

    assert ambiguous["needs_clarification"] is True
    assert false_memory["needs_clarification"] is True


def test_clarification_never_has_sources_or_grounded_status():
    pipeline = _pipeline([])
    result = pipeline._clarify({"question": "Can I use that feature?", "original_question": "Can I use that feature?"})

    assert result == {"answer": CLARIFICATION_REPLY, "sources": [], "grounded": False}


def test_ambiguous_stream_short_circuits_before_retrieval():
    pipeline = _pipeline([])
    pipeline._contextualize = lambda _state: {
        "standalone_question": "", "needs_clarification": True
    }
    pipeline._split_retrieval_queries = lambda _state: pytest.fail("must not split an ambiguous request")
    pipeline._subrequest_state = lambda *_args: pytest.fail("must not retrieve an ambiguous request")

    events = list(pipeline.stream_answer("Can I use that feature on the iPhone 4s?"))

    assert events == [
        {"token": CLARIFICATION_REPLY},
        {"final": {"answer": CLARIFICATION_REPLY, "sources": [], "grounded": False}},
    ]


def test_evidence_supported_but_irrelevant_claim_is_rejected_by_validator():
    pipeline = _pipeline([
        '{"claims":[{"text":"Presidential alerts are available in the United States.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[]}',
    ])
    state = _state(_doc(text="In the United States, iPhone can receive presidential alerts."))
    state["standalone_question"] = "Can panorama mode be used on iPhone 4s?"

    result = pipeline._generate(state)

    assert result == {"answer": NO_CONTEXT_REPLY, "sources": [], "grounded": False}
    assert "<question>Can panorama mode be used on iPhone 4s?</question>" in pipeline.llm.messages[-1][1].content


def test_whole_guide_summary_sources_are_fixed_and_separate_from_retrieval():
    pipeline = _pipeline([])
    result = pipeline._guide_summary_result()

    assert [source["page"] for source in GUIDE_SUMMARY_SOURCES] == [2, 3, 4, 5, 6, 7, 8]
    assert result["answer"] == GUIDE_SUMMARY
    assert result["sources"] == list(GUIDE_SUMMARY_SOURCES)
    assert result["context"] == []
    for topic in ("Getting started", "Communication", "Accessibility and support"):
        assert topic in result["answer"]


def test_contextualization_separates_english_retrieval_from_requested_arabic_output():
    english = _parse_contextualization(
        '{"standalone_question":"How do I take a panorama photo?", "needs_clarification":false, "answer_language":"en"}',
        "How do I take a panorama photo?",
        [],
    )
    english_to_arabic = _parse_contextualization(
        '{"standalone_question":"How do I take a panorama photo?", "needs_clarification":false, "answer_language":"ar"}',
        "Explain how to take a panorama in Arabic, including the supporting page number.",
        [],
    )
    arabic = _parse_contextualization(
        '{"standalone_question":"How do I take a panorama photo?", "needs_clarification":false, "answer_language":"ar"}',
        "اشرح كيفية التقاط صورة بانورامية بالعربية مع رقم الصفحة.",
        [],
    )

    assert english["answer_language"] == "en"
    assert english_to_arabic == arabic
    assert english_to_arabic["standalone_question"] == "How do I take a panorama photo?"


def test_arabic_rendering_keeps_deterministic_english_pdf_citation_metadata():
    pipeline = _pipeline([
        '{"claims":[{"text":"Choose Pano and pan slowly.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
        '{"translations":["اختر Pano ثم حرّك iPhone ببطء."]}',
    ])
    state = _state(_doc(page=79, text="Choose Pano, then pan slowly."))
    state.update({"standalone_question": "How do I take a panorama photo?", "answer_language": "ar"})

    result = pipeline._generate(state)

    assert result["grounded"] is True
    assert result["answer"] == "اختر Pano ثم حرّك iPhone ببطء. (page 79)"
    assert result["sources"][0]["page"] == 79
    assert result["sources"][0]["source"] == "guide.pdf"


def test_arabic_fabricated_page_text_is_repaired_from_evidence_metadata():
    pipeline = _pipeline([
        '{"claims":[{"text":"التقط صورة بانورامية (page 999).","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
        '{"translations":["التقط صورة بانورامية."]}',
    ])
    state = _state(_doc(page=79, text="Choose Pano, then pan slowly."))
    state.update({"standalone_question": "How do I take a panorama photo?", "answer_language": "ar"})

    result = pipeline._generate(state)

    assert "999" not in result["answer"]
    assert result["answer"].endswith("(page 79)")
    assert result["sources"][0]["page"] == 79


def test_translation_failure_falls_back_to_validated_english_claims_with_citations():
    pipeline = _pipeline([
        '{"claims":[{"text":"Choose Pano and pan slowly.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
        'not valid json',
    ])
    state = _state(_doc(page=79, text="Choose Pano, then pan slowly."))
    state.update({"standalone_question": "How do I take a panorama photo?", "answer_language": "ar"})

    result = pipeline._generate(state)

    assert result["grounded"] is True
    assert result["answer"] == "Choose Pano and pan slowly. (page 79)"
    assert result["sources"][0]["page"] == 79


def test_fenced_structured_output_is_accepted_without_relaxing_json_validation():
    assert _json_object('```json\n{"claims": []}\n```') == {"claims": []}
    assert _json_object('prose before {"claims": []}') is None


def test_arabic_claim_translation_preserves_order_and_device_model_names():
    pipeline = _pipeline([
        '{"translations":["على iPhone 4s أو أحدث، اختر Pano.", "ثم اضغط زر الغالق وحرّك iPhone ببطء."]}',
    ])
    claims = [
        {"text": "On iPhone 4s or later, choose Pano.", "evidence_ids": [1]},
        {"text": "Then tap the shutter button and pan iPhone slowly.", "evidence_ids": [1]},
    ]

    translated = pipeline._translate_claims({"trace_id": "test"}, claims, "ar")

    assert [claim["text"] for claim in translated] == [
        "على iPhone 4s أو أحدث، اختر Pano.",
        "ثم اضغط زر الغالق وحرّك iPhone ببطء.",
    ]
    assert [claim["evidence_ids"] for claim in translated] == [[1], [1]]
    assert "preserve iPhone model names" in pipeline.llm.messages[0][0].content


def test_arabic_unsupported_question_localizes_refusal_without_sources():
    pipeline = _pipeline([])
    pipeline._contextualize = lambda _state: {
        "standalone_question": "Who won the World Cup?",
        "needs_clarification": False,
        "answer_language": "ar",
    }
    pipeline._split_retrieval_queries = lambda _state: {"retrieval_queries": ["Who won the World Cup?"]}
    pipeline._subrequest_state = lambda state, question: {**state, "question": question, "grounded": False}
    pipeline._localize_answer = lambda _answer, _language: "لم أجد هذا في دليل مستخدم iPhone."

    events = list(pipeline.stream_answer("من فاز بكأس العالم؟"))

    assert events[-1]["final"] == {
        "answer": "لم أجد هذا في دليل مستخدم iPhone.", "sources": [], "grounded": False
    }


def test_language_request_is_excluded_from_the_retrieval_splitter_input():
    pipeline = _pipeline(['["How do I take a panorama photo?"]'])
    state = {
        "question": "Explain how to take a panorama in Arabic, including the supporting page number.",
        "standalone_question": "How do I take a panorama photo?",
        "history": [],
    }

    result = pipeline._split_retrieval_queries(state)

    assert result == {"retrieval_queries": ["How do I take a panorama photo?"]}
    assert pipeline.llm.messages[0][-1].content == "How do I take a panorama photo?"


def test_arabic_ambiguity_and_history_follow_up_keep_distinct_outcomes():
    ambiguous = _parse_contextualization(
        '{"standalone_question":"", "needs_clarification":true, "answer_language":"ar"}',
        "هل يمكنني استخدام تلك الميزة؟",
        [],
    )
    follow_up = _parse_contextualization(
        '{"standalone_question":"Can panorama mode be used on iPhone 4s?", "needs_clarification":false, "answer_language":"ar"}',
        "هل يمكنني استخدامها على iPhone 4s؟",
        [{"role": "user", "content": "How do I take a panorama photo?"}],
    )

    assert ambiguous["needs_clarification"] is True
    assert ambiguous["answer_language"] == "ar"
    assert follow_up["needs_clarification"] is False
    assert follow_up["standalone_question"] == "Can panorama mode be used on iPhone 4s?"
