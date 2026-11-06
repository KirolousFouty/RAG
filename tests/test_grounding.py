"""Unit tests for fail-closed claim/evidence grounding and citation rendering."""

from langchain_core.documents import Document
from dataclasses import replace
import json
import pytest

import src.rag_graph as rag_graph
from src.prompts import CLAIM_VALIDATION_PROMPT, CLARIFICATION_REPLY, NO_CONTEXT_REPLY
from src.pdf_loader import Segment
from src.rag_graph import (
    _OpenAIChatModel,
    GuideRetrievalUnavailable,
    RagPipeline,
    _bounded_history,
    _json_object,
    _normalise_retrieval_question,
    _parse_contextualization,
    _parse_retrieval_queries,
    _requires_model_split,
    _sanitize_question,
    _split_fallback_queries,
)


class _Response:
    def __init__(self, content):
        self.content = content


class _ScriptedModel:
    active_name = "test"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def set_stage(self, _stage):
        pass

    def set_trace(self, _trace_id):
        pass

    def invoke(self, messages, **_kwargs):
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


def test_openai_model_uses_provider_default_temperature(monkeypatch):
    """Models that only accept their default temperature must remain callable."""
    created = {}

    class _FakeChatModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_openai(**kwargs):
        created["openai"] = kwargs
        return _FakeChatModel(**kwargs)

    monkeypatch.setattr(rag_graph, "ChatOpenAI", fake_openai)
    monkeypatch.setattr(rag_graph, "get_vectorstore", lambda: object())
    monkeypatch.setattr(RagPipeline, "_build", lambda _self: object())
    monkeypatch.setattr(rag_graph, "settings", replace(rag_graph.settings, openai_api_key="test-key"))

    RagPipeline()

    assert "temperature" not in created["openai"]
    assert created["openai"]["max_completion_tokens"] == rag_graph.settings.openai_max_completion_tokens
    assert created["openai"]["reasoning_effort"] == rag_graph.settings.openai_reasoning_effort


def test_openai_uses_json_mode_only_for_object_shaped_structured_stages():
    class _Model:
        def __init__(self):
            self.bound = []

        def bind(self, **kwargs):
            self.bound.append(kwargs)
            return self

        def invoke(self, _messages):
            return _Response('{"claims": []}')

    openai = _Model()
    model = _OpenAIChatModel(openai)

    model.set_stage("generate_claims")
    model.invoke([])
    model.set_stage("split_retrieval_queries")
    model.invoke([])

    assert openai.bound == [{"response_format": _OpenAIChatModel._CLAIMS_RESPONSE_FORMAT}]


def test_validator_prompt_accepts_concise_directly_supported_procedural_claims_without_relaxing_entailment():
    assert "every stated step is expressly supported" in CLAIM_VALIDATION_PROMPT
    assert "adds, changes, contradicts" in CLAIM_VALIDATION_PROMPT


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


def test_explicit_structured_refusal_fails_closed():
    pipeline = _pipeline(['{"claims":null,"refusal":true}'])

    assert pipeline._generate(_state(_doc())) == {
        "answer": NO_CONTEXT_REPLY,
        "sources": [],
        "grounded": False,
    }


def test_claim_validation_failure_sets_a_reason_when_retrieval_left_an_empty_one():
    pipeline = _pipeline([
        '{"claims":[{"text":"Open Settings.","evidence_ids":[1]}],"refusal":null}',
        '{"supported_claim_indexes":[]}',
    ])
    state = _state(_doc())
    state["refusal_reason"] = ""

    pipeline._generate(state)

    assert state["refusal_reason"] == "claim_validation_failed"


def test_contextualization_failure_falls_back_to_the_sanitized_question(monkeypatch):
    pipeline = object.__new__(RagPipeline)
    monkeypatch.setattr(pipeline, "_contextualize", lambda _state: (_ for _ in ()).throw(RuntimeError()))

    result = pipeline._contextualize_with_retry({"question": "How do I set an alarm?", "history": []})

    assert result["standalone_question"] == "How do I set an alarm?"
    assert result["needs_clarification"] is False


def test_claim_retry_adds_schema_correction_instruction():
    state = _state(_doc())
    state["claim_generation_attempt"] = 2

    messages = _pipeline([])._claim_messages(state)

    assert len(messages) == 3
    assert "could not be safely rendered" in messages[-1].content


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


def test_clarification_is_retried_up_to_three_times_before_rendering():
    pipeline = _pipeline([])
    calls = 0

    def contextualize(_state):
        nonlocal calls
        calls += 1
        return {"standalone_question": "", "needs_clarification": True, "answer_language": "en"}

    pipeline._contextualize = contextualize

    result = pipeline._contextualize_with_retry({"question": "What about it?", "trace_id": "test"})

    assert calls == 4  # initial decision plus three retries
    assert result["needs_clarification"] is True


def test_no_context_is_retried_up_to_three_times_without_retrying_unsupported_models():
    pipeline = object.__new__(RagPipeline)
    pipeline.llm = _ScriptedModel([])
    attempts = 0

    def no_evidence(_state, _question):
        nonlocal attempts
        attempts += 1
        return {"grounded": False, "refusal_reason": "no_relevant_evidence", "question": "volume"}

    pipeline._subrequest_state = no_evidence
    pipeline._refuse = lambda state: {"answer": NO_CONTEXT_REPLY, "sources": [], "grounded": False}

    result, reason = pipeline._resolve_subrequest_with_retry({"trace_id": "test"}, "volume")

    assert attempts == 4  # initial retrieval plus three retries
    assert result["answer"] == NO_CONTEXT_REPLY
    assert reason == "no_relevant_evidence"


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


def test_whole_guide_summary_is_generated_from_extracted_pdf_segments(monkeypatch):
    pipeline = _pipeline([
        '{"sections":[{"heading":"Guide overview","claims":['
        '{"text":"iPhone overview","source_pages":[8]},'
        '{"text":"Phone features","source_pages":[12]}]}]}'
    ])
    segments = [
        Segment("iPhone overview", 8, 12, "Chapter 1: iPhone at a Glance", "Overview"),
        Segment("Phone features", 12, 16, "Chapter 1: iPhone at a Glance", "Features"),
    ]
    monkeypatch.setattr(rag_graph, "load_segments", lambda _path: segments)

    result = pipeline._guide_summary_result()

    assert result["answer"] == (
        "## Guide overview\n\n"
        "- iPhone overview (page 8)\n"
        "- Phone features (page 12)"
    )
    assert [source["page"] for source in result["sources"]] == [8, 12]
    assert result["context"] == []
    request = pipeline.llm.messages[0]
    assert "Requested detail: brief" in request[1].content
    assert "[Page 8 | Chapter 1: iPhone at a Glance | Overview]" in request[1].content
    assert "iPhone overview" in request[1].content


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
    events = list(pipeline.stream_answer("من فاز بكأس العالم؟"))

    assert events[-1]["final"] == {
        "answer": "لم أجد معلومات كافية في دليل مستخدم iPhone للإجابة عن هذا الطلب.",
        "sources": [],
        "grounded": False,
    }


def test_language_request_is_excluded_from_the_retrieval_splitter_input():
    pipeline = _pipeline([])
    state = {
        "question": "Explain how to take a panorama in Arabic, including the supporting page number.",
        "standalone_question": "How do I take a panorama photo?",
        "history": [],
    }

    result = pipeline._split_retrieval_queries(state)

    assert result == {"retrieval_queries": ["How do I take a panorama photo?"]}
    assert pipeline.llm.messages == []


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


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("how to increasese volume", "how to increase volume"),
        ("how to increse volume", "how to increase volume"),
        ("how to incraese volume", "how to increase volume"),
        ("how to decrase volume", "how to decrease volume"),
        ("how to tak a panaroma pikchure", "how to take a panorama picture"),
        ("how to use Airdorp", "how to use AirDrop"),
    ],
)
def test_typo_normalization_repairs_only_clear_safe_terms(question, expected):
    assert _normalise_retrieval_question(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        "How do I increase volume?", "How do I decrease volume?",
        "How do I disable AirDrop?", "How do I enable AirDrop?",
        "Does this work on iPhone 4?", "Does this work on iPhone 4s?",
        "What does page 11 say?", "What does page 111 say?",
        "How do I use iSight and Pano?", "How do I use XyloFlux?",
    ],
)
def test_typo_normalization_preserves_intent_identifiers_and_numbers(question):
    assert _normalise_retrieval_question(question) == question


def test_typo_and_correct_query_share_retrieval_validation_and_citation_flow():
    class _VolumeVectorStore:
        def __init__(self):
            self.queries = []

        def similarity_search_with_score(self, query, **_kwargs):
            self.queries.append(query)
            if query != "how to increase volume":
                return []
            return [
                (_doc(page=11, text="Use the volume buttons to adjust volume."), 0.91),
                (_doc(page=66, text="Sound settings also control volume."), 0.89),
            ]

    def run(question):
        pipeline = _pipeline([
            f'{{"standalone_question":"{question}","needs_clarification":false,"answer_language":"en"}}',
            '{"claims":[{"text":"Use the volume buttons.","evidence_ids":[1]},'
            '{"text":"You can also adjust sound settings.","evidence_ids":[2]}]}',
            '{"supported_claim_indexes":[0,1]}',
        ])
        store = _VolumeVectorStore()
        pipeline.vectorstore = store
        return list(pipeline.stream_answer(question))[-1]["final"], store, pipeline

    correct, correct_store, correct_pipeline = run("how to increase volume")
    typo, typo_store, typo_pipeline = run("how to increasese volume")

    assert correct_store.queries == typo_store.queries == ["how to increase volume"]
    assert typo == correct
    assert typo["grounded"] is True
    assert "(page 11)" in typo["answer"] and "(page 66)" in typo["answer"]
    assert len(correct_pipeline.llm.messages) == len(typo_pipeline.llm.messages) == 3
    assert "<question>how to increase volume</question>" in typo_pipeline.llm.messages[-1][1].content


def test_splitter_is_skipped_for_single_request_and_retained_for_multi_request():
    pipeline = _pipeline(['["How do I increase volume?", "How do I decrease volume?"]'])

    simple = pipeline._split_retrieval_queries(
        {"question": "How do I increase volume?", "standalone_question": "How do I increase volume?"}
    )
    multi = pipeline._split_retrieval_queries(
        {
            "question": "How do I increase volume and decrease volume?",
            "standalone_question": "How do I increase volume and decrease volume?",
        }
    )

    assert simple == {"retrieval_queries": ["How do I increase volume?"]}
    assert multi == {"retrieval_queries": ["How do I increase volume?", "How do I decrease volume?"]}
    assert len(pipeline.llm.messages) == 1
    assert not _requires_model_split("How do I increase volume?")
    assert _requires_model_split("How do I increase volume and decrease volume?")


def test_malformed_or_collapsed_splitter_output_preserves_explicit_multi_intent_boundaries():
    original = "Tell me about volume controls; also explain Pano."
    # This is a realistic contextualizer variation: it keeps both intents but
    # removes the semicolon that the deterministic fallback previously relied on.
    rewritten = "Tell me about volume controls and explain Pano."
    fallback = _split_fallback_queries(rewritten, original)

    assert fallback == ["Tell me about volume controls", "also explain Pano"]
    assert _parse_retrieval_queries("not JSON", fallback) == fallback
    assert _parse_retrieval_queries('["Tell me about volume controls and explain Pano."]', fallback) == fallback


def test_volume_and_pano_survive_contextualizer_and_splitter_variability_end_to_end():
    """Reproduce the live failure path without a provider or Pinecone call.

    The contextualizer's valid rewrite removes the user's semicolon and the
    splitter returns malformed JSON. Before the fallback-boundary fix, retrieval
    received one combined query, no feature-specific evidence, and refused.
    """
    class _TopicStore:
        def __init__(self):
            self.queries = []

        def similarity_search_with_score(self, query, **_kwargs):
            self.queries.append(query)
            if "volume" in query and "Pano" not in query:
                return [(_doc(11, text="Use the volume buttons to adjust volume."), 0.91)]
            if "Pano" in query and "volume" not in query:
                return [(_doc(79, text="Choose Pano, then pan slowly."), 0.92)]
            return []

    pipeline = _pipeline([
        '{"standalone_question":"Tell me about volume controls and explain Pano.",'
        '"needs_clarification":false,"answer_language":"en"}',
        "not JSON",
        '{"claims":[{"text":"Use the volume buttons.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
        '{"claims":[{"text":"Choose Pano and pan slowly.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
    ])
    store = _TopicStore()
    pipeline.vectorstore = store

    result = list(pipeline.stream_answer(
        "Tell me about volume controls; also explain Pano."
    ))[-1]["final"]

    assert store.queries == ["Tell me about volume controls", "also explain Pano"]
    assert result["grounded"] is True
    assert "(page 11)" in result["answer"] and "(page 79)" in result["answer"]


def test_incomplete_contextualizer_information_needs_do_not_bypass_explicit_multi_intent_split():
    result = _parse_contextualization(
        '{"standalone_question":"What are the volume controls, and what is Pano?",'
        '"needs_clarification":false,"answer_language":"en",'
        '"information_needs":["What are the volume controls, and what is Pano?"],'
        '"relationship":"single_question"}',
        "Tell me about volume controls; also explain Pano.",
        [],
    )

    assert result["standalone_question"] == "What are the volume controls, and what is Pano?"
    assert "information_needs" not in result


def test_incomplete_contextualizer_plan_follows_splitter_and_recovers_both_supported_topics():
    class _TopicStore:
        def similarity_search_with_score(self, query, **_kwargs):
            if "volume" in query:
                return [(_doc(11, text="Use the volume buttons to adjust volume."), 0.91)]
            if "Pano" in query:
                return [(_doc(79, text="Choose Pano, then pan slowly."), 0.92)]
            return []

    pipeline = _pipeline([
        '{"standalone_question":"What are the volume controls, and what is Pano?",'
        '"needs_clarification":false,"answer_language":"en",'
        '"information_needs":["What are the volume controls, and what is Pano?"],'
        '"relationship":"single_question"}',
        '["What volume controls are available on iPhone?", "What is Pano on iPhone?"]',
        '{"claims":[{"text":"Use the volume buttons.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
        '{"claims":[{"text":"Choose Pano and pan slowly.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
    ])
    pipeline.vectorstore = _TopicStore()

    result = list(pipeline.stream_answer(
        "Tell me about volume controls; also explain Pano.", trace_id="trace"
    ))[-1]["final"]

    assert result["grounded"] is True
    assert "(page 11)" in result["answer"] and "(page 79)" in result["answer"]


def test_contextualized_information_needs_retrieve_two_supported_topics_without_splitter():
    class _TopicStore:
        def __init__(self):
            self.queries = []

        def similarity_search_with_score(self, query, **_kwargs):
            self.queries.append(query)
            if "increase" in query:
                return [(_doc(11, text="Use the volume buttons to adjust volume."), 0.92)]
            if "panorama" in query:
                return [(_doc(79, text="Choose Pano, then pan slowly."), 0.93)]
            return []

    pipeline = _pipeline([
        '{"standalone_question":"Explain volume and panorama.","needs_clarification":false,'
        '"answer_language":"en","information_needs":["How do I increase volume?",'
        '"How do I take a panorama photo?"],"relationship":"independent_procedures"}',
        '{"claims":[{"text":"Use the volume buttons.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
        '{"claims":[{"text":"Choose Pano and pan slowly.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
    ])
    store = _TopicStore()
    pipeline.vectorstore = store

    result = list(pipeline.stream_answer(
        "Explain how to increase the volume and how to take a panorama photo."
    ))[-1]["final"]

    assert store.queries == ["How do I increase volume?", "How do I take a panorama photo?"]
    assert "(page 11)" in result["answer"] and "(page 79)" in result["answer"]
    assert NO_CONTEXT_REPLY not in result["answer"]
    # Contextualization plus generation/validation for each need: no splitter call.
    assert len(pipeline.llm.messages) == 5


def test_combined_behavior_keeps_supported_facts_separate_from_undocumented_interaction():
    class _TopicStore:
        def similarity_search_with_score(self, query, **_kwargs):
            if "interaction" in query:
                return []
            if "volume" in query:
                return [(_doc(11, text="Use the volume buttons to adjust volume."), 0.92)]
            return [(_doc(79, text="Choose Pano, then pan slowly."), 0.93)]

    pipeline = _pipeline([
        '{"standalone_question":"Can I make iPhone louder while taking panorama?",'
        '"needs_clarification":false,"answer_language":"en","information_needs":['
        '"How do I increase volume?","How do I take a panorama photo?",'
        '"Does the guide document the interaction between volume controls and panorama photography?"],'
        '"relationship":"combined_behavior"}',
        '{"claims":[{"text":"Use the volume buttons.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
        '{"claims":[{"text":"Choose Pano and pan slowly.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
    ])
    pipeline.vectorstore = _TopicStore()

    result = list(pipeline.stream_answer(
        "How can I make my iPhone louder while taking a panorama photo?"
    ))[-1]["final"]

    assert "(page 11)" in result["answer"] and "(page 79)" in result["answer"]
    assert "does not provide enough evidence to establish the requested interaction" in result["answer"]
    assert NO_CONTEXT_REPLY not in result["answer"]


def test_egyptian_arabic_multi_intent_keeps_all_visible_outcomes_in_arabic():
    class _TopicStore:
        def similarity_search_with_score(self, query, **_kwargs):
            if "volume" in query:
                return [(_doc(11, text="Use the volume buttons to adjust volume."), 0.92)]
            return [(_doc(79, text="Choose Pano, then pan slowly."), 0.93)]

    pipeline = _pipeline([
        '{"standalone_question":"Increase volume and take panorama.","needs_clarification":false,'
        '"answer_language":"en","information_needs":["How do I increase volume?",'
        '"How do I take a panorama photo?"],"relationship":"independent_procedures"}',
        '{"claims":[{"text":"Use the volume buttons.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
        '{"translations":["استخدم أزرار الصوت."]}',
        '{"claims":[{"text":"Choose Pano and pan slowly.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
        '{"translations":["اختر Pano وحرّك iPhone ببطء."]}',
    ])
    pipeline.vectorstore = _TopicStore()

    result = list(pipeline.stream_answer("ازاي اعلي صوت الايفون واصور بانوراما؟"))[-1]["final"]

    assert result["grounded"] is True
    assert "استخدم أزرار الصوت." in result["answer"]
    assert "اختر Pano" in result["answer"]
    assert "(page 11)" in result["answer"] and "(page 79)" in result["answer"]
    assert NO_CONTEXT_REPLY not in result["answer"]


@pytest.mark.parametrize(
    "question",
    [
        "how to increase volume",
        "How do you increase the volume on iPhone?",
        "How do I turn up the volume on my iPhone?",
        "How do the volume buttons work?",
        "How do I adjust the ringer volume?",
        "how to increasese volume",
    ],
)
def test_equivalent_volume_queries_reach_the_real_relevance_gate_with_page_11_evidence(question):
    class _VolumeStore:
        def __init__(self):
            self.queries = []

        def similarity_search_with_score(self, query, **_kwargs):
            self.queries.append(query)
            return [(_doc(page=11, text="The side buttons adjust audio and ringer volume."), 0.82)]

    pipeline = object.__new__(RagPipeline)
    pipeline.vectorstore = _VolumeStore()
    normalized = _normalise_retrieval_question(question)
    result = pipeline._retrieve({
        "question": normalized,
        "standalone_question": normalized,
        "retrieval_queries": [normalized],
        "trace_id": "volume-regression",
    })

    assert result["grounded"] is True
    assert result["top_score"] == 0.82
    assert result["documents"][0].metadata["page"] == 11
    assert pipeline.vectorstore.queries == [normalized]


def test_weak_retrieval_is_retried_once_before_a_ui_refusal():
    class _TransientVolumeStore:
        def __init__(self):
            self.calls = 0

        def similarity_search_with_score(self, _query, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return []
            return [(_doc(page=11, text="Use the side buttons to adjust volume."), 0.82)]

    pipeline = _pipeline([
        '{"standalone_question":"How do I increase volume?","needs_clarification":false,"answer_language":"en"}',
        '{"claims":[{"text":"Use the side buttons.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
    ])
    store = _TransientVolumeStore()
    pipeline.vectorstore = store

    result = list(pipeline.stream_answer("How do I increase volume?"))[-1]["final"]

    assert store.calls == 2
    assert result["grounded"] is True
    assert result["answer"] == "Use the side buttons. (page 11)"


def test_claim_validation_is_retried_once_with_the_same_retrieved_evidence():
    class _VolumeStore:
        def __init__(self):
            self.calls = 0

        def similarity_search_with_score(self, _query, **_kwargs):
            self.calls += 1
            return [(_doc(page=11, text="Use the side buttons to adjust volume."), 0.82)]

    pipeline = _pipeline([
        '{"standalone_question":"How do I increase volume?","needs_clarification":false,"answer_language":"en"}',
        '{"claims":[{"text":"Unsupported wording.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[]}',
        '{"claims":[{"text":"Use the side buttons.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
    ])
    store = _VolumeStore()
    pipeline.vectorstore = store

    result = list(pipeline.stream_answer("How do I increase volume?"))[-1]["final"]

    assert store.calls == 1
    assert result["grounded"] is True
    assert result["answer"] == "Use the side buttons. (page 11)"
    # Contextualization plus two generation/validation attempts; no simple-query splitter call.
    assert len(pipeline.llm.messages) == 5


_MULTIPART_TOPICS = {
    "alarm": (85, "Create an alarm in Clock."),
    "reminder": (94, "Create a reminder in Reminders."),
    "calendar": (70, "Add an event in Calendar."),
    "volume": (11, "Use the volume buttons."),
    "pano": (79, "Choose Pano and pan slowly."),
    "screenshot": (17, "Take a screenshot with the Sleep/Wake and Home buttons."),
}


def _multipart_pipeline(question, needs, language="en"):
    """Exercise stream_answer with independent mocked retrieval and LLM stages."""
    responses = [json.dumps({
        "standalone_question": "; ".join(needs),
        "needs_clarification": False,
        "answer_language": language,
        "information_needs": needs,
        "relationship": "independent_procedures" if len(needs) > 1 else "single_question",
    })]
    for need in needs:
        key = next(key for key in _MULTIPART_TOPICS if key in need.lower())
        _page, claim = _MULTIPART_TOPICS[key]
        responses.extend([
            json.dumps({"claims": [{"text": claim, "evidence_ids": [1]}]}),
            json.dumps({"supported_claim_indexes": [0]}),
        ])
        if language == "ar":
            responses.append(json.dumps({"translations": [f"خطوات {key}."]}))

    class _TopicStore:
        def __init__(self):
            self.queries = []

        def similarity_search_with_score(self, query, **_kwargs):
            self.queries.append(query)
            for key, (page, text) in _MULTIPART_TOPICS.items():
                if key in query.lower():
                    return [(_doc(page, text=text), 0.88)]
            return []

    pipeline = _pipeline(responses)
    store = _TopicStore()
    pipeline.vectorstore = store
    return pipeline, store


@pytest.mark.parametrize(
    ("question", "need", "page"),
    [
        ("How do I set an alarm?", "How do I set an alarm?", 85),
        ("How do I create a reminder?", "How do I create a reminder?", 94),
        ("How do I add a calendar event?", "How do I add a calendar event?", 70),
    ],
)
def test_distant_chapter_procedures_retrieve_and_validate_independently(question, need, page):
    pipeline, store = _multipart_pipeline(question, [need])

    result = list(pipeline.stream_answer(question))[-1]["final"]

    assert result["grounded"] is True
    assert f"(page {page})" in result["answer"]
    assert store.queries == [need]


def test_alarm_reminder_and_calendar_are_isolated_in_one_request():
    needs = [
        "How do I set an alarm?",
        "How do I create a reminder?",
        "How do I add a calendar event?",
    ]
    pipeline, store = _multipart_pipeline(
        "How do I set an alarm, create a reminder, and add a calendar event?", needs,
    )

    result = list(pipeline.stream_answer(
        "How do I set an alarm, create a reminder, and add a calendar event?"
    ))[-1]["final"]

    assert result["grounded"] is True
    assert store.queries == needs
    assert all(f"(page {page})" in result["answer"] for page in (85, 94, 70))


def test_three_independent_topics_do_not_share_a_retrieval_result_limit():
    needs = [
        "How do I adjust volume?",
        "How do I use Pano?",
        "How do I take a screenshot?",
    ]
    pipeline, store = _multipart_pipeline("Volume, Pano, and screenshot instructions.", needs)

    result = list(pipeline.stream_answer("Volume, Pano, and screenshot instructions."))[-1]["final"]

    assert result["grounded"] is True
    assert store.queries == needs
    assert all(f"(page {page})" in result["answer"] for page in (11, 79, 17))


@pytest.mark.parametrize(
    "question",
    [
        "كيف أضبط منبهًا، وأنشئ تذكيرًا، وأضيف حدثًا إلى التقويم؟",
        "ازاي أظبط منبه وأعمل تذكير وأضيف حدث في التقويم؟",
    ],
)
def test_arabic_and_egyptian_arabic_multi_part_requests_keep_independent_citations(question):
    needs = [
        "How do I set an alarm?",
        "How do I create a reminder?",
        "How do I add a calendar event?",
    ]
    pipeline, _store = _multipart_pipeline(question, needs, language="ar")

    result = list(pipeline.stream_answer(question))[-1]["final"]

    assert result["grounded"] is True
    assert "خطوات alarm." in result["answer"]
    assert all(f"(page {page})" in result["answer"] for page in (85, 94, 70))


def test_supported_and_unsupported_parts_keep_the_supported_result_visible():
    class _Store:
        def similarity_search_with_score(self, query, **_kwargs):
            if "alarm" in query.lower():
                return [(_doc(85, text="Create an alarm in Clock."), 0.88)]
            return []

    pipeline = _pipeline([
        '{"standalone_question":"Set an alarm and predict the weather.",'
        '"needs_clarification":false,"answer_language":"en",'
        '"information_needs":["How do I set an alarm?", "How do I predict the weather?"],'
        '"relationship":"independent_procedures"}',
        '{"claims":[{"text":"Create an alarm in Clock.","evidence_ids":[1]}]}',
        '{"supported_claim_indexes":[0]}',
    ])
    pipeline.vectorstore = _Store()

    result = list(pipeline.stream_answer("Set an alarm and predict the weather."))[-1]["final"]

    assert result["grounded"] is True
    assert "(page 85)" in result["answer"]
    assert "requested part 2" in result["answer"]


def test_repeated_identical_multi_part_requests_are_deterministic_with_mocked_dependencies():
    question = "How do I set an alarm, create a reminder, and add a calendar event?"
    needs = ["How do I set an alarm?", "How do I create a reminder?", "How do I add a calendar event?"]

    first_pipeline, _ = _multipart_pipeline(question, needs)
    second_pipeline, _ = _multipart_pipeline(question, needs)
    first = list(first_pipeline.stream_answer(question))[-1]["final"]
    second = list(second_pipeline.stream_answer(question))[-1]["final"]

    assert first == second
