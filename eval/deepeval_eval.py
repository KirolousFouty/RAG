"""LLM-level evaluation of the RAG answers with DeepEval.

Rana suggested in the interview that I use DeepEval to evaluate the LLM side of the
system, so I added this alongside the retrieval benchmark in run_eval.py. Where
run_eval.py measures *retrieval* (did the right page come back?), this measures the
*generated answer* against the retrieved context with Claude as the judge:

  * Faithfulness        — is every claim in the answer supported by the context?
                          (this is the groundedness / anti-hallucination check)
  * Answer Relevancy    — does the answer actually address the question?
  * Contextual Relevancy — was the retrieved context relevant to the question?

It runs against the live Pinecone index and calls the chat model, so it needs the
same .env the app uses. Kept to a few questions to stay cheap.

Run:  python -m eval.deepeval_eval
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# DeepEval phones home and can prompt for a login otherwise; keep it fully local.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_DISABLE_PROGRESS_BAR", "YES")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepeval.metrics import (  # noqa: E402
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase  # noqa: E402

from eval.claude_judge import ClaudeJudge  # noqa: E402
from src.rag_graph import RagPipeline  # noqa: E402

# A few in-guide questions; the grounding/refusal behaviour itself is covered by the
# offline cases in run_eval and the manual checks in the README.
QUESTIONS = [
    "How do I turn on Do Not Disturb?",
    "How do I take a panorama photo?",
    "How do I unlock my iPhone with my fingerprint?",
    "How do I make the screen brighter?",
]


def main():
    judge = ClaudeJudge()
    pipeline = RagPipeline()

    metrics = [
        FaithfulnessMetric(threshold=0.7, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=0.7, model=judge, include_reason=True),
        ContextualRelevancyMetric(threshold=0.7, model=judge, include_reason=True),
    ]

    totals = {m.__class__.__name__: [] for m in metrics}
    for q in QUESTIONS:
        result = pipeline.answer(q)
        case = LLMTestCase(
            input=q,
            actual_output=result["answer"],
            retrieval_context=result["context"],
        )
        print(f"\nQ: {q}")
        for m in metrics:
            m.measure(case)
            totals[m.__class__.__name__].append(m.score)
            print(f"  {m.__class__.__name__:24} {m.score:.2f}  ({m.reason[:90]})")

    print("\n== Averages ==")
    for name, scores in totals.items():
        print(f"  {name:24} {sum(scores) / len(scores):.2f}")


if __name__ == "__main__":
    main()
