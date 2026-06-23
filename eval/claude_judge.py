"""A DeepEval judge model backed by Claude.

DeepEval defaults to OpenAI as the judge LLM. I don't use OpenAI anywhere else in
this project, so I wrap the same Anthropic chat model the app uses as the evaluator
instead — one provider, one key. DeepEval's metrics ask the judge for structured
(schema-typed) output, which I get from langchain's `with_structured_output`.
"""

from __future__ import annotations

from deepeval.models import DeepEvalBaseLLM
from langchain_anthropic import ChatAnthropic

from src.config import settings


class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(self, model: str | None = None):
        self.model = model or settings.chat_model
        self._chat = ChatAnthropic(model=self.model, max_tokens=2048, timeout=120)

    def load_model(self):
        return self._chat

    def generate(self, prompt: str, schema=None):
        if schema is not None:
            return self._chat.with_structured_output(schema).invoke(prompt)
        return self._chat.invoke(prompt).content

    async def a_generate(self, prompt: str, schema=None):
        if schema is not None:
            return await self._chat.with_structured_output(schema).ainvoke(prompt)
        return (await self._chat.ainvoke(prompt)).content

    def get_model_name(self) -> str:
        return self.model
