"""A DeepEval judge model backed by the configured OpenAI chat model."""

from __future__ import annotations

from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI

from src.config import settings


class OpenAIJudge(DeepEvalBaseLLM):
    def __init__(self, model: str | None = None):
        self.model = model or settings.openai_chat_model
        self._chat = ChatOpenAI(
            model=self.model,
            max_completion_tokens=2048,
            timeout=120,
            api_key=settings.openai_api_key,
        )

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
