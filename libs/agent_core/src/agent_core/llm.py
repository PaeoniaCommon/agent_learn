"""LLM access. All agent and learner LLM calls go through `StructuredLLM.structured`."""

from __future__ import annotations

from typing import Protocol, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from .config import LLMSettings

T = TypeVar("T", bound=BaseModel)

INTERNAL_TAG = "rpa_internal"


class StructuredLLM(Protocol):
    def structured(self, schema: type[T], system: str, user: str, *, step: str) -> T: ...

    def chat_model(self): ...


class OpenAIStructuredLLM:
    """ChatOpenAI configured from config.yaml. The model is created lazily."""

    def __init__(self, cfg: LLMSettings):
        self.cfg = cfg
        self._model = None

    def chat_model(self):
        if self._model is None:
            from langchain_openai import ChatOpenAI

            kwargs = dict(
                model=self.cfg.model,
                temperature=self.cfg.temperature,
                timeout=self.cfg.timeout_s,
                max_retries=self.cfg.max_retries,
            )
            if self.cfg.api_key:
                kwargs["api_key"] = self.cfg.api_key
            if self.cfg.base_url:
                kwargs["base_url"] = self.cfg.base_url
            self._model = ChatOpenAI(**kwargs)
        return self._model

    def structured(self, schema: type[T], system: str, user: str, *, step: str) -> T:
        runnable = self.chat_model().with_structured_output(schema, method=self.cfg.structured_output_method)
        return runnable.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)],
            config={"tags": [INTERNAL_TAG], "metadata": {"rpa_step": step}},
        )
