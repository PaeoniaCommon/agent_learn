"""TextStreamModel: streams text that was built in code through LangGraph's `messages` mode.

Invoked inside a node like any chat model, so `stream_mode="messages"` yields its chunks
tagged with that node — exactly like model tokens from a create_agent object. It makes no
network call; the "response" is the content of the last input message.
"""

from __future__ import annotations

from typing import Any, Iterator

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult


class TextStreamModel(BaseChatModel):
    chunk_chars: int = 16

    @property
    def _llm_type(self) -> str:
        return "rpa-text-stream"

    def _text(self, messages: list[BaseMessage]) -> str:
        return str(messages[-1].content) if messages else ""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self._text(messages)))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        text = self._text(messages)
        n = max(1, self.chunk_chars)
        for i in range(0, len(text), n):
            piece = text[i:i + n]
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=piece))
            if run_manager:
                run_manager.on_llm_new_token(piece, chunk=chunk)
            yield chunk


def stream_text(model: TextStreamModel, text: str) -> AIMessage:
    """Emit `text` as streamed chunks (when a stream is listening) and return the full message."""
    msg = model.invoke([HumanMessage(content=text)])
    return AIMessage(content=text, id=msg.id)
