"""Dependencies shared by the graph nodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable

from .backend import Backend
from .config import Settings
from .data_store import DataStore
from .dates import today_in
from .knowledge import KnowledgeBase
from .learning import LearningWorker
from .streaming import TextStreamModel


@dataclass
class Deps:
    settings: Settings
    backend: Backend
    llm: object  # StructuredLLM
    kb: KnowledgeBase
    store: DataStore
    learning: LearningWorker
    streamer: TextStreamModel
    clock: Callable[[], datetime | date] | None = None

    def today(self) -> date:
        return today_in(self.settings.dates, self.clock)


def progress(stage: str, **data) -> None:
    """Emit a `custom` stream event; no-op outside a streaming run."""
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"stage": stage, **data})
    except Exception:
        pass
