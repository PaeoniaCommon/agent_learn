"""Dependencies shared by the graph nodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable

from ..config import Settings
from ..data.backend import Backend
from ..data.store import DataStore
from ..knowledge.learning import LearningWorker
from ..knowledge.store import KnowledgeBase
from ..llm.streaming import TextStreamModel
from ..validation.dates import today_in


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
