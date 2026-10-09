"""Dependencies shared by the graph nodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Callable

from agent_core.dates import today_in
from agent_core.streaming import TextStreamModel

from .backend import Backend
from .config import Settings
from .data_store import DataStore
from .knowledge import KnowledgeBase
from .learning import LearningWorker


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

