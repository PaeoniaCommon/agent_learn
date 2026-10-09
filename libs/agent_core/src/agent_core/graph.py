"""LangGraph helpers shared by agents: progress events and chat-loop input."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.types import Command


def progress(stage: str, **data) -> None:
    """Emit a `custom` stream event; no-op outside a streaming run."""
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()({"stage": stage, **data})
    except Exception:
        pass


def is_waiting(agent, config: dict) -> bool:
    """True if the thread is paused on an interrupt (a question to the user)."""
    try:
        snapshot = agent.get_state(config)
        return bool(snapshot.interrupts) or any(t.interrupts for t in snapshot.tasks)
    except Exception:
        return False


def make_input(agent, config: dict, user_text: str):
    """Command(resume=...) if the thread is waiting on a question, else a new request."""
    if is_waiting(agent, config):
        return Command(resume=user_text)
    return {"messages": [HumanMessage(content=user_text)]}
