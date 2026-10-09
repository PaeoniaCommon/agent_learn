"""StateGraph wiring and build_agent()."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Callable

from langgraph.graph import END, START, StateGraph

from agent_core.graph import make_input  # noqa: F401  (re-exported)
from agent_core.llm import OpenAIStructuredLLM
from agent_core.streaming import TextStreamModel

from . import nodes
from .backend import Backend
from .config import Settings, load_config
from .data_store import DataStore, get_data_store
from .deps import Deps
from .knowledge import KnowledgeBase
from .learning import LearningWorker
from .state import RPAState

USER_FACING_NODES = frozenset({"ask_view", "ask_param", "respond"})


def _bind(fn, deps: Deps):
    def node(state):
        return fn(state, deps)
    node.__name__ = fn.__name__
    return node


def build_graph(deps: Deps) -> StateGraph:
    g = StateGraph(RPAState)
    for name in ("ingest", "resolve_view", "ask_view", "confirm_view", "resolve_params", "ask_param",
                 "confirm_param", "finalize_params", "fetch", "respond"):
        g.add_node(name, _bind(getattr(nodes, name), deps))
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "resolve_view")
    g.add_conditional_edges("resolve_view", nodes.route_after_resolve_view,
                            ["resolve_params", "ask_view", "respond"])
    g.add_edge("ask_view", "confirm_view")
    g.add_conditional_edges("confirm_view", nodes.route_after_confirm_view,
                            ["resolve_params", "ask_view", "respond"])
    g.add_conditional_edges("resolve_params", nodes.route_after_resolve_params, ["ask_param", "finalize_params"])
    g.add_edge("ask_param", "confirm_param")
    g.add_conditional_edges("confirm_param", nodes.route_after_confirm_param, ["ask_param", "finalize_params"])
    g.add_conditional_edges("finalize_params", nodes.route_after_finalize, ["fetch", "respond"])
    g.add_edge("fetch", "respond")
    g.add_edge("respond", END)
    return g


def build_agent(
    config: str | Path | dict | Settings | None = None,
    *,
    checkpointer=None,
    backend: Backend | None = None,
    llm=None,
    learning_llm=None,
    data_store: DataStore | None = None,
    clock: Callable[[], datetime | date] | None = None,
):
    """Build the agent. Returns a compiled LangGraph graph (same type as create_agent returns).

    Extras attached: `.data_store`, `.learning` (worker), `.knowledge`, `.deps`.
    """
    settings = load_config(config)
    if backend is None:
        if not settings.backend.module:
            raise ValueError("No backend: pass backend=Backend(...) or set backend.module in the config")
        backend = Backend.from_module(settings.backend.module)
    llm = llm or OpenAIStructuredLLM(settings.llm)
    learning_llm = learning_llm or (llm if settings.learning_llm is None
                                    else OpenAIStructuredLLM(settings.effective_learning_llm))
    kb = KnowledgeBase(settings.paths.knowledge_dir)
    kb.sync_views(backend.views())
    store = data_store or get_data_store(settings.data_store.max_entries)
    learning = LearningWorker(kb, learning_llm, backend, settings)
    deps = Deps(settings=settings, backend=backend, llm=llm, kb=kb, store=store, learning=learning,
                streamer=TextStreamModel(chunk_chars=settings.respond.stream_chunk_chars), clock=clock)

    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver

        checkpointer = InMemorySaver()
    agent = build_graph(deps).compile(checkpointer=checkpointer, name="rpa_agent")
    agent.data_store = store
    agent.learning = learning
    agent.knowledge = kb
    agent.deps = deps
    return agent

