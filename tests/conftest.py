from __future__ import annotations

import re
import time
from datetime import date

import pandas as pd
import pytest
from langgraph.types import Command

from rpa_agent import USER_FACING_NODES, Backend, DataStore, build_agent

TODAY = date(2026, 10, 9)

VIEWS = ["sales_daily", "sales_weekly", "stock_level", "sales_forecast"]
VIEW_PARAMS = {
    "sales_daily": {"mandatory": ["region", "date_from", "date_to"], "optional": ["currency", "store_id"],
                    "default": {"currency": "GBP"}},
    "sales_weekly": {"mandatory": ["region", "date_from", "date_to"], "optional": ["currency"],
                     "default": {"currency": "GBP"}},
    "stock_level": {"mandatory": ["region", "as_of_date"], "optional": [], "default": {}},
    "sales_forecast": {"mandatory": ["region", "horizon"], "optional": [], "default": {"horizon": "12w"}},
}
VALID = {"region": ["UK", "IE", "FR"], "currency": ["GBP", "EUR"]}


class FakeBackend:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def validate_view(self, v):
        return v in VIEWS

    def get_view_params(self, v):
        return VIEW_PARAMS[v]

    def valid_params(self, p):
        return VALID.get(p, [])

    def get_data(self, view, params):
        self.calls.append((view, dict(params)))
        for p in ("date_from", "date_to", "as_of_date"):
            if p in params and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(params[p])):
                raise ValueError(f"{p}: unparseable date '{params[p]}', expected YYYY-MM-DD")
        if "store_id" in params and not str(params["store_id"]).isdigit():
            raise ValueError("store_id must be numeric")
        if params.get("region") == "FR":
            raise LookupError("No data for the given filters")
        return pd.DataFrame({"store_id": [101, 102, 103], "sku": ["A-22", "B-1", "C-9"],
                             "date": ["2024-01-01"] * 3, "units": [4, 2, 7], "revenue": [19.96, 5.0, 33.1]})

    def backend(self) -> Backend:
        return Backend(VIEWS, self.validate_view, self.get_view_params, self.valid_params, self.get_data)


class ScriptedLLM:
    """Returns scripted outputs per schema name; fails loudly on unexpected calls."""

    def __init__(self, **handlers):
        self.handlers = handlers
        self.calls: list[tuple[str, str]] = []
        self.delay = 0.0

    def structured(self, schema, system, user, *, step):
        self.calls.append((schema.__name__, user))
        if self.delay:
            time.sleep(self.delay)
        h = self.handlers.get(schema.__name__)
        if h is None:
            raise AssertionError(f"unexpected LLM call: {schema.__name__} ({step})")
        out = h.pop(0) if isinstance(h, list) else h(user)
        return out if isinstance(out, schema) else schema(**out)

    def chat_model(self):
        raise AssertionError("chat_model not expected")


@pytest.fixture
def fake_backend():
    return FakeBackend()


@pytest.fixture
def make_agent(tmp_path, fake_backend):
    agents = []

    def _make(llm=None, learning_llm=None, **cfg):
        config = {"paths": {"knowledge_dir": str(tmp_path / "knowledge")},
                  "learning": {"debounce_s": 0, **cfg.pop("learning", {})},
                  "respond": {"stream_chunk_chars": 8}, **cfg}
        llm = llm or ScriptedLLM()
        learning_llm = learning_llm or ScriptedLLM()
        agent = build_agent(config, backend=fake_backend.backend(), llm=llm, learning_llm=learning_llm,
                            data_store=DataStore(), clock=lambda: TODAY)
        agent.test_llm, agent.test_learning_llm = llm, learning_llm
        agents.append(agent)
        return agent

    yield _make
    for a in agents:
        a.learning.shutdown(wait=False)


class Turn:
    def __init__(self):
        self.chunks: dict[str, list[str]] = {}
        self.interrupts: list = []
        self.state: dict = {}

    def text(self, node: str = "respond") -> str:
        return "".join(self.chunks.get(node, []))

    @property
    def user_text(self) -> str:
        return "".join("".join(v) for k, v in self.chunks.items() if k in USER_FACING_NODES)


def run(agent, thread: str, user_input, **extra) -> Turn:
    config = {"configurable": {"thread_id": thread}}
    if isinstance(user_input, str):
        inputs = {"messages": [{"role": "user", "content": user_input}], **extra}
    else:
        inputs = user_input
    t = Turn()
    for mode, chunk in agent.stream(inputs, config, stream_mode=["messages", "updates"]):
        if mode == "messages":
            msg, meta = chunk
            node = meta.get("langgraph_node")
            if node in USER_FACING_NODES and msg.content:
                t.chunks.setdefault(node, []).append(msg.content)
        elif mode == "updates" and "__interrupt__" in chunk:
            t.interrupts.extend(chunk["__interrupt__"])
    t.state = agent.get_state(config).values
    return t


def resume(agent, thread: str, answer) -> Turn:
    return run(agent, thread, Command(resume=answer))
