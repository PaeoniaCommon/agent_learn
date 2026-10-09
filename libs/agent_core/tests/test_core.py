from langchain_core.messages import HumanMessage

from agent_core.config import LLMSettings, expand_env, merge_llm_settings
from agent_core.streaming import TextStreamModel
from agent_core.text import norm, tokens


def test_expand_env(monkeypatch):
    monkeypatch.setenv("X_KEY", "abc")
    monkeypatch.delenv("X_MISSING", raising=False)
    assert expand_env({"a": "${X_KEY}", "b": ["${X_MISSING:-dflt}"], "c": "${X_MISSING}"}) == \
        {"a": "abc", "b": ["dflt"], "c": None}


def test_merge_llm_settings():
    base = LLMSettings(model="m1", api_key="k")
    merged = merge_llm_settings(base, LLMSettings(model="m2"))
    assert merged.model == "m2" and merged.api_key == "k"


def test_text_stream_model_chunks():
    model = TextStreamModel(chunk_chars=4)
    chunks = [c.content for c in model.stream([HumanMessage(content="hello world")]) if c.content]
    assert chunks == ["hell", "o wo", "rld"]
    assert model.invoke([HumanMessage(content="hi")]).content == "hi"


def test_text_helpers():
    assert norm("Sales Daily") == norm("sales_daily")
    assert tokens("dateFrom") == ["date", "from"]
