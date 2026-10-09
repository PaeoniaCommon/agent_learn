"""Unit tests: dates, validation, parsing, knowledge, learning gate details."""

from __future__ import annotations

import pytest
from conftest import TODAY, FakeBackend, ScriptedLLM

from rpa_agent.agent.nodes.ingest import parse_explicit
from rpa_agent.config import DateSettings, Settings
from rpa_agent.knowledge.learning import LearningEvent, LearningWorker
from rpa_agent.knowledge.store import KnowledgeBase
from rpa_agent.validation.dates import infer_formats, resolve, resolve_value
from rpa_agent.validation.values import check_value, match_name

CFG = DateSettings(fiscal_year_start_month=4, day_first=True)


@pytest.mark.parametrize("text,start,end", [
    ("last month", "2026-09-01", "2026-09-30"),
    ("Q1 2024", "2024-01-01", "2024-03-31"),
    ("YTD", "2026-01-01", "2026-10-09"),
    ("since 3 March", "2026-03-03", "2026-10-09"),
    ("03/04/2024", "2024-04-03", "2024-04-03"),
    ("FY24", "2023-04-01", "2024-03-31"),
    ("last 7 days", "2026-10-03", "2026-10-09"),
    ("between Jan 2024 and Mar 2024", "2024-01-01", "2024-03-31"),
    ("end of last month", "2026-09-30", "2026-09-30"),
    ("this week", "2026-10-05", "2026-10-11"),
    ("in 2023", "2023-01-01", "2023-12-31"),
])
def test_dates_resolve(text, start, end):
    spans, unresolved = resolve(text, TODAY, CFG)
    assert len(spans) == 1 and (spans[0].start, spans[0].end) == (start, end) and not unresolved


def test_dates_unresolved_and_may():
    spans, unresolved = resolve("sales around Easter", TODAY, CFG)
    assert spans == [] and unresolved == ["around Easter"]
    spans, _ = resolve("I may need it", TODAY, CFG)
    assert spans == []
    spans, _ = resolve("store 12 sold 2024 units", TODAY, CFG)
    assert spans == []


def test_dates_month_first_config():
    span = resolve_value("03/04/2024", TODAY, DateSettings(day_first=False))
    assert span.start == "2024-03-04"


def test_infer_formats():
    assert infer_formats("2024-01-31") == ["%Y-%m-%d"]
    assert "%d/%m/%Y" in infer_formats("31/01/2024")
    assert infer_formats("GBP") == []


def test_match_name():
    assert match_name("Sales Daily", ["sales_daily", "sales_weekly"], 0.92) == ("sales_daily", True)
    assert match_name("sales_daily", ["sales_daily"], 0.92) == ("sales_daily", False)
    assert match_name("sales", ["sales_daily", "sales_weekly"], 0.92) == (None, False)


def test_check_value():
    assert check_value("uk", ["UK", "IE"]).value == "UK"
    assert not check_value("DE", ["UK", "IE"]).ok
    assert check_value("5", [5, 6]).value == 5
    assert not check_value("A1", [], r"\d+").ok
    assert check_value("anything", []).ok


def test_parse_explicit():
    view, params, free = parse_explicit('view: sales_daily params: {"region": "UK"} @date_from="1 Jan 2024" for me')
    assert view == "sales_daily" and params == {"region": "UK", "date_from": "1 Jan 2024"} and free == "for me"
    view, params, free = parse_explicit("@view=stock_level @region=IE")
    assert view == "stock_level" and params == {"region": "IE"} and free == ""


def test_knowledge_shared_between_instances(tmp_path):
    a, b = KnowledgeBase(tmp_path), KnowledgeBase(tmp_path)
    a.sync_views(["v1", "v2"])
    assert b.views() == {"v1": "", "v2": ""}
    a.set_view_description("v1", "first")
    assert b.views()["v1"] == "first"
    a.update_param("p", lambda k: setattr(k, "format", "digits") or True)
    assert b.param("p").format == "digits"


def _worker(tmp_path, llm, **learning):
    s = Settings(learning={"debounce_s": 0, **learning})
    kb = KnowledgeBase(tmp_path)
    kb.sync_views(["sales_daily"])
    return LearningWorker(kb, llm, FakeBackend().backend(), s, start=False), kb


def test_conflicting_signal_needs_threshold(tmp_path):
    llm = ScriptedLLM(ViewLearn=lambda u: {"description": "Forecast sales, not actuals.", "changed": True,
                                           "conflicts": True})
    w, kb = _worker(tmp_path, llm, conflict_threshold=2)
    kb.set_view_description("sales_daily", "Daily actual sales.")
    ev = LearningEvent(kind="view_corrected", request_text="forecast numbers next quarter", view="sales_daily")
    w.process([ev])
    assert kb.views()["sales_daily"] == "Daily actual sales."
    w.process([ev.model_copy(update={"request_text": "projected figures"})])
    assert kb.views()["sales_daily"] == "Forecast sales, not actuals."


def test_rewrite_adding_nothing_discarded(tmp_path):
    llm = ScriptedLLM(ViewLearn=lambda u: {"description": "Daily actual sales sales.", "changed": True})
    w, kb = _worker(tmp_path, llm)
    kb.set_view_description("sales_daily", "Daily actual sales.")
    w.process([LearningEvent(kind="view_confirmed", request_text="widgets by hour", view="sales_daily")])
    assert kb.views()["sales_daily"] == "Daily actual sales."


def test_bad_pattern_rejected(tmp_path):
    llm = ScriptedLLM(ParamLearn=lambda u: {"pattern": r"[A-Z]+", "changed": True})
    w, kb = _worker(tmp_path, llm)
    w.process([LearningEvent(kind="fetch_success", view="sales_daily", params={"store_id": "123"})])
    w.process([LearningEvent(kind="param_corrected_by_user", param="store_id", original_value="x",
                             final_value="456")])
    assert kb.param("store_id").pattern == ""


def test_date_format_learned_in_code(tmp_path):
    w, kb = _worker(tmp_path, ScriptedLLM(ViewLearn=lambda u: {"description": "", "changed": False}))
    w.process([LearningEvent(kind="fetch_success", view="sales_daily",
                             params={"date_from": "31/01/2024", "to_ccy": "GBP"})])
    assert kb.param("date_from").kind == "date" and kb.param("date_from").date_format == "%d/%m/%Y"
    assert kb.param("to_ccy").kind == "text"
