from datetime import date

import pytest

from agent_core.config import DateSettings
from agent_core.dates import date_role, infer_formats, looks_like_date_param, resolve, resolve_value

TODAY = date(2026, 10, 9)
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
def test_resolve(text, start, end):
    spans, unresolved = resolve(text, TODAY, CFG)
    assert len(spans) == 1 and (spans[0].start, spans[0].end) == (start, end) and not unresolved


def test_unresolved_and_false_positives():
    spans, unresolved = resolve("sales around Easter", TODAY, CFG)
    assert spans == [] and unresolved == ["around Easter"]
    assert resolve("I may need it", TODAY, CFG)[0] == []
    assert resolve("store 12 sold 2024 units", TODAY, CFG)[0] == []


def test_month_first_config():
    assert resolve_value("03/04/2024", TODAY, DateSettings(day_first=False)).start == "2024-03-04"


def test_infer_formats():
    assert infer_formats("2024-01-31") == ["%Y-%m-%d"]
    assert "%d/%m/%Y" in infer_formats("31/01/2024")
    assert infer_formats("GBP") == []


def test_param_name_heuristics():
    assert date_role("date_from") == "start" and date_role("periodEnd") == "end" and date_role("as_of_date") is None
    assert looks_like_date_param("as_of_date") and not looks_like_date_param("region")
