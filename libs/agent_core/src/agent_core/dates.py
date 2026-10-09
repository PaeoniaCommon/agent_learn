"""Deterministic date resolution and formatting. The LLM never calculates dates.

`resolve(text, today, cfg)` scans free text for date expressions and returns concrete
spans plus any date-like phrases it could not interpret. `resolve_value(s, today, cfg)`
parses a single value (e.g. an explicit param value or a reply to a question).

FY naming: a fiscal year is named by the calendar year it ends in (FY24 with an April
start = 2023-04-01 .. 2024-03-31). With a January start, FY24 == calendar 2024.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from typing import Callable

from dateutil.relativedelta import relativedelta

from .config import DateSettings
from .text import tokens

# --------------------------------------------------------------------------- helpers

_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3, "apr": 4,
    "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8,
    "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10, "october": 10,
    "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_NUM_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
MONTH = (r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
         r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)")
NUM = r"(?:\d{1,3}|" + "|".join(sorted(_NUM_WORDS, key=len, reverse=True)) + r")"
UNIT = r"(?:day|week|month|quarter|year)s?"
PERIOD_WORD = r"(?:week|month|quarter|year|fy|financial year|fiscal year)"
ORD = r"(?:st|nd|rd|th)?"


@dataclass
class DateSpan:
    id: str
    phrase: str
    start: str  # ISO date
    end: str    # ISO date
    grain: str

    def as_dict(self) -> dict:
        return asdict(self)


Span = tuple[date, date, str]


def _month_span(y: int, m: int) -> Span:
    start = date(y, m, 1)
    return start, start + relativedelta(months=1) - timedelta(days=1), "month"


def _quarter_span(y: int, q: int) -> Span:
    start = date(y, 3 * (q - 1) + 1, 1)
    return start, start + relativedelta(months=3) - timedelta(days=1), "quarter"


def _year_span(y: int) -> Span:
    return date(y, 1, 1), date(y, 12, 31), "year"


def _fy_span(end_year: int, start_month: int) -> Span:
    if start_month == 1:
        s, e, _ = _year_span(end_year)
        return s, e, "fy"
    start = date(end_year - 1, start_month, 1)
    return start, start + relativedelta(years=1) - timedelta(days=1), "fy"


def _fy_of(d: date, start_month: int) -> int:
    if start_month == 1:
        return d.year
    return d.year + 1 if d.month >= start_month else d.year


def _week_span(d: date, week_start: str) -> Span:
    ws = 0 if week_start == "monday" else 6
    start = d - timedelta(days=(d.weekday() - ws) % 7)
    return start, start + timedelta(days=6), "week"


def _num(s: str) -> int:
    s = s.lower()
    return _NUM_WORDS[s] if s in _NUM_WORDS else int(s)


def _year(s: str) -> int:
    s = s.lstrip("'")
    y = int(s)
    return 2000 + y if y < 100 else y


def _latest_month(m: int, today: date) -> int:
    """Year of the most recent occurrence of month m that is not in the future."""
    return today.year if m <= today.month else today.year - 1


class _Ctx:
    def __init__(self, today: date, cfg: DateSettings):
        self.today = today
        self.cfg = cfg


# --------------------------------------------------------------------------- point patterns
# Each handler: (match, ctx) -> Span | None. Patterns are used both for scanning
# free text and (anchored) for parsing single values / range endpoints.


def _h_today(m, c):
    w = m.group(0).lower()
    d = c.today + timedelta(days={"today": 0, "yesterday": -1, "tomorrow": 1}[w])
    return d, d, "day"


def _h_to_date(m, c):
    w = re.sub(r"\s+", " ", m.group(0).lower())
    t = c.today
    if w in ("ytd", "year to date"):
        return date(t.year, 1, 1), t, "range"
    if w in ("mtd", "month to date"):
        return date(t.year, t.month, 1), t, "range"
    if w in ("qtd", "quarter to date"):
        return _quarter_span(t.year, (t.month - 1) // 3 + 1)[0], t, "range"
    return _fy_span(_fy_of(t, c.cfg.fiscal_year_start_month), c.cfg.fiscal_year_start_month)[0], t, "range"


def _h_relative_period(m, c):
    rel, unit = m.group(1).lower(), m.group(2).lower()
    shift = {"this": 0, "current": 0, "last": -1, "previous": -1, "prior": -1, "next": 1}[rel]
    t = c.today
    if unit == "week":
        return _week_span(t + timedelta(weeks=shift), c.cfg.week_start)
    if unit == "month":
        d = t + relativedelta(months=shift)
        return _month_span(d.year, d.month)
    if unit == "quarter":
        d = t + relativedelta(months=3 * shift)
        return _quarter_span(d.year, (d.month - 1) // 3 + 1)
    if unit == "year":
        return _year_span(t.year + shift)
    sm = c.cfg.fiscal_year_start_month
    return _fy_span(_fy_of(t, sm) + shift, sm)


def _h_last_n(m, c):
    direction, n, unit = m.group(1).lower(), _num(m.group(2)), m.group(3).lower().rstrip("s")
    t = c.today
    delta = {"day": relativedelta(days=n), "week": relativedelta(weeks=n),
             "month": relativedelta(months=n), "quarter": relativedelta(months=3 * n),
             "year": relativedelta(years=n)}[unit]
    if direction == "next":
        return t, t + delta - timedelta(days=1), "range"
    return t - delta + timedelta(days=1), t, "range"


def _h_ago(m, c):
    n, unit = _num(m.group(1)), m.group(2).lower().rstrip("s")
    delta = {"day": relativedelta(days=n), "week": relativedelta(weeks=n),
             "month": relativedelta(months=n), "quarter": relativedelta(months=3 * n),
             "year": relativedelta(years=n)}[unit]
    d = c.today - delta
    return d, d, "day"


def _h_iso(m, c):
    d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return d, d, "day"


def _h_numeric(m, c):
    a, b, y = int(m.group(1)), int(m.group(2)), _year(m.group(3))
    if a > 12:
        day, mon = a, b
    elif b > 12:
        day, mon = b, a
    elif c.cfg.day_first:
        day, mon = a, b
    else:
        day, mon = b, a
    d = date(y, mon, day)
    return d, d, "day"


def _day_month(day: int, mon: int, year: str | None, c: _Ctx) -> Span:
    if year:
        d = date(_year(year), mon, day)
    else:
        d = date(c.today.year, mon, day)
        if d > c.today:
            d = date(c.today.year - 1, mon, day)
    return d, d, "day"


def _h_dmy(m, c):
    return _day_month(int(m.group(1)), _MONTHS[m.group(2).lower().rstrip(".")], m.group(3), c)


def _h_mdy(m, c):
    return _day_month(int(m.group(2)), _MONTHS[m.group(1).lower().rstrip(".")], m.group(3), c)


def _h_month_year(m, c):
    return _month_span(_year(m.group(2)), _MONTHS[m.group(1).lower()])


def _h_month(m, c):
    mon = _MONTHS[m.group(1).lower()]
    return _month_span(_latest_month(mon, c.today), mon)


def _h_quarter(m, c):
    q = int(m.group(1))
    if m.group(2):
        return _quarter_span(_year(m.group(2)), q)
    cur_q = (c.today.month - 1) // 3 + 1
    return _quarter_span(c.today.year if q <= cur_q else c.today.year - 1, q)


def _h_year_quarter(m, c):
    return _quarter_span(int(m.group(1)), int(m.group(2)))


def _h_fy(m, c):
    end = _year(m.group(2) or m.group(1))
    return _fy_span(end, c.cfg.fiscal_year_start_month)


def _h_year(m, c):
    return _year_span(int(m.group(1)))


_P = Callable[[re.Match, _Ctx], Span | None]

# (name, regex, handler, use_in_scan)  -- order = priority
_POINTS: list[tuple[str, str, _P, bool]] = [
    ("to_date", r"\b(?:ytd|mtd|qtd|fytd|year to date|month to date|quarter to date|"
                r"(?:fiscal|financial) year to date)\b", _h_to_date, True),
    ("ago", rf"\b({NUM})\s+({UNIT})\s+ago\b", _h_ago, True),
    ("last_n", rf"\b(last|past|previous|next)\s+({NUM})\s+({UNIT})\b", _h_last_n, True),
    ("relative", rf"\b(this|current|last|previous|prior|next)\s+({PERIOD_WORD})\b", _h_relative_period, True),
    ("iso", r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b", _h_iso, True),
    ("numeric", r"\b(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4}|\d{2})\b", _h_numeric, True),
    ("dmy", rf"\b(\d{{1,2}}){ORD}\s+(?:of\s+)?({MONTH})\.?(?:,?\s+(\d{{4}}))?(?!\d)\b", _h_dmy, True),
    ("mdy", rf"\b({MONTH})\.?\s+(\d{{1,2}}){ORD}(?!\d)(?:,?\s+(\d{{4}}))?\b", _h_mdy, True),
    ("month_year", rf"\b({MONTH})\.?[\s\-]+(\d{{4}}|'\d{{2}})\b", _h_month_year, True),
    ("quarter", r"\bq([1-4])(?:\s*[-/]?\s*(\d{4}|'?\d{2}))?\b", _h_quarter, True),
    ("year_quarter", r"\b(\d{4})\s*[-/]?\s*q([1-4])\b", _h_year_quarter, True),
    ("fy", r"\bfy\s*'?(\d{2}|\d{4})(?:\s*[/\-]\s*(\d{2}|\d{4}))?\b", _h_fy, True),
    ("today", r"\b(?:today|yesterday|tomorrow)\b", _h_today, True),
    # scan-only contextual forms (avoid "may" the verb and stray numbers)
    ("ctx_month", rf"\b(?:in|for|during|of|over)\s+({MONTH})\b",
     _h_month, True),
    ("month", rf"\b({MONTH})\b", _h_month, False),
    ("ctx_year", r"\b(?:in|for|during|year|of|over)\s+((?:19|20)\d{2})\b", _h_year, True),
    ("year", r"\b((?:19|20)\d{2})\b", _h_year, False),
]
_COMPILED = [(n, re.compile(rx, re.I), h, scan) for n, rx, h, scan in _POINTS]

# "month" in scan mode: any month name except bare "may"/"march" (verbs) needs context
_SCAN_MONTH = re.compile(rf"\b(?!may\b|march\b)({MONTH})\b", re.I)

_POINT_ALT = "|".join(f"(?:{rx})" for n, rx, _, _ in _POINTS
                      if n not in ("ctx_month", "ctx_year"))
_POINT_ALT = re.sub(r"\((?!\?)", "(?:", _POINT_ALT)  # make inner groups non-capturing
_EDGE = re.compile(r"\b(start|beginning|end)\s+of\s+(?:the\s+)?(.+)$", re.I)
_EDGE_SCAN = re.compile(rf"\b(?:start|beginning|end)\s+of\s+(?:the\s+)?(?:{_POINT_ALT})", re.I)
_RANGES = [
    re.compile(rf"\b(?:from|between)\s+((?:{_POINT_ALT}))\s+(?:to|and|until|till|through|-|–)\s+"
               rf"((?:{_POINT_ALT}))", re.I),
    re.compile(rf"((?:{_POINT_ALT}))\s+(?:-|–|to|until|till|through)\s+((?:{_POINT_ALT}))", re.I),
]
_SINCE = re.compile(rf"\bsince\s+((?:{_POINT_ALT}))", re.I)

_VAGUE = re.compile(
    rf"\b(?:around|about|circa|approx(?:imately)?)\s+\w+(?:\s+\w+)?"
    rf"|\b(?:early|mid|late)[\s\-]+{MONTH}\b"
    r"|\b(?:easter|christmas|xmas|new year|half[\s\-]?term|holidays?|(?:busy|peak|off)[\s\-]season|season)\b"
    r"|\b(?:recently|lately|the other day|a while ago|some time ago|soon)\b"
    r"|\b(?:h[12]|first half|second half)\b",
    re.I,
)


def _parse_point(s: str, c: _Ctx) -> Span | None:
    s = s.strip().rstrip(".,;")
    m = _EDGE.match(s)
    if m:
        inner = _parse_point(m.group(2), c)
        if inner is None:
            return None
        d = inner[0] if m.group(1).lower() in ("start", "beginning") else inner[1]
        return d, d, "day"
    for _, rx, handler, _scan in _COMPILED:
        mm = rx.fullmatch(s)
        if mm:
            try:
                return handler(mm, c)
            except (ValueError, KeyError):
                return None
    return None


def _blank(text: str, a: int, b: int) -> str:
    return text[:a] + " " * (b - a) + text[b:]


def today_in(cfg: DateSettings, clock: Callable[[], datetime] | None = None) -> date:
    if clock is not None:
        now = clock()
        return now.date() if isinstance(now, datetime) else now
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo(cfg.timezone)).date()
    except Exception:
        return datetime.now().date()


def resolve(text: str, today: date, cfg: DateSettings) -> tuple[list[DateSpan], list[str]]:
    """Find date expressions in free text. Returns (spans, unresolved_phrases)."""
    c = _Ctx(today, cfg)
    work = text
    found: list[tuple[int, str, Span]] = []

    def take(m: re.Match, span: Span | None) -> None:
        nonlocal work
        if span is not None:
            found.append((m.start(), text[m.start():m.end()], span))
            work = _blank(work, m.start(), m.end())

    for rx in _RANGES:
        for m in list(rx.finditer(work)):
            a, b = _parse_point(m.group(1), c), _parse_point(m.group(2), c)
            if a and b and a[0] <= b[1]:
                take(m, (a[0], b[1], "range"))
    for m in list(_SINCE.finditer(work)):
        a = _parse_point(m.group(1), c)
        if a:
            take(m, (a[0], today, "range"))
    for m in list(_EDGE_SCAN.finditer(work)):
        take(m, _parse_point(m.group(0), c))
    for name, rx, handler, scan in _COMPILED:
        if not scan:
            continue
        for m in list(rx.finditer(work)):
            if not work[m.start():m.end()].strip():
                continue
            try:
                take(m, handler(m, c))
            except (ValueError, KeyError):
                pass
    for m in list(_SCAN_MONTH.finditer(work)):
        take(m, _h_month(m, c))

    unresolved = [m.group(0).strip() for m in _VAGUE.finditer(text)]
    found.sort(key=lambda f: f[0])
    spans = [
        DateSpan(id=f"d{i + 1}", phrase=phrase.strip(), start=s.isoformat(), end=e.isoformat(), grain=g)
        for i, (_, phrase, (s, e, g)) in enumerate(found)
    ]
    return spans, unresolved


def resolve_value(value: str, today: date, cfg: DateSettings) -> DateSpan | None:
    """Parse a whole string as one date expression or range."""
    if not isinstance(value, str) or not value.strip():
        return None
    c = _Ctx(today, cfg)
    s = value.strip()
    for rx in _RANGES:
        m = rx.fullmatch(s)
        if m:
            a, b = _parse_point(m.group(1), c), _parse_point(m.group(2), c)
            if a and b:
                return DateSpan("v", s, a[0].isoformat(), b[1].isoformat(), "range")
    m = _SINCE.fullmatch(s)
    if m and (a := _parse_point(m.group(1), c)):
        return DateSpan("v", s, a[0].isoformat(), today.isoformat(), "range")
    p = _parse_point(s, c)
    if p:
        return DateSpan("v", s, p[0].isoformat(), p[1].isoformat(), p[2])
    return None


# --------------------------------------------------------------------------- params & formats

CANDIDATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d", "%d-%m-%Y", "%d.%m.%Y",
                     "%Y/%m/%d", "%d %b %Y", "%d-%b-%Y", "%Y-%m-%dT%H:%M:%S"]

_START_TOKENS = {"from", "start", "begin", "since", "min", "first"}
_END_TOKENS = {"to", "end", "until", "thru", "through", "max", "last"}
_DATE_TOKENS = {"date", "dt", "day", "period", "from", "to", "start", "end", "since", "until", "asof"}


def date_role(param: str) -> str | None:
    t = set(tokens(param))
    if t & _START_TOKENS:
        return "start"
    if t & _END_TOKENS:
        return "end"
    return None


def looks_like_date_param(param: str) -> bool:
    return bool(set(tokens(param)) & _DATE_TOKENS)


def format_date(iso: str, fmt: str) -> str:
    return date.fromisoformat(iso).strftime(fmt)


def infer_formats(value: str) -> list[str]:
    """Formats that parse `value` and round-trip to exactly the same string."""
    out = []
    for fmt in CANDIDATE_FORMATS:
        try:
            if datetime.strptime(value, fmt).strftime(fmt) == value:
                out.append(fmt)
        except (ValueError, TypeError):
            continue
    return out


def span_value(span: DateSpan | dict, part: str | None, fmt: str) -> str | None:
    """Pick start/end/single from a span and format it. None if 'single' is ambiguous."""
    s = span if isinstance(span, dict) else span.as_dict()
    if part == "start":
        return format_date(s["start"], fmt)
    if part == "end":
        return format_date(s["end"], fmt)
    if s["start"] == s["end"]:
        return format_date(s["start"], fmt)
    return None
