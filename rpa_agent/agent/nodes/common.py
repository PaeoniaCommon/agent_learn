"""Helpers shared by nodes."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage

from ...validation import dates
from ...validation.values import ValueCheck, check_value

YES = {"y", "yes", "ok", "okay", "accept", "correct", "right", "yep", "yeah", "sure", "confirm", "confirmed"}


def message_text(msg: BaseMessage | Any) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def answer_text(answer: Any) -> str:
    if isinstance(answer, dict):
        for key in ("text", "value", "view", "id"):
            if key in answer and answer[key] is not None:
                return str(answer[key])
        if answer.get("accept"):
            return "yes"
        return str(answer)
    return str(answer)


def human_date_format(fmt: str) -> str:
    out = fmt
    for a, b in (("%Y", "YYYY"), ("%m", "MM"), ("%d", "DD"), ("%b", "Mon"), ("%H", "hh"), ("%M", "mm"), ("%S", "ss")):
        out = out.replace(a, b)
    return out


def check_param_value(
    value: Any,
    *,
    allowed: list,
    pattern: str | None,
    is_date: bool,
    date_fmt: str,
    role: str | None,
    today,
    date_cfg,
) -> ValueCheck:
    """Validate one value; date params are resolved and formatted in code."""
    if is_date and isinstance(value, str) and value.strip():
        sval = value.strip()
        span = dates.resolve_value(sval, today, date_cfg)
        if span is not None:
            v = dates.span_value(span, role, date_fmt)
            if v is None:
                return ValueCheck(False, reason=f'"{sval}" covers {span.start} to {span.end}; which single date do you mean?')
            res = check_value(v, [], pattern)
            if res.ok and v != sval:
                res.changed = True
                res.reason = (f"reformatted to {human_date_format(date_fmt)}" if dates.infer_formats(sval)
                              else f"date resolved to {span.start}..{span.end}" if span.start != span.end
                              else "date resolved")
            return res
    return check_value(value, allowed, pattern)
