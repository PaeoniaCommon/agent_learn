"""ingest: reset per-request state, parse explicit view/params, resolve dates. No LLM."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage

from ...validation import dates
from ..deps import Deps
from .common import message_text

_PARAMS_START = re.compile(r"(?:^|(?<=\s))params\s*[:=]\s*(?=\{)", re.I)
_AT_PAIR = re.compile(r"(?:^|(?<=\s))@([A-Za-z_][\w.\-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|(\S+))")
_VIEW = re.compile(r"(?:^|(?<=\s))view\s*[:=]\s*(?:\"([^\"]+)\"|'([^']+)'|([A-Za-z0-9_.\-]+))", re.I)


def _balanced(text: str, start: int) -> int | None:
    depth, in_str, esc = 0, None, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == in_str:
                in_str = None
        elif ch in "\"'":
            in_str = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def parse_explicit(text: str) -> tuple[str | None, dict[str, Any], str]:
    """Extract `view: X`, `params: {...}` and `@name=value` from a message.

    Returns (view, params, free_text) where free_text has the explicit syntax removed.
    """
    params: dict[str, Any] = {}
    view = None
    cuts: list[tuple[int, int]] = []

    m = _PARAMS_START.search(text)
    if m:
        end = _balanced(text, m.end())
        if end:
            blob = text[m.end():end]
            parsed = None
            try:
                parsed = json.loads(blob)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(blob)
                except (ValueError, SyntaxError):
                    parsed = None
            if isinstance(parsed, dict):
                params.update(parsed)
                cuts.append((m.start(), end))

    for m in _AT_PAIR.finditer(text):
        if any(a <= m.start() < b for a, b in cuts):
            continue
        key = m.group(1)
        val = next(g for g in m.groups()[1:] if g is not None)
        if key.lower() == "view":
            view = val
        else:
            params[key] = val
        cuts.append((m.start(), m.end()))

    if view is None:
        for m in _VIEW.finditer(text):
            if any(a <= m.start() < b for a, b in cuts):
                continue
            view = next(g for g in m.groups() if g is not None)
            cuts.append((m.start(), m.end()))
            break

    free = text
    for a, b in sorted(cuts, reverse=True):
        free = free[:a] + " " + free[b:]
    free = re.sub(r"\s+", " ", free).strip(" ,;")
    return view, params, free


def ingest(state: dict, deps: Deps) -> dict:
    last_human = next((m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)), None)
    text = message_text(last_human) if last_human is not None else ""
    inline_view, inline_params, free = parse_explicit(text)

    explicit_view = state.get("requested_view") or inline_view
    explicit_params = {**inline_params, **(state.get("requested_params") or {})}

    spans, unresolved = dates.resolve(free, deps.today(), deps.settings.dates)
    return {
        "request_text": text,
        "free_text": free,
        "explicit_view": explicit_view,
        "explicit_params": explicit_params,
        "view": None,
        "view_source": None,
        "view_note": None,
        "view_candidates": [],
        "view_reasks": 0,
        "param_spec": {},
        "params": {},
        "ignored_params": [],
        "invalid_params": [],
        "pending_questions": [],
        "date_spans": [s.as_dict() for s in spans],
        "unresolved_dates": unresolved,
        "dataset_id": None,
        "error": None,
    }
