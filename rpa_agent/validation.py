"""Deterministic matching, normalisation and value validation (no LLM)."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Iterable


def norm(s: Any) -> str:
    """Case-fold and drop separators: 'Sales Daily' == 'sales_daily' == 'sales-daily'."""
    return re.sub(r"[\s_\-.]+", "", str(s).strip().casefold())


def match_name(candidate: str, names: Iterable[str], cutoff: float) -> tuple[str | None, bool]:
    """Match a user-supplied name to one of `names`.

    Returns (match, changed). `changed` is True when the match is not an exact string match.
    Fuzzy matching only accepts a single, unambiguous close match.
    """
    names = list(names)
    if candidate in names:
        return candidate, False
    key = norm(candidate)
    normed = [n for n in names if norm(n) == key]
    if len(normed) == 1:
        return normed[0], True
    if len(normed) > 1:
        return None, False
    by_norm = {norm(n): n for n in names}
    close = difflib.get_close_matches(key, list(by_norm), n=2, cutoff=cutoff)
    if len(close) == 1:
        return by_norm[close[0]], True
    if len(close) == 2:
        # accept only if clearly better than the runner-up
        r0 = difflib.SequenceMatcher(None, key, close[0]).ratio()
        r1 = difflib.SequenceMatcher(None, key, close[1]).ratio()
        if r0 - r1 >= 0.05:
            return by_norm[close[0]], True
    return None, False


@dataclass
class ValueCheck:
    ok: bool
    value: Any = None          # canonical value to use when ok
    changed: bool = False      # canonical value differs from the input
    reason: str | None = None  # why it failed / what changed


def check_value(value: Any, allowed: list, pattern: str | None = None) -> ValueCheck:
    """Validate a value against an allowed-values list and/or a learned regex."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return ValueCheck(False, reason="no value given")
    if allowed:
        if value in allowed:
            return ValueCheck(True, value)
        sval = str(value).strip()
        for a in allowed:
            if str(a) == sval:
                return ValueCheck(True, a, changed=type(a) is not type(value))
        for a in allowed:
            if norm(a) == norm(sval):
                return ValueCheck(True, a, changed=True, reason=f'matched allowed value "{a}"')
        return ValueCheck(False, reason="not an accepted value")
    if pattern:
        try:
            if not re.fullmatch(pattern, str(value).strip()):
                return ValueCheck(False, reason="does not match the known format")
        except re.error:
            pass  # a broken learned pattern must never block a request
    return ValueCheck(True, value)


def tokens(name: str) -> list[str]:
    """Split a param name into lowercase tokens: 'dateFrom' / 'date_from' -> ['date', 'from']."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]
