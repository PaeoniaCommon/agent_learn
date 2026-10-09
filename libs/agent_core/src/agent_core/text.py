"""Small text helpers shared by agents."""

from __future__ import annotations

import re
from typing import Any


def norm(s: Any) -> str:
    """Case-fold and drop separators: 'Sales Daily' == 'sales_daily' == 'sales-daily'."""
    return re.sub(r"[\s_\-.]+", "", str(s).strip().casefold())


def tokens(name: str) -> list[str]:
    """Split an identifier into lowercase tokens: 'dateFrom' / 'date_from' -> ['date', 'from']."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]
