"""Adapter over the host RPA functions.

This is the only module that touches RPA_VIEWS / validate_view / get_view_params /
valid_params / get_data. If the host signatures change, change them here.
"""

from __future__ import annotations

import importlib
import threading
import time
from typing import Any, Callable


class Backend:
    def __init__(
        self,
        rpa_views: list[str] | Callable[[], list[str]],
        validate_view: Callable[[str], bool],
        get_view_params: Callable[[str], dict],
        valid_params: Callable[[str], list],
        get_data: Callable[[str, dict], Any],
        cache_ttl_s: float = 300,
    ):
        self._views = rpa_views
        self._validate_view = validate_view
        self._get_view_params = get_view_params
        self._valid_params = valid_params
        self._get_data = get_data
        self._ttl = cache_ttl_s
        self._cache: dict[tuple, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_module(cls, module: str, **kwargs) -> "Backend":
        mod = importlib.import_module(module)
        return cls(
            rpa_views=getattr(mod, "RPA_VIEWS"),
            validate_view=getattr(mod, "validate_view"),
            get_view_params=getattr(mod, "get_view_params"),
            valid_params=getattr(mod, "valid_params"),
            get_data=getattr(mod, "get_data"),
            **kwargs,
        )

    def _cached(self, key: tuple, fn: Callable[[], Any]) -> Any:
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(key)
            if hit and now - hit[0] < self._ttl:
                return hit[1]
        value = fn()
        with self._lock:
            self._cache[key] = (now, value)
        return value

    # -- views -------------------------------------------------------------
    def views(self) -> list[str]:
        views = self._views() if callable(self._views) else self._views
        return list(views)

    def validate_view(self, view: str) -> bool:
        try:
            return bool(self._validate_view(view))
        except Exception:
            return False

    def view_params(self, view: str) -> dict:
        def load() -> dict:
            raw = self._get_view_params(view) or {}
            return {
                "mandatory": list(raw.get("mandatory") or []),
                "optional": list(raw.get("optional") or []),
                "default": dict(raw.get("default") or {}),
            }
        return self._cached(("view_params", view), load)

    # -- params (view independent) ------------------------------------------
    def allowed_values(self, param: str) -> list:
        """Allowed values for a param; [] means no fixed list."""
        return self._cached(("valid_params", param), lambda: list(self._valid_params(param) or []))

    # -- data ---------------------------------------------------------------
    def get_data(self, view: str, params: dict):
        return self._get_data(view, params)
