"""fetch: call get_data, store the DataFrame, emit learning events."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from agent_core.graph import progress

from ..deps import Deps
from ..learning import LearningEvent
from ..validation import mentioned_params

log = logging.getLogger(__name__)


def classify_error(exc: BaseException, params: dict) -> tuple[str, list[str]]:
    msg = str(exc) or exc.__class__.__name__
    low = msg.lower()
    mentioned = mentioned_params(msg, params)
    if any(w in low for w in ("no data", "no rows", "no records", "empty result", "nothing found")):
        return "no_data", mentioned
    if isinstance(exc, (ConnectionError, TimeoutError)) or any(
            w in low for w in ("connection", "timed out", "timeout", "unreachable", "refused", "unavailable")):
        return "connection_error", mentioned
    if mentioned or any(w in low for w in ("param", "invalid", "format", "expected", "must be", "not allowed",
                                           "unparseable", "parse", "missing")):
        return "param_error", mentioned
    if "view" in low:
        return "view_error", mentioned
    return "unknown", mentioned


def fetch(state: dict, deps: Deps) -> dict:
    view = state["view"]
    decisions = state.get("params") or {}
    param_dict = {p: d["value"] for p, d in decisions.items()}
    progress("fetching", view=view)
    try:
        df = deps.backend.get_data(view, param_dict)
        if df is None:
            raise RuntimeError("get_data returned no result")
    except Exception as exc:  # noqa: BLE001 - report every backend failure to the user
        log.exception("get_data failed for view=%s params=%s", view, param_dict)
        kind, mentioned = classify_error(exc, param_dict)
        message = str(exc) or exc.__class__.__name__
        deps.learning.submit(LearningEvent(kind="fetch_error", request_text=state.get("request_text", ""),
                                           view=view, params=param_dict, error_message=message))
        return {"error": {"kind": kind, "message": message, "detail": {"params": mentioned,
                                                                        "type": exc.__class__.__name__}}}

    dataset_id = deps.store.put(df, view=view, params=param_dict, retrieved_at=datetime.now(timezone.utc))
    deps.learning.submit(LearningEvent(kind="fetch_success", request_text=state.get("request_text", ""),
                                       view=view, params=param_dict, columns=[str(c) for c in df.columns]))
    for p, d in decisions.items():
        if d.get("fixed_by") == "llm" and d.get("source") == "user_corrected":
            deps.learning.submit(LearningEvent(kind="param_llm_fix_succeeded", view=view, param=p,
                                               original_value=d.get("original_value"), final_value=d["value"],
                                               request_text=state.get("request_text", "")))
    return {"dataset_id": dataset_id}
