"""Graph state. Everything here is JSON-serialisable (DataFrames live in the DataStore)."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

ParamSource = Literal["user", "user_corrected", "default", "inferred", "user_confirmed"]
ViewSource = Literal["user", "user_corrected", "inferred", "user_confirmed"]


def decision(
    name: str,
    value: Any,
    source: ParamSource,
    *,
    original_value: Any = None,
    note: str | None = None,
    fixed_by: Literal["code", "llm", "user"] | None = None,
) -> dict:
    """A ParamDecision (stored as a plain dict so the checkpointer can serialise it)."""
    return {
        "name": name, "value": value, "source": source, "original_value": original_value,
        "note": note, "fixed_by": fixed_by,
    }


class RPAState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    # explicit inputs (may be passed alongside `messages`; cleared after each request)
    requested_view: str | None
    requested_params: dict[str, Any] | None
    # per-request
    request_text: str
    free_text: str
    explicit_view: str | None
    explicit_params: dict[str, Any]
    view: str | None
    view_source: ViewSource | None
    view_note: str | None
    view_candidates: list[dict]
    view_reasks: int
    param_spec: dict
    params: dict[str, dict]
    ignored_params: list[dict]
    invalid_params: list[dict]
    pending_questions: list[dict]
    date_spans: list[dict]
    unresolved_dates: list[str]
    dataset_id: str | None
    error: dict | None
    # conversation-level
    last_request: dict | None
