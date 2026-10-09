"""respond: build the reply in code and stream it."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from ..deps import Deps
from ..streaming import stream_text
from .common import human_date_format

_VIEW_SOURCE = {
    "user": "you specified it",
    "inferred": "chosen from your request",
    "user_confirmed": "you confirmed it",
}

_ERROR_HEAD = {
    "param_error": "The data request was rejected because of a parameter problem.",
    "view_error": "The data service rejected the view.",
    "no_data": "The request ran but returned no data.",
    "connection_error": "I couldn't reach the data service.",
    "unknown": "The data request failed.",
}


def _cell(v, n: int) -> str:
    s = str(v).replace("|", "\\|").replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _source_label(d: dict) -> str:
    src = d["source"]
    note = d.get("note")
    if src == "user":
        return "you specified"
    if src == "user_corrected":
        return f'corrected from "{d.get("original_value")}" — {note or "normalised"}'
    if src == "default":
        return "default"
    if src == "inferred":
        return "inferred from your request" + (f" ({note})" if note else "")
    if d.get("original_value") is not None:
        return f'you confirmed (originally "{d["original_value"]}")'
    return "you confirmed"


def _view_line(state: dict) -> str:
    src = state.get("view_source")
    if src == "user_corrected":
        label = state.get("view_note") or "corrected"
    else:
        label = _VIEW_SOURCE.get(src, "")
        if src == "inferred" and state.get("view_note"):
            label += f": {state['view_note']}"
    return f"View: `{state['view']}` ({label})"


def _params_table(state: dict, n: int) -> list[str]:
    params = state.get("params") or {}
    if not params:
        return ["Parameters: none"]
    lines = ["Parameters:", "| param | value | source |", "|---|---|---|"]
    for p, d in params.items():
        lines.append(f"| {p} | {_cell(d['value'], n)} | {_cell(_source_label(d), 120)} |")
    return lines


def _ignored(state: dict) -> list[str]:
    ign = state.get("ignored_params") or []
    if not ign:
        return []
    return ["Ignored: " + "; ".join(f"`{i['key']}` ({i['reason']})" for i in ign)]


def render_success(state: dict, deps: Deps) -> str:
    acfg = deps.settings.agent
    meta = deps.store.meta(state["dataset_id"])
    df = deps.store.get(state["dataset_id"])
    lines = [f"**Data retrieved** — id `{meta.id}`", _view_line(state)]
    lines += _params_table(state, acfg.sample_value_max_chars)
    lines += _ignored(state)
    lines.append(f"Size: {meta.n_rows:,} rows × {meta.n_cols} columns")
    cols = meta.columns[: acfg.max_columns_listed]
    more = meta.n_cols - len(cols)
    lines.append("Columns: " + ", ".join(cols) + (f", … (+{more} more)" if more > 0 else ""))
    if meta.n_rows:
        row = df.head(1).to_dict(orient="records")[0]
        shown = list(row.items())[: acfg.max_columns_listed]
        body = ", ".join(f"{k}: {_cell(repr(v) if isinstance(v, str) else v, acfg.sample_value_max_chars)}"
                         for k, v in shown)
        lines.append("Sample row: {" + body + (", …" if len(row) > len(shown) else "") + "}")
    else:
        lines.append("Sample row: (no rows)")
    return "\n".join(lines)


def _hint(p: str, deps: Deps, state: dict) -> str | None:
    pk = deps.kb.param(p)
    allowed = ((state.get("param_spec") or {}).get("allowed") or {}).get(p) or []
    if allowed:
        return f"`{p}` must be one of: " + ", ".join(map(str, allowed[:10])) + (" …" if len(allowed) > 10 else "")
    if pk.format:
        return f"`{p}`: {pk.format}"
    if pk.date_format:
        return f"`{p}` may need format `{human_date_format(pk.date_format)}`"
    return None


def render_error(state: dict, deps: Deps) -> str:
    err = state["error"]
    kind = err.get("kind")
    if kind == "no_view":
        lines = ["I couldn't find a view that matches your request."]
        if state.get("view_note"):
            lines.append(state["view_note"])
        views = deps.backend.views()
        lines.append("Available views include: " + ", ".join(f"`{v}`" for v in views[:15])
                     + (" …" if len(views) > 15 else ""))
        lines.append("Next step: name the view (e.g. `view: <name>`) or describe the data you need.")
        return "\n".join(lines)

    if kind == "invalid_params":
        lines = [f"I couldn't build a valid request for `{state.get('view')}`:"]
        for d in err.get("detail") or []:
            val = f" (`{d['value']}`)" if d.get("value") is not None else ""
            hint = _hint(d["name"], deps, state)
            lines.append(f"- **{d['name']}**{val}: {d['reason']}" + (f" — {hint}" if hint else ""))
        lines.append("Next step: send the missing or corrected values, e.g. `@name=value`.")
        return "\n".join(lines)

    lines = [_ERROR_HEAD.get(kind, _ERROR_HEAD["unknown"])]
    if state.get("view"):
        lines.append(_view_line(state))
    lines += _params_table(state, deps.settings.agent.sample_value_max_chars)
    msg = " ".join(str(err.get("message", "")).split())
    lines.append(f"Service message: {msg[:300]}{'…' if len(msg) > 300 else ''}")
    if kind == "param_error":
        hints = [h for p in (err.get("detail") or {}).get("params", []) if (h := _hint(p, deps, state))]
        lines.append("Next step: " + ("; ".join(hints) if hints else "check the parameter values and try again."))
    elif kind == "no_data":
        lines.append("Next step: try widening the date range or relaxing filters.")
    elif kind == "connection_error":
        lines.append("Next step: try again in a moment.")
    elif kind == "view_error":
        lines.append("Next step: check the view name or pick another view.")
    else:
        lines.append("Next step: try again, or adjust the request.")
    return "\n".join(lines)


def respond(state: dict, deps: Deps) -> dict:
    text = render_error(state, deps) if state.get("error") else render_success(state, deps)
    messages = []
    if deps.settings.respond.respond_with_llm:
        try:
            intro = deps.llm.chat_model().invoke([
                SystemMessage(content="In one or two friendly sentences, summarise this result for the user. "
                                      "Do not repeat ids, values or tables; they follow your text."),
                HumanMessage(content=text),
            ])
            messages.append(intro)
        except Exception:
            pass
    messages.append(stream_text(deps.streamer, text))
    out = {"messages": messages, "requested_view": None, "requested_params": None}
    if not state.get("error") and state.get("dataset_id"):
        out["last_request"] = {"view": state["view"],
                               "params": {p: d["value"] for p, d in (state.get("params") or {}).items()}}
    return out
