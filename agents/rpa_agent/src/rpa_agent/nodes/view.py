"""resolve_view, ask_view, confirm_view."""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from agent_core.graph import progress
from agent_core.streaming import stream_text

from .. import prompts
from ..deps import Deps
from ..learning import LearningEvent
from ..validation import match_name
from .common import answer_text


def _shortlist(views: list[str], descriptions: dict[str, str], request: str, hint: str | None, limit: int) -> list[str]:
    if len(views) <= limit:
        return views
    req = set(re.findall(r"[a-z0-9]+", f"{request} {hint or ''}".lower()))

    def score(v: str) -> int:
        words = set(re.findall(r"[a-z0-9]+", f"{v.replace('_', ' ')} {descriptions.get(v, '')}".lower()))
        return len(req & words)

    return sorted(views, key=score, reverse=True)[:limit]


def resolve_view(state: dict, deps: Deps) -> dict:
    views = deps.backend.views()
    acfg = deps.settings.agent
    explicit = state.get("explicit_view")
    hint = None

    if explicit:
        match, changed = match_name(str(explicit), views, acfg.fuzzy_match_cutoff)
        if match and deps.backend.validate_view(match):
            return {
                "view": match,
                "view_source": "user_corrected" if changed else "user",
                "view_note": f'you wrote "{explicit}"; matched to `{match}`' if changed else None,
            }
        hint = str(explicit)

    progress("selecting_view")
    names = _shortlist(views, deps.kb.views(), state.get("request_text", ""), hint, acfg.view_shortlist_max)
    last = state.get("last_request")
    user = prompts.VIEW_USER.format(
        request=state.get("free_text") or state.get("request_text", ""),
        hint=f'User named the view "{hint}" but it is not valid; it is a strong hint.\n' if hint else "",
        last=f"Previous request: {json.dumps(last)}\n" if last else "",
        views=deps.kb.views_prompt(names),
    )
    choice = deps.llm.structured(prompts.ViewChoice, prompts.VIEW_SYSTEM, user, step="select_view")

    seen, cands = set(), []
    for c in choice.candidates:
        if c.view in views and c.view not in seen and deps.backend.validate_view(c.view):
            seen.add(c.view)
            cands.append(c)
    if not cands:
        note = f'"{hint}" isn\'t a valid view.' if hint else None
        return {"view": None, "view_note": note,
                "error": {"kind": "no_view", "message": "No view matches the request."}}

    high = [c for c in cands if c.confidence == "high"]
    if len(high) == 1 and cands[0] is high[0] and not choice.needs_confirmation:
        chosen = high[0]
        if hint:
            return {"view": chosen.view, "view_source": "user_corrected",
                    "view_note": f'"{hint}" isn\'t a valid view; I used `{chosen.view}` instead ({chosen.reason})'}
        return {"view": chosen.view, "view_source": "inferred", "view_note": chosen.reason}

    return {
        "view": None,
        "view_candidates": [c.model_dump() for c in cands[: acfg.max_options_listed]],
        "view_note": f'"{hint}" isn\'t a valid view.' if hint else None,
    }


def route_after_resolve_view(state: dict) -> str:
    if state.get("error"):
        return "respond"
    return "resolve_params" if state.get("view") else "ask_view"


def _view_question(state: dict) -> str:
    lines = []
    if state.get("view_reasks"):
        lines.append("Sorry, I couldn't tell which one you meant.")
    elif state.get("view_note"):
        lines.append(state["view_note"])
    lines.append("Which view should I use?")
    for i, c in enumerate(state.get("view_candidates", []), 1):
        lines.append(f"{i}. `{c['view']}` — {c['reason']}")
    lines.append("Reply with the number or the view name.")
    return "\n".join(lines)


def ask_view(state: dict, deps: Deps) -> dict:
    return {"messages": [stream_text(deps.streamer, _view_question(state))]}


def _match_answer(answer, cands: list[dict], deps: Deps) -> str | None:
    options = [c["view"] for c in cands]
    if isinstance(answer, dict) and answer.get("view"):
        answer = answer["view"]
    elif isinstance(answer, dict) and answer.get("id") is not None:
        answer = str(answer["id"])
    text = str(answer).strip().strip(".`'\"")
    m = re.fullmatch(r"(?:option\s*)?#?(\d+)", text, re.I)
    if m and 1 <= int(m.group(1)) <= len(options):
        return options[int(m.group(1)) - 1]
    cutoff = deps.settings.agent.fuzzy_match_cutoff
    match, _ = match_name(text, options, cutoff)
    if match:
        return match
    views = deps.backend.views()
    match, _ = match_name(text, views, cutoff)
    if match and deps.backend.validate_view(match):
        return match
    numbered = "\n".join(f"{i}. {v}" for i, v in enumerate(options, 1))
    out = deps.llm.structured(prompts.ViewAnswer, prompts.VIEW_ANSWER_SYSTEM,
                              f"Options:\n{numbered}\nUser reply: {text}", step="confirm_view")
    if out.view in options:
        return out.view
    if out.view and out.view in views and deps.backend.validate_view(out.view):
        return out.view
    return None


def confirm_view(state: dict, deps: Deps) -> dict:
    cands = state.get("view_candidates", [])
    answer = interrupt({
        "type": "confirm_view",
        "question": "Which view should I use?",
        "options": [{"id": i, "view": c["view"], "reason": c["reason"]} for i, c in enumerate(cands, 1)],
        "message": _view_question(state),
    })
    human = HumanMessage(content=answer_text(answer))
    chosen = _match_answer(answer, cands, deps)
    if chosen:
        options = [c["view"] for c in cands]
        deps.learning.submit(LearningEvent(
            kind="view_confirmed" if chosen in options else "view_corrected",
            request_text=state.get("request_text", ""),
            view=chosen,
            rejected_views=[v for v in options if v != chosen],
        ))
        return {"messages": [human], "view": chosen, "view_source": "user_confirmed",
                "view_note": None, "view_candidates": []}
    reasks = state.get("view_reasks", 0) + 1
    if reasks > deps.settings.agent.max_view_reasks:
        return {"messages": [human], "view_reasks": reasks,
                "error": {"kind": "no_view", "message": "I couldn't tell which view you meant."}}
    return {"messages": [human], "view_reasks": reasks}


def route_after_confirm_view(state: dict) -> str:
    if state.get("error"):
        return "respond"
    return "resolve_params" if state.get("view") else "ask_view"
