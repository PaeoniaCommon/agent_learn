"""resolve_params, ask_param, confirm_param, finalize_params."""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from agent_core import dates
from agent_core.graph import progress
from agent_core.streaming import stream_text

from .. import prompts
from ..deps import Deps
from ..learning import LearningEvent
from ..state import decision
from ..validation import match_name
from .common import YES, answer_text, check_param_value, human_date_format

_FILLER = {
    "please", "pls", "thanks", "thank", "you", "can", "could", "would", "get", "me", "the", "data",
    "for", "pull", "fetch", "show", "give", "i", "want", "need", "a", "an", "of", "and", "with", "run",
    "it", "this", "that", "to", "from", "in", "on", "use", "using", "view", "params", "hi", "hello",
}


class _ParamCtx:
    """Per-request view of params: allowed lists, knowledge, date handling."""

    def __init__(self, state: dict, deps: Deps):
        self.deps = deps
        spec = deps.backend.view_params(state["view"])
        self.mandatory: list[str] = spec["mandatory"]
        self.optional: list[str] = [p for p in spec["optional"] if p not in spec["mandatory"]]
        self.defaults: dict[str, Any] = spec["default"]
        self.accepted = self.mandatory + self.optional
        self.allowed = {p: deps.backend.allowed_values(p) for p in self.accepted}
        for p in self.accepted:
            deps.kb.ensure_param(p)
        self.know = {p: deps.kb.param(p) for p in self.accepted}
        self.today = deps.today()
        self.date_cfg = deps.settings.dates

    def is_date(self, p: str) -> bool:
        k = self.know[p]
        return not self.allowed[p] and (k.kind == "date" or (k.kind == "" and dates.looks_like_date_param(p)))

    def date_fmt(self, p: str) -> str:
        return self.know[p].date_format or self.date_cfg.default_date_format

    def check(self, p: str, value: Any):
        return check_param_value(
            value, allowed=self.allowed[p], pattern=self.know[p].pattern or None, is_date=self.is_date(p),
            date_fmt=self.date_fmt(p), role=dates.date_role(p), today=self.today, date_cfg=self.date_cfg)

    def format_hint(self, p: str) -> str | None:
        k = self.know[p]
        if k.format:
            return k.format
        if self.is_date(p):
            return human_date_format(self.date_fmt(p))
        return None

    def question(self, p: str, *, proposed=None, original=None, reason: str) -> dict:
        max_opts = self.deps.settings.agent.max_options_listed
        return {
            "name": p, "proposed": proposed, "original": original, "reason": reason,
            "options": [str(a) for a in self.allowed[p][:max_opts]],
            "n_options": len(self.allowed[p]),
            "format_hint": self.format_hint(p), "reasks": 0,
        }

    def spec_dict(self) -> dict:
        return {"mandatory": self.mandatory, "optional": self.optional, "default": self.defaults,
                "allowed": self.allowed, "date_params": [p for p in self.accepted if self.is_date(p)]}


def _has_request_words(free: str, spans: list[dict]) -> bool:
    for s in spans:
        free = free.replace(s["phrase"], " ")
    words = re.findall(r"[A-Za-z0-9]+", free.lower())
    return any(w not in _FILLER for w in words)


def resolve_params(state: dict, deps: Deps) -> dict:
    ctx = _ParamCtx(state, deps)
    acfg = deps.settings.agent
    spans: list[dict] = state.get("date_spans", [])
    params: dict[str, dict] = {}
    failing: dict[str, dict] = {}
    unmapped: dict[str, Any] = {}
    ignored: list[dict] = []
    questions: dict[str, dict] = {}
    handled: set[str] = set()  # unmapped keys the LLM mapped to a param

    # 1. explicit params: deterministic validation
    for key, val in (state.get("explicit_params") or {}).items():
        name, renamed = match_name(str(key), ctx.accepted, acfg.fuzzy_match_cutoff)
        if name is None:
            unmapped[key] = val
            continue
        if name in params or name in failing:
            continue
        res = ctx.check(name, val)
        if not res.ok:
            failing[name] = {"value": val, "reason": res.reason}
        elif res.changed or renamed:
            notes = []
            if renamed:
                notes.append(f'"{key}" read as {name}')
            if res.changed:
                notes.append(res.reason or "normalised")
            params[name] = decision(name, res.value, "user_corrected", original_value=val,
                                    note="; ".join(notes), fixed_by="code")
        else:
            params[name] = decision(name, res.value, "user")

    # 2. dates from free text: assign deterministically when unambiguous
    unset_dates = [p for p in ctx.accepted if ctx.is_date(p) and p not in params and p not in failing]
    if len(spans) == 1 and unset_dates:
        sp = spans[0]
        for p in unset_dates:
            v = dates.span_value(sp, dates.date_role(p), ctx.date_fmt(p))
            if v is not None and ctx.check(p, v).ok:
                params[p] = decision(p, v, "inferred", note=f'from "{sp["phrase"]}"', fixed_by="code")

    # 3. LLM only when something is missing, invalid, unmapped or implied by free text
    missing = [p for p in ctx.mandatory if p not in params and p not in ctx.defaults]
    unset = [p for p in ctx.accepted if p not in params]
    need_llm = bool(failing or unmapped or [p for p in missing if p not in failing]
                    or (unset and _has_request_words(state.get("free_text", ""), spans)))
    if need_llm:
        progress("resolving_params", view=state["view"])
        proposal = _ask_llm(state, ctx, params, failing, unmapped, missing)
        _apply_proposal(proposal, state, ctx, params, failing, unmapped, ignored, questions, handled)

    # 4. leftovers: unfixed failures, defaults, missing mandatory, unresolved dates
    for p, f in failing.items():
        if p not in params and p not in questions:
            questions[p] = ctx.question(p, original=f["value"], reason=f["reason"] or "not valid")
    for key in unmapped:
        if key not in handled and not any(i["key"] == key for i in ignored):
            ignored.append({"key": key, "reason": "not a parameter of this view"})
    for p, v in ctx.defaults.items():
        if p in ctx.accepted and p not in params and p not in questions:
            params[p] = decision(p, v, "default")
    unresolved = state.get("unresolved_dates") or []
    for p in ctx.mandatory:
        if p not in params and p not in questions:
            reason = "required"
            if unresolved and ctx.is_date(p):
                reason = f'I couldn\'t interpret "{unresolved[0]}" as a date'
            questions[p] = ctx.question(p, reason=reason)
    if unresolved and not any(ctx.is_date(q) for q in questions):
        for p in ctx.optional:
            if ctx.is_date(p) and p not in params and p not in questions:
                questions[p] = ctx.question(p, reason=f'I couldn\'t interpret "{unresolved[0]}" as a date')

    order = {p: i for i, p in enumerate(ctx.accepted)}
    pending = sorted(questions.values(), key=lambda q: (q["name"] not in ctx.mandatory, order.get(q["name"], 0)))
    return {"param_spec": ctx.spec_dict(), "params": params, "ignored_params": ignored,
            "pending_questions": pending, "invalid_params": []}


def _ask_llm(state, ctx: _ParamCtx, params, failing, unmapped, missing) -> prompts.ParamProposal:
    max_vals = ctx.deps.settings.agent.max_valid_values_in_prompt
    lines = []
    for p in ctx.accepted:
        line = f"- {p} [{'mandatory' if p in ctx.mandatory else 'optional'}]"
        allowed = ctx.allowed[p]
        if allowed:
            shown = ", ".join(map(str, allowed[:max_vals]))
            more = f" …({len(allowed) - max_vals} more; value must be from list)" if len(allowed) > max_vals else ""
            line += f" allowed={shown}{more}"
        if ctx.is_date(p):
            line += " (date)"
        if p in ctx.defaults:
            line += f" default={ctx.defaults[p]}"
        lines.append(line)
    knowledge = ctx.deps.kb.params_prompt(ctx.accepted)
    spans = state.get("date_spans") or []
    failing_lines = [f"- {p}: user gave {json.dumps(f['value'], default=str)} — {f['reason']}"
                     for p, f in failing.items()]
    failing_lines += [f"- {p}: missing (mandatory)" for p in missing if p not in failing]
    last = state.get("last_request")
    user = prompts.PARAM_USER.format(
        request=state.get("free_text") or state.get("request_text", ""),
        view=state["view"],
        last=f"Previous request: {json.dumps(last, default=str)}" if last else "",
        params="\n".join(lines),
        knowledge=f"Param notes:\n{knowledge}" if knowledge else "",
        spans="\n".join(f'{s["id"]}: "{s["phrase"]}" = {s["start"]}..{s["end"]}' for s in spans) or "(none)",
        fixed=json.dumps({p: d["value"] for p, d in params.items()}, default=str) or "{}",
        failing="\n".join(failing_lines) or "(none)",
        unmapped=json.dumps(unmapped, default=str) if unmapped else "(none)",
    )
    return ctx.deps.llm.structured(prompts.ParamProposal, prompts.PARAM_SYSTEM, user, step="resolve_params")


def _apply_proposal(proposal, state, ctx: _ParamCtx, params, failing, unmapped, ignored, questions, handled) -> None:
    spans = {s["id"]: s for s in state.get("date_spans") or []}
    for pv in proposal.params:
        p = pv.name
        if p not in ctx.accepted or p in params:
            continue  # unknown or FIXED
        original = failing.get(p, {}).get("value")
        if ctx.is_date(p):
            # The LLM never supplies a date value: only a reference to a span resolved in code.
            sp = spans.get(pv.span or "")
            if sp is None and pv.value:
                lit = dates.resolve_value(pv.value, ctx.today, ctx.date_cfg)
                sp = next((s for s in spans.values() if lit and s["start"] == lit.start and s["end"] == lit.end), None)
            if sp is None:
                questions[p] = ctx.question(p, original=original, reason=pv.reason or "which date?")
                continue
            part = pv.part if pv.part in ("start", "end") else dates.date_role(p)
            value = dates.span_value(sp, part, ctx.date_fmt(p))
            if value is None:
                questions[p] = ctx.question(p, original=original, reason=f'"{sp["phrase"]}" is a range; which date?')
                continue
            note_src = f'from "{sp["phrase"]}"'
        else:
            if pv.value is None:
                continue
            value, note_src = pv.value, pv.reason
        res = ctx.check(p, value)
        if not res.ok:
            questions[p] = ctx.question(p, original=original, reason=res.reason or "not valid")
            continue
        if not pv.certain:
            questions[p] = ctx.question(p, proposed=res.value, original=original, reason=pv.reason or "please confirm")
            continue
        if original is not None:
            params[p] = decision(p, res.value, "user_corrected", original_value=original,
                                 note=pv.reason or failing[p]["reason"], fixed_by="llm")
        else:
            params[p] = decision(p, res.value, "inferred", note=note_src, fixed_by="llm")

    for uk in proposal.unmapped_user_keys:
        if uk.key not in unmapped:
            continue
        target = uk.mapped_to
        if target in ctx.accepted and target not in params and target not in questions:
            handled.add(uk.key)
            val = unmapped[uk.key]
            res = ctx.check(target, val)
            if res.ok:
                params[target] = decision(target, res.value, "user_corrected", original_value=f"{uk.key}={val}",
                                          note=f'treated "{uk.key}" as {target}', fixed_by="llm")
            else:
                questions[target] = ctx.question(target, original=val, reason=res.reason or "not valid")
        else:
            ignored.append({"key": uk.key, "reason": uk.reason or "not a parameter of this view"})


def route_after_resolve_params(state: dict) -> str:
    return "ask_param" if state.get("pending_questions") else "finalize_params"


# --------------------------------------------------------------------------- questions


def _param_question(q: dict) -> str:
    name = f"**{q['name']}**"
    lines = []
    if q.get("retry_reason"):
        lines.append(f"`{q.get('retry_value')}` didn't work for {name}: {q['retry_reason']}.")
    if q.get("original") is not None and q.get("proposed") is not None:
        lines.append(f"For {name} you wrote `{q['original']}` ({q['reason']}). Did you mean `{q['proposed']}`?")
    elif q.get("proposed") is not None:
        lines.append(f"For {name} I'd use `{q['proposed']}` ({q['reason']}). Is that right?")
    elif q.get("original") is not None:
        lines.append(f"For {name} you wrote `{q['original']}`, but {q['reason']}. What value should I use?")
    else:
        lines.append(f"What value should I use for {name}? ({q['reason']})")
    if q.get("options"):
        more = q.get("n_options", 0) - len(q["options"])
        lines.append("Options: " + ", ".join(f"`{o}`" for o in q["options"]) + (f" (+{more} more)" if more > 0 else ""))
    if q.get("format_hint"):
        lines.append(f"Format: {q['format_hint']}")
    if q.get("proposed") is not None:
        lines.append("Reply yes to accept, or give a value.")
    return "\n".join(lines)


def ask_param(state: dict, deps: Deps) -> dict:
    q = state["pending_questions"][0]
    return {"messages": [stream_text(deps.streamer, _param_question(q))]}


def confirm_param(state: dict, deps: Deps) -> dict:
    pending = list(state.get("pending_questions") or [])
    q = pending[0]
    answer = interrupt({
        "type": "confirm_param", "name": q["name"], "proposed": q.get("proposed"),
        "original": q.get("original"), "reason": q.get("reason"), "options": q.get("options"),
        "format_hint": q.get("format_hint"), "message": _param_question(q),
    })
    text = answer_text(answer)
    human = HumanMessage(content=text)
    params = dict(state.get("params") or {})
    invalid = list(state.get("invalid_params") or [])
    ctx = _ParamCtx(state, deps)
    p = q["name"]

    accept = (isinstance(answer, dict) and answer.get("accept")) or text.strip().lower().strip("!.") in YES
    if accept and q.get("proposed") is not None:
        ok, value, reason, raw = True, q["proposed"], None, q["proposed"]
    elif accept:
        ok, value, reason, raw = False, None, "please give a value", text
    else:
        raw = answer.get("value", text) if isinstance(answer, dict) else text
        res = ctx.check(p, raw)
        ok, value, reason = res.ok, res.value, res.reason

    if ok:
        params[p] = decision(p, value, "user_confirmed", original_value=q.get("original"),
                             note=None if accept else f"you gave {raw}", fixed_by="user")
        deps.learning.submit(LearningEvent(
            kind="param_confirmed" if accept else "param_corrected_by_user",
            request_text=state.get("request_text", ""), view=state.get("view"), param=p,
            original_value=q.get("original") if q.get("original") is not None else q.get("proposed"),
            final_value=value,
        ))
        pending = pending[1:]
    else:
        q2 = {**q, "reasks": q.get("reasks", 0) + 1, "retry_reason": reason, "retry_value": raw}
        if q2["reasks"] > deps.settings.agent.max_param_reasks:
            invalid.append({"name": p, "value": raw, "reason": reason})
            pending = pending[1:]
        else:
            pending = [q2] + pending[1:]
    return {"messages": [human], "params": params, "pending_questions": pending, "invalid_params": invalid}


def route_after_confirm_param(state: dict) -> str:
    return "ask_param" if state.get("pending_questions") else "finalize_params"


def finalize_params(state: dict, deps: Deps) -> dict:
    spec = state.get("param_spec") or {}
    params = state.get("params") or {}
    accepted = set(spec.get("mandatory", [])) | set(spec.get("optional", []))
    problems = list(state.get("invalid_params") or [])
    for p in spec.get("mandatory", []):
        if p not in params and not any(x["name"] == p for x in problems):
            problems.append({"name": p, "value": None, "reason": "missing (mandatory)"})
    for p, d in params.items():
        allowed = (spec.get("allowed") or {}).get(p) or []
        if p not in accepted:
            problems.append({"name": p, "value": d["value"], "reason": "not accepted by this view"})
        elif allowed and d["value"] not in allowed:
            problems.append({"name": p, "value": d["value"], "reason": "not an accepted value"})
    if problems:
        return {"error": {"kind": "invalid_params", "message": "Some parameters are missing or invalid.",
                          "detail": problems}}
    return {}


def route_after_finalize(state: dict) -> str:
    return "respond" if state.get("error") else "fetch"
