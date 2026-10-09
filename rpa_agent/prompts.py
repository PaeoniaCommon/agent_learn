"""Structured-output schemas and prompts for agent and learner LLM calls."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- schemas


class ViewCandidate(BaseModel):
    view: str = Field(description="Exact view name from the list")
    reason: str = Field(description="Short reason (<= 15 words)")
    confidence: Literal["high", "medium", "low"]


class ViewChoice(BaseModel):
    candidates: list[ViewCandidate] = Field(description="Plausible views, best first; empty if none fit")
    needs_confirmation: bool = Field(description="True if more than one view is genuinely plausible")


class ViewAnswer(BaseModel):
    view: str | None = Field(description="The option the user chose, exactly as listed, or null if unclear")


class ParamValue(BaseModel):
    name: str = Field(description="Exact param name from the list")
    value: str | None = Field(None, description="Value for non-date params. Never a computed date.")
    span: str | None = Field(None, description="For date params: id of a resolved date span, e.g. 'd1'")
    part: Literal["start", "end", "single"] | None = Field(None, description="Which part of the span")
    certain: bool = Field(description="False if you are guessing or the request is ambiguous")
    reason: str = Field(description="Short reason (<= 15 words)")


class UnmappedKey(BaseModel):
    key: str
    mapped_to: str | None = Field(None, description="Param name it should map to, or null")
    reason: str = ""


class ParamProposal(BaseModel):
    params: list[ParamValue] = Field(default_factory=list)
    unmapped_user_keys: list[UnmappedKey] = Field(default_factory=list)


class ViewLearn(BaseModel):
    description: str
    changed: bool
    conflicts: bool = Field(False, description="True if the evidence contradicts the existing description")


class ParamLearn(BaseModel):
    description: str | None = None
    format: str | None = None
    pattern: str | None = Field(None, description="Python regex fully matching valid values, or null")
    invalid_examples: list[str] = Field(default_factory=list,
                                        description="value (short backend reason)")
    changed: bool
    conflicts: bool = False


# --------------------------------------------------------------------------- agent prompts

VIEW_SYSTEM = """You choose which data view answers a user's request.
Views are listed as `- name: description`. A view with no description must be judged by its name.
Rules:
- Only use view names exactly as listed. Never invent views.
- Mark confidence high only when one view clearly fits.
- Set needs_confirmation=true when two or more views are genuinely plausible."""

VIEW_USER = """Request: {request}
{hint}{last}
<views>
{views}
</views>"""

VIEW_ANSWER_SYSTEM = """The user was asked to choose a view from numbered options. Map their reply to one
option name exactly as listed, or null if the reply doesn't clearly pick one."""

PARAM_SYSTEM = """You set parameter values for a data request.
Rules:
- Only use param names listed. Never invent params.
- Params marked FIXED are already decided: do not return them.
- For params with an allowed list, the value must be from that list.
- Dates: NEVER write a date value yourself. For a date param, set `span` to the id of a resolved
  date span and `part` to start, end or single. If no span fits, leave it out.
- Only set optional params the request actually mentions.
- certain=false whenever you are guessing or the request is ambiguous.
- For each unmapped user key, give the param it most likely means (mapped_to) or null."""

PARAM_USER = """Request: {request}
View: {view}
{last}
Params (name [mandatory|optional] allowed=...):
{params}
{knowledge}
Resolved date spans:
{spans}
FIXED (do not change):
{fixed}
Need values / fixing:
{failing}
Unmapped user keys: {unmapped}"""

# --------------------------------------------------------------------------- learner prompts

LEARN_VIEW_SYSTEM = """You maintain a one-line description of a data view used to pick the right view
for future requests. Shared by all users: record general facts only, never one user's preferences.
Keep correct existing info, integrate the new evidence, and prefer distinguishing facts (content,
grain, when to use, `(→ other_view)` for easily confused views). Telegraphic style, max {max_chars} chars.
If the evidence adds nothing, return the existing text with changed=false.
Set conflicts=true if the evidence contradicts the existing description."""

LEARN_PARAM_SYSTEM = """You maintain notes about one request parameter, used to fill and format it in
future requests. Shared by all users: general facts only, no single-request values except as examples.
Fields: description (<= {desc_chars} chars, only if the name alone isn't clear), format (<= {fmt_chars}
chars, concrete and actionable), pattern (Python regex that fully matches every known valid value, or
null), invalid_examples (max {max_examples}, "value (short backend reason)").
{enum_note}Return null for fields you would not change. If the evidence adds nothing, changed=false.
Set conflicts=true if the evidence contradicts existing notes."""

LEARN_USER = """Current entry:
{current}
New evidence:
{evidence}
Recent history:
{history}"""
