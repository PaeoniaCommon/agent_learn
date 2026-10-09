"""Background learning: events in, lean knowledge-file updates out.

Graph nodes only call `LearningWorker.submit(event)` (non-blocking). A single daemon thread
applies the learning gate (spec §9.2): most events stop at cheap code checks; only events
showing friction or a knowledge gap reach the learning LLM, batched per target file.
"""

from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent_core.dates import infer_formats, looks_like_date_param

from . import prompts
from .backend import Backend
from .config import Settings
from .knowledge import KnowledgeBase, ParamKnowledge
from .validation import mentioned_params

log = logging.getLogger(__name__)

EventKind = Literal["view_confirmed", "view_corrected", "param_confirmed", "param_corrected_by_user",
                    "param_llm_fix_succeeded", "fetch_success", "fetch_error"]


class LearningEvent(BaseModel):
    kind: EventKind
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_text: str = ""
    view: str | None = None
    rejected_views: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    param: str | None = None
    original_value: Any = None
    final_value: Any = None
    error_message: str | None = None
    columns: list[str] = Field(default_factory=list)


def _words(s: str) -> set[str]:
    return set(re.findall(r"\w+", (s or "").lower()))


def _budget(text: str | None, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    return cut.rsplit(" ", 1)[0] if " " in cut else cut


def _adds_nothing(new: str, old: str) -> bool:
    """A rewrite that is not shorter and introduces no new words adds nothing."""
    return bool(old) and _words(new) <= _words(old) and len(new) >= len(old)


class EventLog:
    """Append-only JSONL evidence log (never sent to the LLM in the request path)."""

    def __init__(self, path: Path, tail_lines: int = 5000):
        self.path = path
        self.tail_lines = tail_lines
        self._lock = threading.Lock()

    def append(self, record: dict) -> None:
        line = json.dumps(record, default=str)
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def tail(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self._lock, open(self.path, encoding="utf-8") as fh:
            lines = deque(fh, maxlen=self.tail_lines)
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return out

    def success_values(self, param: str) -> list[str]:
        return [str(r["params"][param]) for r in self.tail()
                if r.get("type") == "event" and r.get("kind") == "fetch_success" and param in (r.get("params") or {})]

    def history(self, target: tuple[str, str], n: int) -> list[dict]:
        kind, name = target
        recs = []
        for r in self.tail():
            if r.get("type") != "event":
                continue
            if kind == "view" and r.get("view") == name and r["kind"] in ("view_confirmed", "view_corrected"):
                recs.append({"kind": r["kind"], "request": r.get("request_text", "")[:120],
                             "rejected": r.get("rejected_views")})
            elif kind == "param" and (r.get("param") == name or
                                      (r["kind"] == "fetch_error" and name in (r.get("params") or {}))):
                rec = {k: r.get(k) for k in ("kind", "original_value", "final_value", "error_message")
                       if r.get(k) is not None}
                if r["kind"] == "fetch_error":
                    rec["value"] = r["params"][name]
                recs.append(rec)
        return recs[-n:]

    def conflict_count(self, target: tuple[str, str]) -> int:
        n = 0
        for r in self.tail():
            if r.get("target") == list(target):
                if r.get("type") == "conflict":
                    n += 1
                elif r.get("type") == "change":
                    n = 0  # an applied change resets pending conflicts
        return n


class LearningWorker:
    def __init__(self, kb: KnowledgeBase, llm, backend: Backend, settings: Settings, start: bool = True):
        self.kb = kb
        self.llm = llm
        self.backend = backend
        self.cfg = settings.learning
        self.date_cfg = settings.dates
        self.log = EventLog(kb.log_dir / "events.jsonl")
        self.enabled = self.cfg.enabled
        self.llm_calls = 0
        self._q: queue.Queue[LearningEvent | None] = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        if start and self.enabled:
            self.start()

    # -- public API -------------------------------------------------------------
    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._run, name="rpa-learning", daemon=True)
            self._thread.start()

    def submit(self, event: LearningEvent) -> None:
        if self.enabled:
            self._q.put_nowait(event)

    def flush(self, timeout: float | None = None) -> bool:
        """Block until all submitted events are processed (for tests / shutdown)."""
        if not self.enabled:
            return True
        end = None if timeout is None else time.monotonic() + timeout
        while self._q.unfinished_tasks:
            if end is not None and time.monotonic() > end:
                return False
            time.sleep(0.01)
        return True

    def shutdown(self, wait: bool = True, timeout: float | None = 30) -> None:
        if wait:
            self.flush(timeout)
        self._stop.set()
        self._q.put_nowait(None)
        if self._thread and wait:
            self._thread.join(timeout)

    # -- worker loop ----------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            batch = [first]
            deadline = time.monotonic() + self.cfg.debounce_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._q.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                self.process([e for e in batch if e is not None])
            except Exception as exc:  # learning must never affect users
                log.warning("learning batch failed: %s", exc)
                log.debug("learning failure detail", exc_info=True)
            finally:
                for _ in batch:
                    self._q.task_done()

    # -- processing -------------------------------------------------------------------
    def process(self, events: list[LearningEvent]) -> None:
        tasks: dict[tuple[str, str], list[dict]] = {}
        for ev in events:
            self.log.append({"type": "event", **ev.model_dump(mode="json")})
            try:
                self._gate(ev, tasks)
            except Exception:
                log.exception("learning gate failed for %s", ev.kind)
        for target, evidence in tasks.items():
            try:
                if target[0] == "view":
                    self._learn_view(target[1], evidence)
                else:
                    self._learn_param(target[1], evidence)
            except Exception as exc:
                log.warning("learning failed for %s: %s", target, exc)
                log.debug("learning failure detail", exc_info=True)

    def _is_enum(self, param: str) -> bool:
        try:
            return bool(self.backend.allowed_values(param))
        except Exception:
            return False

    def _gate(self, ev: LearningEvent, tasks: dict) -> None:
        """Decide in code whether an event is worth an LLM call. Applies code-only updates."""
        add = lambda target, item: tasks.setdefault(target, []).append(item)  # noqa: E731

        if ev.kind in ("view_confirmed", "view_corrected") and ev.view:
            current = self.kb.views().get(ev.view, "")
            req = _words(ev.request_text)
            # skip if the description already covers the request's distinguishing words
            if current and len(req & _words(current)) >= max(2, len(req) // 2):
                return
            add(("view", ev.view), {"signal": "user chose this view" if ev.kind == "view_confirmed"
                                    else "user corrected to this view",
                                    "request": ev.request_text[:200], "rejected": ev.rejected_views})

        elif ev.kind == "fetch_success" and ev.view:
            if not self.kb.views().get(ev.view):
                add(("view", ev.view), {"signal": "first successful use", "request": ev.request_text[:200],
                                        "columns": ev.columns[:25]})
            for p, v in ev.params.items():
                self._code_learn_success(p, v)

        elif ev.kind == "fetch_error" and ev.error_message:
            msg = ev.error_message
            for p in mentioned_params(msg, ev.params):
                if self._is_enum(p):
                    continue
                pk = self.kb.param(p)
                val = str(ev.params[p])
                if pk.date_format and val not in pk.valid_examples and not any(
                        pk.date_format in infer_formats(s) for s in self.log.success_values(p)):
                    self.kb.update_param(p, lambda k: bool(k.date_format) and not setattr(k, "date_format", ""))
                if any(ex.split(" (")[0] == val for ex in pk.invalid_examples):
                    continue  # already recorded
                add(("param", p), {"signal": "backend rejected value", "value": val, "error": msg[:200]})

        elif ev.kind in ("param_confirmed", "param_corrected_by_user") and ev.param:
            p = ev.param
            pk = self.kb.param(p)
            if self._is_enum(p):
                if not pk.description and ev.kind == "param_corrected_by_user":
                    add(("param", p), {"signal": "user corrected", "user_wrote": ev.original_value,
                                       "final": ev.final_value, "request": ev.request_text[:200], "enum": True})
                return
            final = str(ev.final_value)
            if pk.description and pk.format and pk.pattern and _safe_fullmatch(pk.pattern, final):
                return
            add(("param", p), {"signal": ev.kind.replace("_", " "), "user_wrote": ev.original_value,
                               "final": ev.final_value, "request": ev.request_text[:200]})

        elif ev.kind == "param_llm_fix_succeeded" and ev.param:
            p = ev.param
            if self._is_enum(p):
                return
            pk = self.kb.param(p)
            if pk.format and str(ev.final_value) in pk.valid_examples:
                return
            add(("param", p), {"signal": "value fixed then accepted by backend",
                               "original": ev.original_value, "final": ev.final_value})

    def _code_learn_success(self, param: str, value: Any) -> None:
        enum = self._is_enum(param)
        sval = str(value)
        max_ex = self.cfg.max_examples
        default_fmt = self.date_cfg.default_date_format

        def mutate(pk: ParamKnowledge) -> bool:
            changed = False
            if enum:
                return False  # the allowed list is authoritative; store nothing
            if pk.kind == "date" or (pk.kind == "" and looks_like_date_param(param)):
                fmts = infer_formats(sval)
                if fmts:
                    if pk.kind != "date":
                        pk.kind, changed = "date", True
                    if not pk.date_format:
                        choice = default_fmt if default_fmt in fmts else (fmts[0] if len(fmts) == 1 else None)
                        if choice:
                            pk.date_format, changed = choice, True
                elif pk.kind == "":
                    pk.kind, changed = "text", True
            if pk.pattern and not _safe_fullmatch(pk.pattern, sval):
                pk.pattern, changed = "", True  # a pattern must match every value that worked
            if len(pk.valid_examples) < max_ex and sval not in pk.valid_examples and len(sval) <= 60:
                pk.valid_examples.append(sval)
                changed = True
            return changed

        self.kb.update_param(param, mutate)

    # -- LLM learning ---------------------------------------------------------------
    def _conflict_blocks(self, target: tuple[str, str], fields: dict) -> bool:
        """Contradictions need `conflict_threshold` agreeing signals before they apply."""
        pending = self.log.conflict_count(target)
        if pending + 1 < self.cfg.conflict_threshold:
            self.log.append({"type": "conflict", "target": list(target), "fields": fields,
                             "ts": datetime.now(timezone.utc).isoformat()})
            return True
        return False

    def _learn_view(self, view: str, evidence: list[dict]) -> None:
        current = self.kb.views().get(view, "")
        target = ("view", view)
        others = {v: d for v, d in self.kb.views().items()
                  if any(v in (e.get("rejected") or []) for e in evidence)}
        user = prompts.LEARN_USER.format(
            current=f"{view}: {current or '(empty)'}",
            evidence=json.dumps(evidence, default=str),
            history=json.dumps(self.log.history(target, self.cfg.recent_evidence_lines), default=str)
            + ("\nConfusable views: " + json.dumps(others) if others else ""),
        )
        self.llm_calls += 1
        out = self.llm.structured(prompts.ViewLearn,
                                  prompts.LEARN_VIEW_SYSTEM.format(max_chars=self.cfg.view_desc_max_chars),
                                  user, step="learn_view")
        if not out.changed:
            return
        new = _budget(out.description, self.cfg.view_desc_max_chars)
        if not new or new == current or _adds_nothing(new, current):
            return
        if out.conflicts and current and self._conflict_blocks(target, {"description": new}):
            return
        self.kb.set_view_description(view, new)
        self.log.append({"type": "change", "target": list(target), "fields": {"description": new},
                         "ts": datetime.now(timezone.utc).isoformat()})

    def _learn_param(self, param: str, evidence: list[dict]) -> None:
        pk = self.kb.param(param)
        enum = self._is_enum(param) or any(e.get("enum") for e in evidence)
        target = ("param", param)
        enum_note = ("This param has a fixed list of allowed values that is authoritative. Only maintain "
                     "the description; return null for format, pattern and invalid_examples.\n") if enum else ""
        system = prompts.LEARN_PARAM_SYSTEM.format(
            desc_chars=self.cfg.param_desc_max_chars, fmt_chars=self.cfg.param_format_max_chars,
            max_examples=self.cfg.max_examples, enum_note=enum_note)
        user = prompts.LEARN_USER.format(
            current=pk.render(), evidence=json.dumps(evidence, default=str),
            history=json.dumps(self.log.history(target, self.cfg.recent_evidence_lines), default=str))
        self.llm_calls += 1
        out = self.llm.structured(prompts.ParamLearn, system, user, step="learn_param")
        if not out.changed:
            return

        updates: dict[str, Any] = {}
        if out.description:
            d = _budget(out.description, self.cfg.param_desc_max_chars)
            if d != pk.description and not _adds_nothing(d, pk.description):
                updates["description"] = d
        if not enum:
            if out.format:
                f = _budget(out.format, self.cfg.param_format_max_chars)
                if f != pk.format and not _adds_nothing(f, pk.format):
                    updates["format"] = f
            if out.pattern and out.pattern != pk.pattern:
                known_good = set(self.log.success_values(param)) | set(pk.valid_examples)
                try:
                    re.compile(out.pattern)
                    if all(_safe_fullmatch(out.pattern, v) for v in known_good):
                        updates["pattern"] = out.pattern
                except re.error:
                    pass
            if out.invalid_examples:
                merged = list(dict.fromkeys(pk.invalid_examples + [_budget(x, 80) for x in out.invalid_examples]))
                merged = merged[-self.cfg.max_examples:]
                if merged != pk.invalid_examples:
                    updates["invalid_examples"] = merged
        if not updates:
            return
        if out.conflicts and not pk.is_empty() and self._conflict_blocks(target, updates):
            return

        def mutate(k: ParamKnowledge) -> bool:
            for key, val in updates.items():
                setattr(k, key, val)
            return True

        self.kb.update_param(param, mutate)
        self.log.append({"type": "change", "target": list(target), "fields": updates,
                         "ts": datetime.now(timezone.utc).isoformat()})


def _safe_fullmatch(pattern: str, value: str) -> bool:
    try:
        return re.fullmatch(pattern, value) is not None
    except re.error:
        return True
